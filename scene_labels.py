"""Label supervision for scene-level training.

The renderer already rasterizes gaussian _id as a color:
    E = clamp(C0 * _id + 0.5, 0, 1)      C0 ~ 0.2821,  render_pkg["mask"], (3,H,W)
So with _id made learnable (GaussianModel.enable_label_learning) we get a
differentiable 3-D embedding per pixel for free -- no rasterizer change.

LabelHead turns that into class logits against learnable prototypes:
    logits[c] = -||E - P_c||^2 / T
Prototypes are kept inside [lo,hi]^3 so they stay in the interior of the clamp
range; a prototype at the boundary would sit where gradients die.

Multi-view consistency is structural: one gaussian has one embedding, so a label
that flickers across views is averaged by the optimizer instead of being trusted
per view.

  from scene_labels import LabelLoader, LabelHead
  loader = LabelLoader(args.label_dir)
  head   = LabelHead(loader.K).cuda()
  ...
  tgt = loader.target(cam.image_name, H, W)          # (H,W) long, IGNORE=65535
  loss_lab = head.loss(render_pkg["mask"], tgt)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IGNORE = 65535


class LabelLoader:
    """Reads labels/<stem>.png (uint16) and union/<stem>.png written by make_label_maps.py."""

    def __init__(self, root, cache=True):
        self.root = os.path.expanduser(root)
        meta = json.load(open(os.path.join(self.root, "id_map.json")))
        self.K = int(meta["K"])                 # instance classes; +1 for background
        self.n_class = self.K + 1
        self.meta = meta
        self._c = {} if cache else None
        print(f"[label] K={self.K} (+background) from {self.root}")

    def _read(self, stem):
        if self._c is not None and stem in self._c:
            return self._c[stem]
        lp = os.path.join(self.root, "labels", stem + ".png")
        up = os.path.join(self.root, "union", stem + ".png")
        lab = np.array(Image.open(lp)).astype(np.int32) if os.path.isfile(lp) else None
        uni = (np.array(Image.open(up)) > 0).astype(np.float32) if os.path.isfile(up) else None
        if self._c is not None:
            self._c[stem] = (lab, uni)
        return lab, uni

    @staticmethod
    def _resize(a, H, W, nearest):
        if a is None or a.shape[:2] == (H, W):
            return a
        m = Image.NEAREST if nearest else Image.BILINEAR
        return np.array(Image.fromarray(a).resize((W, H), m))

    def target(self, stem, H, W):
        """(H,W) int64 label map on cuda. Missing view -> all IGNORE (loss skips it)."""
        lab, _ = self._read(stem)
        if lab is None:
            return torch.full((H, W), IGNORE, dtype=torch.long, device="cuda")
        lab = self._resize(lab.astype(np.int32), H, W, True)
        return torch.from_numpy(lab.astype(np.int64)).cuda()

    def union(self, stem, H, W):
        """(1,H,W) float alpha target, or None if the view has no mask."""
        _, uni = self._read(stem)
        if uni is None:
            return None
        uni = self._resize(uni, H, W, False)
        return torch.from_numpy(np.ascontiguousarray(uni)).float().cuda()[None]


class LabelHead(nn.Module):
    """Learnable prototypes over the rendered 3-D embedding."""

    def __init__(self, K, lo=0.15, hi=0.85, temp=0.1, seed=0):
        super().__init__()
        C = K + 1                                # class 0 = background
        # Farthest-point init: random placement leaves close pairs that are hard to
        # separate (measured min dist 0.077 for C=35 in a 0.7 cube; FPS gives ~0.19).
        g = torch.Generator().manual_seed(seed)
        pool = lo + (hi - lo) * torch.rand(4096, 3, generator=g)
        idx = [0]
        d = (pool - pool[0]).pow(2).sum(1)
        for _ in range(C - 1):
            i = int(d.argmax())
            idx.append(i)
            d = torch.minimum(d, (pool - pool[i]).pow(2).sum(1))
        p = pool[idx]
        # store in logit space so the sigmoid below keeps prototypes inside [lo,hi]
        self.raw = nn.Parameter(torch.logit((p - lo) / (hi - lo)))
        self.log_t = nn.Parameter(torch.tensor(float(np.log(temp))))
        self.lo, self.hi = lo, hi

    @property
    def proto(self):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw)   # (C,3)

    def logits(self, E):
        """E: (3,H,W) rendered embedding -> (C,H,W) logits."""
        d2 = ((E[None] - self.proto[:, :, None, None]) ** 2).sum(1)      # (C,H,W)
        return -d2 / self.log_t.exp().clamp_min(1e-3)

    def loss(self, E, target):
        lg = self.logits(E)
        return F.cross_entropy(lg[None], target[None], ignore_index=IGNORE)

    @torch.no_grad()
    def assign(self, ids):
        """Per-gaussian class from raw _id (N,3). Uses the same C0 mapping as the renderer."""
        C0 = 0.28209479177387814
        E = (C0 * ids + 0.5).clamp(0, 1)                                 # (N,3)
        d2 = ((E[:, None] - self.proto[None]) ** 2).sum(-1)              # (N,C)
        return d2.argmin(1)

    @torch.no_grad()
    def report(self, E):
        """Diagnostics: clamp saturation kills gradients, prototype spread must stay > 0."""
        sat = ((E <= 1e-4) | (E >= 1 - 1e-4)).float().mean().item()
        P = self.proto
        d = torch.cdist(P, P) + torch.eye(len(P), device=P.device) * 9
        return {"sat": sat, "proto_min_dist": d.min().item(),
                "temp": self.log_t.exp().item()}