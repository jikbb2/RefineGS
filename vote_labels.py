#!/usr/bin/env python3
"""Assign an instance label to every gaussian by multi-view voting.

Why voting rather than (only) learning: the label CE supervises the ALPHA-COMPOSITED
embedding along a ray, so many object-scale assignments give the same low CE and SGD
has no reason to pick the right one. Measured after 30k iters from a random init:
CE 0.02-0.4 and locally smooth embeddings (knn 0.02), yet only 6 of 34 classes were
spatially compact. Voting reads the assignment straight off the 3-D structure.

Visibility uses GT depth, the same test as make_shaper_input.py: a gaussian counts
in a view only if it is the first surface there (|z - d_gt| < margin). Without it,
points on the far side of an object collect votes through the object.

Output: labels.npy (int32, one class per gaussian) plus a vote-margin array that
says how contested each gaussian was.

  python vote_labels.py \
      --ply OUT/scene/point_cloud/iteration_30000/point_cloud.ply \
      --colmap DATA/sparse/0 --label_dir DATA/labels_scene \
      --gt_depth_dir GTDEPTH --out OUT/scene/vote
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warp_gt_to_pose import read_colmap                       # noqa: E402

IGNORE = 65535


def load_xyz(path):
    p = PlyData.read(os.path.expanduser(path))["vertex"]
    return np.stack([p[k] for k in ("x", "y", "z")], 1).astype(np.float32)


def load_png(path, scale=1.0, dtype=np.float32):
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(dtype) / scale if scale != 1.0 else a.astype(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--gt_depth_dir", default="", help="omit to skip the occlusion test")
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--margin", type=float, default=0.03,
                    help="|z - d_gt| tolerance (m) for 'first surface in this view'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=200000, help="gaussians per projection chunk")
    args = ap.parse_args()

    xyz = load_xyz(args.ply)
    N = len(xyz)
    meta = json.load(open(os.path.join(os.path.expanduser(args.label_dir), "id_map.json")))
    K = int(meta["K"])
    lab_dir = os.path.join(os.path.expanduser(args.label_dir), "labels")
    have = {os.path.splitext(f)[0] for f in os.listdir(lab_dir)}
    cams = [c for c in read_colmap(args.colmap) if c["stem"] in have]
    print(f"[vote] {N:,} gaussians, K={K}, {len(cams)} views with labels")
    assert cams, "no camera stem matches a label map"

    votes = np.zeros((N, K + 1), np.int32)
    n_vis_tot = 0
    for vi, c in enumerate(cams):
        lab = load_png(os.path.join(lab_dir, c["stem"] + ".png"), dtype=np.int32)
        H, W = lab.shape
        sx, sy = W / c["W"], H / c["H"]                       # label map may be resized
        d = None
        if args.gt_depth_dir:
            for nm in (c["stem"].replace("frame", "depth"), c["stem"]):
                p = os.path.join(os.path.expanduser(args.gt_depth_dir), nm + ".png")
                if os.path.isfile(p):
                    d = load_png(p, args.gt_depth_scale)
                    if d.shape != (H, W):
                        d = np.asarray(Image.fromarray(d).resize((W, H), Image.NEAREST))
                    break

        for i in range(0, N, args.chunk):
            X = xyz[i:i + args.chunk]
            Xc = X @ c["R"].T + c["t"]
            z = Xc[:, 2]
            ok = z > 0.05
            u = (c["fx"] * Xc[:, 0] / np.maximum(z, 1e-6) + c["cx"]) * sx
            v = (c["fy"] * Xc[:, 1] / np.maximum(z, 1e-6) + c["cy"]) * sy
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            ui = np.clip(u, 0, W - 1).astype(np.int32)
            vi_ = np.clip(v, 0, H - 1).astype(np.int32)
            if d is not None:                                  # occlusion: first surface only
                dz = d[vi_, ui]
                ok &= (dz > 0.01) & (np.abs(z - dz) < args.margin)
            L = lab[vi_, ui]
            ok &= (L != IGNORE)
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            np.add.at(votes, (idx + i, L[idx]), 1)
            n_vis_tot += len(idx)
        if (vi + 1) % 100 == 0:
            print(f"  {vi + 1}/{len(cams)} views")

    top2 = np.partition(votes, -2, axis=1)[:, -2:]
    tot = votes.sum(1)
    labels = votes.argmax(1).astype(np.int32)
    labels[tot == 0] = -1                                      # never seen -> unassigned
    # margin = how decisive the vote was; low means two labels fought over it
    marg = np.where(tot > 0, (top2[:, 1] - top2[:, 0]) / np.maximum(tot, 1), 0.0)

    od = os.path.expanduser(args.out)
    os.makedirs(od, exist_ok=True)
    np.save(os.path.join(od, "labels.npy"), labels)
    np.save(os.path.join(od, "margin.npy"), marg.astype(np.float32))
    np.save(os.path.join(od, "votes_total.npy"), tot.astype(np.int32))

    unass = int((labels < 0).sum())
    print(f"\n[vote] {n_vis_tot / max(len(cams), 1):,.0f} visible gaussians per view")
    print(f"       unassigned (never visible) {unass:,} ({unass / N * 100:.1f}%)")
    print(f"       vote margin: median {np.median(marg[labels >= 0]):.2f}  "
          f"below 0.2 -> {(marg[labels >= 0] < 0.2).mean() * 100:.0f}% contested")
    print(f"\n{'class':>6}{'gaussians':>11}{'share':>8}")
    for c in range(-1, K + 1):
        n = int((labels == c).sum())
        if n:
            print(f"{c:>6}{n:>11,}{n / N * 100:>7.1f}%"
                  + ("  unassigned" if c < 0 else "  BG" if c == 0 else ""))
    print(f"\n-> {od}/labels.npy   check with check_scene_labels.py --labels")


if __name__ == "__main__":
    main()
