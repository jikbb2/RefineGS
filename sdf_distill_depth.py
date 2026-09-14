#!/usr/bin/env python3
"""RefineGS - SDF distillation from rendered depth (replaces Open3D TSDF).

Same inputs and options as render.py's TSDF fusion, but the watertight mesh comes
from an implicit SDF (IGR-style) instead of Open3D. Pipeline:

  1) render() per view -> surf_depth, rend_normal, rend_alpha, rgb
  2) back-project depth to world points with to_cam_open3d's intrinsics/extrinsics.
     Normals are flipped toward the camera so the SDF sign is globally consistent
     (the root cause of the sponge artefacts in the old sdf_distill.py)
  3) fit an IGR SDF MLP (manifold + normal + eikonal + signed off-surface)
  4) evaluate on a grid, keep observed voxels only (drops the box of unobserved empty
     space; small holes are filled by interpolation)
     -> marching cubes(zero level set)
  5) safe_post_process_mesh for num_cluster (same logic as the TSDF path, clamped)

Run from the RefineGS repo root, next to render.py:

  python sdf_distill_depth.py -m output/replica_room0_v2/scene_whole_orbit -s data/replica_room0_v2 \
    --iteration 7000 --depth_ratio 0 --depth_trunc 6.0 --voxel_size 0.01 \
    --sdf_trunc 0.04 --num_cluster 10000 \
    --sdf_iters 10000 --pts_per_view 40000

  # option mapping against the render.py TSDF command above:
  #   --depth_ratio, --depth_trunc  : passed straight to render() / back-projection
  #   --voxel_size                  : marching-cubes resolution (2*scale/voxel_size)
  #   --sdf_trunc                   : unused on the SDF path; masking is --mask_dist
  #   --num_cluster                 : reused by safe_post_process_mesh (count clamped)
"""
import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
import open3d as o3d

# Default GT depth path. Override with REFINEGS_GT_DEPTH; a missing folder is ignored
# silently so the script still runs on another machine.
DEFAULT_GT_DEPTH_DIR = "/home/elicer/nice-slam/Datasets/Replica/room0/results"


# ---------------------------------------------------------------------------
# Camera intrinsics/extrinsics - exactly the to_cam_open3d (mesh_utils.py) convention.
# ---------------------------------------------------------------------------
def cam_intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    if hasattr(cam, "projection_matrix"):
        ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2],
                                [0, H / 2, 0, (H - 1) / 2],
                                [0, 0, 0, 1]]).float().cuda().T
        intrins = (cam.projection_matrix @ ndc2pix)[:3, :3].T
        fx, fy = intrins[0, 0].item(), intrins[1, 1].item()
        cx, cy = intrins[0, 2].item(), intrins[1, 2].item()
    else:  # MiniCam (extra_poses): derive from FoV
        fx = W / (2.0 * np.tan(cam.FoVx / 2))
        fy = H / (2.0 * np.tan(cam.FoVy / 2))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    extrinsic = cam.world_view_transform.T  # world->camera (w2c), CV convention (+Z forward)
    return fx, fy, cx, cy, W, H, extrinsic


# ---------------------------------------------------------------------------
# Safe version of utils.mesh_utils.post_process_mesh. The original indexes
# sorted[-cluster_to_keep] and raises IndexError when there are fewer components than
# num_cluster -- which actually happens on clean SDF meshes. Clamp to the count.
# ---------------------------------------------------------------------------
def safe_post_process_mesh(mesh, cluster_to_keep=1000):
    print(f"post processing the mesh to have {cluster_to_keep} clusters (clamped)")
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    keep = min(cluster_to_keep, len(cluster_n_triangles))  # clamp (fixes the original bug)
    n_cluster = np.sort(cluster_n_triangles.copy())[-keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


# ---------------------------------------------------------------------------
# IGR-style SDF MLP (geometric init). PE off by default: it causes off-surface ringing.
# ---------------------------------------------------------------------------
class SDFNet(nn.Module):
    def __init__(self, d_hidden=256, n_layers=8, skip_in=(4,), pe_L=0, radius=0.5):
        super().__init__()
        self.pe_L = pe_L
        d_in = 3 + 3 * 2 * pe_L
        self.d_in = d_in
        dims = [d_in] + [d_hidden] * n_layers + [1]
        self.skip_in = set(skip_in)
        self.num_layers = len(dims)
        self.layers = nn.ModuleList()
        for l in range(self.num_layers - 1):
            out_dim = dims[l + 1] - d_in if (l + 1) in self.skip_in else dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if l == self.num_layers - 2:
                nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=1e-4)
                nn.init.constant_(lin.bias, -radius)
            else:
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                nn.init.constant_(lin.bias, 0.0)
            self.layers.append(lin)
        self.act = nn.Softplus(beta=100)

    def pe(self, x):
        if self.pe_L == 0:
            return x
        out = [x]
        for l in range(self.pe_L):
            for fn in (torch.sin, torch.cos):
                out.append(fn(2.0 ** l * np.pi * x))
        return torch.cat(out, -1)

    def forward(self, x):
        inp = self.pe(x)
        h = inp
        for l, lin in enumerate(self.layers):
            if l in self.skip_in:
                h = torch.cat([h, inp], -1) / np.sqrt(2)
            h = lin(h)
            if l < self.num_layers - 2:
                h = self.act(h)
        return h


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def load_carve_points(carve_dir, center, scale, n_max=2000000, margin=0.02,
                      px_per_view=20000, samples_per_ray=4):
    """Free-space samples from a whole-scene depth dump (dump_scene_depth.py).
    Sample each pixel ray over [camera, depth - margin]; keep what lands inside the
    object bbox (normalised |x| < 1.2)."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(os.path.expanduser(carve_dir), "*.npz")))
    assert files, f"no carve depth under: {carve_dir}"
    pts = []
    for f in files:
        z = np.load(f)
        depth = z["depth"].astype(np.float32)
        fx, fy, cx, cy = float(z["fx"]), float(z["fy"]), float(z["cx"]), float(z["cy"])
        c2w = z["c2w"].astype(np.float32)
        vs, us = np.nonzero(depth > 0)
        if len(vs) == 0:
            continue
        sel = np.random.choice(len(vs), min(px_per_view, len(vs)), replace=False)
        v, u = vs[sel], us[sel]
        d = depth[v, u]
        dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, np.float32)], -1) @ c2w[:3, :3].T
        dnorm = np.linalg.norm(dirs, axis=-1)
        dn = dirs / dnorm[:, None]
        on = (c2w[:3, 3] - center) / scale                       # camera centre, normalised
        # sample only the ray/sphere(r=1.2) chord, so every sample is inside the bbox
        # and in front of the surface
        b = (on[None] * dn).sum(-1)
        disc = b * b - ((on * on).sum() - 1.44)
        hit = disc > 0
        if not hit.any():
            continue
        sq = np.sqrt(disc[hit])
        t_in = np.maximum(-b[hit] - sq, 0.02)
        t_out = -b[hit] + sq
        tmax_n = (d[hit] * dnorm[hit] - margin) / scale          # up to depth (unit dir, normalised) with margin
        hi = np.minimum(t_out, tmax_n)
        ok = hi > t_in
        if not ok.any():
            continue
        t_in, hi, dh = t_in[ok], hi[ok], dn[hit][ok]
        for _ in range(samples_per_ray):
            t = t_in + np.random.rand(len(t_in)).astype(np.float32) * (hi - t_in)
            pts.append(on[None] + dh * t[:, None])
    if not pts:
        return np.zeros((0, 3))
    P = np.concatenate(pts)
    if len(P) > n_max:
        P = P[np.random.choice(len(P), n_max, replace=False)]
    return P


# ---------------------------------------------------------------------------
_C0 = 0.28209479177387814


def _set_label_color(gaussians, label):
    """Swap gaussian colour for a label value (label-buffer render), as render_hole_novel."""
    fdc, frest = gaussians._features_dc, gaussians._features_rest
    saved = (fdc.detach().clone(), frest.detach().clone(), int(gaussians.active_sh_degree))
    dc = (label - 0.5) / _C0
    with torch.no_grad():
        fdc[:, 0, 0] = dc; fdc[:, 0, 1] = dc; fdc[:, 0, 2] = dc
        frest.zero_()
    gaussians.active_sh_degree = 0
    return saved


def _restore_color(gaussians, saved):
    fdc_s, frest_s, sh = saved
    with torch.no_grad():
        gaussians._features_dc.copy_(fdc_s)
        gaussians._features_rest.copy_(frest_s)
    gaussians.active_sh_degree = sh


@torch.no_grad()
def render_extra_masks(extra_cams, gaussians, pipe, background, label, thr=0.3):
    """Render object labels at extra poses to get per-view masks (blocks floor bleed)."""
    from gaussian_renderer import render as _render
    saved = _set_label_color(gaussians, label)
    masks = {}
    for cam in extra_cams:
        lab = _render(cam, gaussians, pipe, background)["render"][0].clamp(0, 1)
        masks[cam.image_name] = lab > thr
    _restore_color(gaussians, saved)
    return masks


_mask_info_printed = False


def load_view_mask(mask_dir, image_name, H, W):
    """Load a per-view object mask (None if absent). The value convention is detected:
    - 0/1 binary          -> object is > 0
    - amodal (188/0/255)  -> 188 (visible) only; 255 = bg, 0 = occluded
    - 0/255 binary        -> object is > 127
    """
    global _mask_info_printed
    from PIL import Image
    stem = os.path.splitext(image_name)[0]
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            img = Image.open(p).resize((W, H), Image.NEAREST)
            a = np.array(img)
            if a.ndim == 3 and a.shape[2] == 4:
                a = a[..., 3]          # RGBA: the mask is alpha (RGB is the instance colour code)
            elif a.ndim == 3:
                a = np.array(img.convert("L"))
            if a.max() <= 1:
                mm = a > 0
            elif (a == 188).any():
                mm = a == 188          # amodal convention: 188 = visible
            else:
                mm = a > 127
            if not _mask_info_printed:
                u, c = np.unique(a, return_counts=True)
                print(f"[mask] values in {os.path.basename(p)} after channel handling: "
                      f"{dict(zip(u.tolist()[:6], c.tolist()[:6]))} -> object px {int(mm.sum())}")
                _mask_info_printed = True
            return torch.from_numpy(mm).cuda()
    return None


def load_gt_depth(depth_dir, image_name, H, W, scale):
    """Load dataset GT depth, used to carve free space against the WHOLE scene.
    Unlike the rendered depth (the gaussian model has no table legs) it reflects real
    occlusion, so legs survive the carve.
    stem matching: frame000918 -> depth000918.png / frame000918.png / frame000918.npy"""
    from PIL import Image
    stem = os.path.splitext(image_name)[0]
    for c in (stem.replace("frame", "depth"), stem, stem + "_depth"):
        for ext in (".png", ".npy"):
            p = os.path.join(os.path.expanduser(depth_dir), c + ext)
            if not os.path.exists(p):
                continue
            if ext == ".npy":
                a = np.load(p).astype(np.float32)
            else:
                a = np.array(Image.open(p)).astype(np.float32) / scale
            if a.shape != (H, W):
                a = np.array(Image.fromarray(a).resize((W, H), Image.NEAREST))
            return a
    return None


@torch.no_grad()
def collect_oriented_points(scene, gaussians, pipe, background, args, mask_dir=None,
                            require_mask=False, extra_cams=None, extra_masks=None):
    """Back-project per-view depth to world points; normals are flipped toward the camera.
    mask_dir:     drop pixels outside the object mask (same policy as the TSDF path).
    require_mask: skip training views that have no mask (composed 200-view model with
                  masks for only 8 views).
    extra_cams:   extra novel cameras (MiniCam), unmasked -- constrain with an ROI crop.
                  See3D-refined unseen geometry is visible only at orbit poses."""
    # [speed] with thousands of training views back-projection dominates. The point cloud
    # is subsampled by --pts_per_view/--n_pts anyway, so striding views costs little.
    _stride = max(1, int(getattr(args, "view_stride", 1)))
    views = [(c, True) for c in scene.getTrainCameras()[::_stride]]
    if _stride > 1:
        print(f"[speed] view_stride={_stride} -> using {len(views)} training views")
    if extra_cams:
        views += [(c, False) for c in extra_cams]
    P_all, N_all, C_all, O_all = [], [], [], []
    EO_all, ED_all = [], []
    VB = []  # view buffers (depth+mask). NOTE: grid_fuse builds the OBSERVED TSDF from these too
    n_carve_views = getattr(args, "prior_carve_views", 0)
    # Keep every masked view, then subsample uniformly at the end.
    #   A fixed stride computed up front is wrong whenever only some training views carry a
    #   mask: the stride divides by ALL training views while only masked ones advance the
    #   counter, so the buffer count drops by the mask coverage ratio. Measured on a scene
    #   model with per-object masks: obj6 554/2000 masked -> 42 buffers instead of 184, and
    #   obj31 110/2000 -> 8, which then tripped the observation gate at 0.3% observed
    #   voxels. The list is halved whenever it exceeds 4x the target, so memory stays
    #   bounded regardless of how many views are masked.
    keep_all = n_carve_views > 0
    vb_cap = 4 * max(n_carve_views, 1)
    ti = 0
    n_masked_views = 0
    n_skipped = 0
    for cam, use_mask in views:
        m_obj = None
        pkg = render(cam, gaussians, pipe, background)
        depth = pkg["surf_depth"][0]                      # [H,W]
        alpha = pkg["rend_alpha"][0]                      # [H,W]
        rgb = pkg["render"].permute(1, 2, 0)              # [H,W,3]
        nrm = torch.nn.functional.normalize(pkg["rend_normal"], dim=0).permute(1, 2, 0)  # [H,W,3] world

        fx, fy, cx, cy, W, H, extrinsic = cam_intrinsics(cam)
        c2w = torch.inverse(extrinsic)                    # camera->world
        cam_center = c2w[:3, 3]

        vv, uu = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32),
                                torch.arange(W, device="cuda", dtype=torch.float32),
                                indexing="ij")
        x = (uu - cx) * depth / fx
        y = (vv - cy) * depth / fy
        pts_cam = torch.stack([x, y, depth], -1)          # [H,W,3]
        pts_w = pts_cam @ c2w[:3, :3].T + cam_center      # [H,W,3] world

        valid = (depth > 0) & (depth < args.depth_trunc) & (alpha > args.alpha_thr)
        if mask_dir is not None and use_mask:
            m = load_view_mask(mask_dir, cam.image_name, H, W)
            if m is not None:
                valid &= m
                m_obj = m
                n_masked_views += 1
            elif require_mask:
                n_skipped += 1
                continue          # skip training views without a mask (keeps whole-scene points out)

        # store depth+mask buffers; subsampled to prior_carve_views after the loop
        if use_mask and keep_all and m_obj is not None:
            ti += 1
            ds = max(1, int(getattr(args, "prior_carve_ds", 1)))
            dbuf = torch.where(alpha > args.alpha_thr, depth,
                               torch.zeros_like(depth))[::ds, ::ds].cpu().numpy()
            mbuf = m_obj[::ds, ::ds].cpu().numpy()
            w2c = extrinsic.cpu().numpy()
            b = {"R": w2c[:3, :3], "t": w2c[:3, 3],
                 "fx": fx / ds, "fy": fy / ds, "cx": cx / ds, "cy": cy / ds,
                 "W": dbuf.shape[1], "H": dbuf.shape[0],
                 "depth": dbuf, "mask": mbuf}
            if getattr(args, "gt_depth_dir", ""):
                dg = load_gt_depth(args.gt_depth_dir, cam.image_name, H, W,
                                   args.gt_depth_scale)
                if dg is not None:
                    b["dgt"] = dg[::ds, ::ds]
            VB.append(b)
            if len(VB) > vb_cap:                # halve, coverage stays uniform
                VB = VB[::2]
        if not use_mask and extra_masks is not None and cam.image_name in extra_masks:
            valid &= extra_masks[cam.image_name]   # extra poses: label-buffer object mask
        pts_w = pts_w[valid]
        n = nrm[valid]
        c = rgb[valid].clamp(0, 1)

        # flip normals toward the camera so the SDF sign is globally defined
        view_dir = cam_center[None] - pts_w
        flip = (n * view_dir).sum(-1) < 0
        n[flip] = -n[flip]
        n = torch.nn.functional.normalize(n, dim=-1)

        if len(pts_w) > args.pts_per_view:
            sel = torch.randperm(len(pts_w), device="cuda")[:args.pts_per_view]
            pts_w, n, c = pts_w[sel], n[sel], c[sel]

        P_all.append(pts_w.cpu()); N_all.append(n.cpu()); C_all.append(c.cpu())
        O_all.append(cam_center[None].expand(len(pts_w), 3).cpu())  # per-point observing camera centre

        # empty rays: alpha ~ 0 means "nothing observed along this ray".
        # Collect only from real training views. At extra/novel poses alpha ~ 0 means
        # "the model has no geometry here", not "empty space was observed" -- that
        # distinction is what stopped the carve from deleting generated legs.
        if args.empty_per_view > 0 and use_mask:
            em = alpha < args.empty_alpha
            eidx = em.nonzero(as_tuple=False)
            if len(eidx) > 0:
                sel_e = eidx[torch.randperm(len(eidx), device="cuda")[:args.empty_per_view]]
                ve, ue = sel_e[:, 0].float(), sel_e[:, 1].float()
                de = torch.stack([(ue - cx) / fx, (ve - cy) / fy, torch.ones_like(ue)], -1)
                de = torch.nn.functional.normalize(de @ c2w[:3, :3].T, dim=-1)
                EO_all.append(cam_center[None].expand(len(de), 3).cpu())
                ED_all.append(de.cpu())

    if mask_dir is not None:
        print(f"[mask] applied to {n_masked_views}/{len(views)} views, skipped {n_skipped} ({mask_dir})")
    if extra_cams:
        print(f"[extra] rendered {len(extra_cams)} poses (unmasked)")
    P = torch.cat(P_all).numpy().astype(np.float64)
    N = torch.cat(N_all).numpy().astype(np.float64)
    C = torch.cat(C_all).numpy().astype(np.float64)
    O = torch.cat(O_all).numpy().astype(np.float64)
    EO = torch.cat(EO_all).numpy().astype(np.float64) if EO_all else np.zeros((0, 3))
    ED = torch.cat(ED_all).numpy().astype(np.float64) if ED_all else np.zeros((0, 3))
    if keep_all and len(VB) > n_carve_views:
        sel = np.linspace(0, len(VB) - 1, n_carve_views).round().astype(int)
        VB = [VB[i] for i in np.unique(sel)]
    print(f"[rays] {len(EO)} empty rays, {len(VB)} view buffers "
          f"(from {ti} masked views, target {n_carve_views})")
    return P, N, C, O, EO, ED, VB


def train_sdf(P, N, O, EO, ED, args, CV=None, W=None, OBS=None, PV=None, PS=None):
    """Fit an IGR SDF to a normalised oriented point cloud.
    O      per-point observing camera centre (normalised): free-space carving near the surface.
    EO/ED  camera centre/direction of empty rays (alpha ~ 0 pixels): empty-ray carving.
    CV     free-space sample pool from whole-scene depth (carve_depth_dir); replaces
           empty-ray carving when present.
    OBS    bool mask marking real observed points. l_free applies only to them: prior
           points carry a fake origin (O = center) and used to carve the object interior.
    PV/PS  prior-mesh volume sample coords / target SDF (normalised), truncated-L1.
           Supervising both sides of a thin leg as positive blocks inflation by construction."""
    dev = "cuda"
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    Nt = torch.tensor(N, dtype=torch.float32, device=dev)
    Ot = torch.tensor(O, dtype=torch.float32, device=dev)
    Wt = torch.tensor(W, dtype=torch.float32, device=dev) if W is not None else None
    EOt = torch.tensor(EO, dtype=torch.float32, device=dev) if EO is not None and len(EO) else None
    EDt = torch.tensor(ED, dtype=torch.float32, device=dev) if ED is not None and len(ED) else None
    CVt = torch.tensor(CV, dtype=torch.float32, device=dev) if CV is not None and len(CV) else None
    obs_idx = None
    if OBS is not None:
        obs_idx = torch.tensor(np.nonzero(OBS)[0], dtype=torch.long, device=dev)
        if len(obs_idx) == len(Pt):
            obs_idx = None                       # all points observed: no gating needed
    PVt = torch.tensor(PV, dtype=torch.float32, device=dev) if PV is not None and len(PV) else None
    PSt = torch.tensor(PS, dtype=torch.float32, device=dev) if PVt is not None else None
    net = SDFNet(pe_L=args.pe_L).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    delta = args.offsurf_delta
    for it in range(args.sdf_iters):
        bi = torch.randint(0, len(Pt), (args.batch,), device=dev)
        pts = Pt[bi].clone().requires_grad_(True)
        nrm = Nt[bi]
        sdf = net(pts)
        g = grad(sdf, pts)
        if Wt is not None:
            w = Wt[bi]; wsum = w.sum().clamp(min=1e-8)
            l_man = (sdf.abs().squeeze(-1) * w).sum() / wsum
            l_nrm = ((1 - torch.nn.functional.cosine_similarity(g, nrm, dim=-1)) * w).sum() / wsum
            pp = Pt[bi] + delta * nrm; pm = Pt[bi] - delta * nrm
            l_sign = (((net(pp) - delta).abs().squeeze(-1) * w).sum()
                      + ((net(pm) + delta).abs().squeeze(-1) * w).sum()) / wsum
        else:
            l_man = sdf.abs().mean()
            l_nrm = (1 - torch.nn.functional.cosine_similarity(g, nrm, dim=-1)).mean()
            pp = Pt[bi] + delta * nrm; pm = Pt[bi] - delta * nrm
            l_sign = (net(pp) - delta).abs().mean() + (net(pm) + delta).abs().mean()

        # free-space carving near the surface: stepping back from an observed point p
        # toward its camera by s in [2d, free_range] lands in empty space -> SDF >= 0.
        # Observed points only: prior points have a fake origin (O = center), so they
        # carved the object interior and fought l_sign, warping the surface.
        l_free = torch.tensor(0.0, device=dev)
        if args.w_free > 0:
            bf = (obs_idx[torch.randint(0, len(obs_idx), (args.batch,), device=dev)]
                  if obs_idx is not None else bi)
            dirv = Pt[bf] - Ot[bf]
            dist = dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dirn = dirv / dist
            s = torch.rand(len(bf), 1, device=dev) * (args.free_range - 2 * delta) + 2 * delta
            s = torch.minimum(s, dist * 0.95)
            xf = Pt[bf] - dirn * s
            l_free = torch.relu(-net(xf)).mean()

        # empty-ray carving: a rendered alpha ~ 0 pixel means nothing was observed along
        # that ray. Sample only the chord inside the r=1.2 sphere (bbox) and force SDF >= 0.
        l_empty = torch.tensor(0.0, device=dev)
        if args.w_empty > 0 and CVt is not None:
            bj = torch.randint(0, len(CVt), (args.batch,), device=dev)
            l_empty = torch.relu(-net(CVt[bj])).mean()
        elif args.w_empty > 0 and EOt is not None and len(EOt) > 0:
            bj = torch.randint(0, len(EOt), (args.batch,), device=dev)
            o, dn = EOt[bj], EDt[bj]
            t0 = -(o * dn).sum(-1, keepdim=True)                 # ray parameter closest to the origin
            cp = o + dn * t0
            half = (1.44 - (cp * cp).sum(-1, keepdim=True)).clamp(min=0.0).sqrt()
            t = (t0 + (torch.rand_like(t0) * 2 - 1) * half).clamp(min=0.05)
            xe = o + dn * t
            l_empty = torch.relu(-net(xe)).mean()                         # penalise negative (inside) only

        # prior-mesh volume SDF distillation, truncated L1. Unlike injecting surface
        # points only, the empty space on BOTH sides of a leg is supervised as positive.
        l_prior = torch.tensor(0.0, device=dev)
        if PVt is not None and args.w_prior_sdf > 0:
            bp = torch.randint(0, len(PVt), (args.batch,), device=dev)
            l_prior = (net(PVt[bp]).squeeze(-1) - PSt[bp]).abs().mean()

        # eikonal: near-surface plus uniform random
        rp = torch.cat([Pt[torch.randint(0, len(Pt), (args.batch,), device=dev)]
                        + 0.02 * torch.randn(args.batch, 3, device=dev),
                        torch.rand(args.batch, 3, device=dev) * 2 - 1], 0).requires_grad_(True)
        ge = grad(net(rp), rp)
        l_eik = ((ge.norm(dim=-1) - 1) ** 2).mean()

        loss = (l_man + args.w_normal * l_nrm + args.w_sign * l_sign
                + args.w_eik * l_eik + args.w_free * l_free + args.w_empty * l_empty
                + args.w_prior_sdf * l_prior)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"[{it}] man {l_man.item():.4f} nrm {l_nrm.item():.4f} "
                  f"sign {l_sign.item():.4f} eik {l_eik.item():.4f} "
                  f"free {l_free.item():.4f} empty {l_empty.item():.4f} "
                  f"prior {l_prior.item():.4f}")
    return net


def grid_fuse_tsdf(VB, sd_fn, center, scale, args, debug_pts=None):
    """Deterministic SDF fusion, no MLP. Conflicts are resolved by PRIORITY, not averaging:
        observed TSDF > observed free space (occlusion-aware carve, +trunc) > generated SDF.
    This removes MLP interpolation pathologies (inflation, sponge) by construction; quality
    then depends only on alignment and generation. Occlusion-aware: space BEHIND an observed
    surface (leg regions) is left undecided so the prior can fill it."""
    from skimage.measure import marching_cubes
    from scipy import ndimage
    trunc = args.prior_trunc
    margin = args.prior_carve_margin
    G = args.grid if args.grid > 0 else int(round(2 * scale / args.voxel_size))
    G = int(min(G, args.max_grid))
    print(f"[grid-fuse] G={G} voxel~{2*scale/(G-1):.4f}m trunc={trunc}m views={len(VB)}")
    # [obs confidence] Observations near the silhouette are unreliable: grazing angles
    # give large depth error, and mask-border pixels flicker between fg and bg. Integrating
    # every pixel with weight 1 smeared the surface at the seen/unseen boundary.
    #   w_pix = (eroded mask interior) x (|cos(view, normal)| >= cos_min) x (|cos| or 1)
    # Normals come from the depth map; at a depth discontinuity the normal is orthogonal to
    # the view, so cos ~ 0 and cos_min excludes those pixels on its own.
    # Off by default -- an unseen_open default once changed results silently, so this must
    # be requested explicitly.
    _obs_on = args.obs_erode > 0 or args.obs_cos_min > 0 or args.obs_cos_weight

    def _cos_map(b):
        """|cos(view, surface normal)| from the depth map; not cached, to save memory."""
        d = b["depth"].astype(np.float32)
        H_, W_ = d.shape
        uu_, vv_ = np.meshgrid(np.arange(W_, dtype=np.float32),
                               np.arange(H_, dtype=np.float32))
        P = np.stack([(uu_ - b["cx"]) * d / b["fx"],
                      (vv_ - b["cy"]) * d / b["fy"], d], -1)
        nrm = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
        ln = np.linalg.norm(nrm, axis=-1); vd = np.linalg.norm(P, axis=-1)
        return np.abs((nrm * P).sum(-1)) / np.maximum(ln * vd, 1e-9)

    if _obs_on:
        VBo = [b for b in VB if "depth" in b and "obsw" not in b]
        # [reject guard] Erosion is cheap on screen-large objects and fatal on thin ones
        # (measured obj6: 3.4% of screen, erode=2 -> 16% rejected; at 0.5% it exceeds 40%).
        # Measure the reject rate on a sample view first and back erode off if it is high.
        probe = VBo[:: max(1, len(VBo) // 20)][:20]
        er = int(args.obs_erode)
        while er >= 0:
            kp = tot = 0.0
            for b in probe:
                m0 = b["mask"] & (b["depth"] > 0)
                m = ndimage.binary_erosion(m0, iterations=er) if er > 0 else m0
                tot += float(m0.sum())
                kp += float((m & (_cos_map(b) >= args.obs_cos_min)).sum())
            rej = 1.0 - kp / max(tot, 1.0)
            if rej <= args.obs_max_reject or er == 0:
                break
            print(f"[obs-conf] reject {rej*100:.0f}% > {args.obs_max_reject*100:.0f}% "
                  f"-> erode {er}->{er-1} (protects thin structure)")
            er -= 1
        if rej > args.obs_max_reject:
            print(f"[obs-conf] WARN reject {rej*100:.0f}% even at erode=0 -- cos_min is too "
                  f"strict here (mostly grazing views). Inspect the result visually.")
        for b in VBo:
            m = b["mask"] & (b["depth"] > 0)
            if er > 0:
                m = ndimage.binary_erosion(m, iterations=er)
            c = _cos_map(b)
            ok = m & (c >= args.obs_cos_min)
            b["obsw"] = np.where(ok, c if args.obs_cos_weight else 1.0, 0.0).astype(np.float32)
        args.obs_erode_used = er
        print(f"[obs-conf] erode={er}px (asked {args.obs_erode}) cos_min={args.obs_cos_min} "
              f"cos_weight={args.obs_cos_weight} -> rejected {rej*100:.0f}%")

    n_gt = sum(1 for b in VB if "dgt" in b)
    print(f"[grid-fuse] GT depth buffers {n_gt}/{len(VB)} views"
          + ("" if n_gt else "  WARN no GT depth -- falling back to silhouette carve, legs at risk"))
    # [gt-check] GT depth vs rendered depth. A large value (cm scale) means a scale or
    # frame-matching error, which invalidates every free-space decision. Expect ~0.
    for b in VB[:5]:
        dg = b.get("dgt")
        if dg is not None:
            mm = b["mask"] & (b["depth"] > 0) & (dg > 0.01)
            if mm.sum() > 100:
                d = (dg - b["depth"])[mm]
                print(f"[gt-check] median(dgt-render)={np.median(d):+.4f}m  "
                      f"median |d|={np.median(np.abs(d)):.4f}m  (mask {int(mm.sum())}px)")
    # [edge-bleed guard] Drop GT-depth discontinuity (silhouette) pixels from the free vote.
    # Off (=0) by default. It was added to treat "thin legs get carved", but the real cause
    # turned out to be --unseen_open 0.015 (78935 voxels deleted). Re-measured after fixing
    # that, it protects nothing and only narrows the carve, raising free violations:
    #   obj22  free 10.8% -> 7.5% when off (-31%), unseen F@2 0.3670 -> 0.3637 (noise)
    #   obj6   free  4.9% -> 4.7%,                 unseen F@2 0.6586 -> 0.6592 (legs intact)
    # Try 0.1 only on data where edges genuinely bleed, e.g. nearest-resized depth.
    if args.gt_edge_thr > 0:
        for b in VB:
            dg = b.get("dgt")
            if dg is not None and "dgt_ok" not in b:
                gy_, gx_ = np.gradient(dg.astype(np.float32))
                edge = (np.abs(gx_) + np.abs(gy_)) > args.gt_edge_thr
                edge = ndimage.binary_dilation(edge, iterations=1)
                b["dgt_ok"] = (dg > 0.01) & ~edge
    else:
        for b in VB:
            dg = b.get("dgt")
            if dg is not None and "dgt_ok" not in b:
                b["dgt_ok"] = dg > 0.01
    # [GPU] Upload the view buffers once. 200 views x 2.1M voxels x 64 slabs is 27e9
    # projections: 28 min in numpy, tens of seconds on GPU. The formulas are unchanged, so
    # results must match (--fuse_device cpu to compare). The full grid (5GB at 512^3) stays
    # on CPU and only slabs are exchanged, so the GPU holds ~1.5GB.
    _dev = args.fuse_device
    if _dev == "auto":
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _gpu = _dev.startswith("cuda")
    # [precision] The CPU path projects Xw in float64. float32 on GPU differs by 1e-7, but
    # the gate (alpha<0.25 -> Wo<2) and keep_connected are threshold statistics, which
    # amplify it (measured obj6: arrays differ 0.1%p, the gate reads 34.6% vs 38.8%).
    # FP64 is only 2x slower than FP32 on A100, so float64 costs little and keeps accuracy.
    _dt = torch.float64 if args.fuse_dtype == "float64" else torch.float32
    if _gpu:
        _t0g = time.time()
        for b in VB:
            b["_R"] = torch.as_tensor(b["R"], dtype=_dt, device=_dev)
            b["_t"] = torch.as_tensor(b["t"], dtype=_dt, device=_dev)
            b["_depth"] = torch.as_tensor(b["depth"], dtype=_dt, device=_dev)
            b["_mask"] = torch.as_tensor(np.ascontiguousarray(b["mask"]),
                                         dtype=torch.bool, device=_dev)
            if "obsw" in b:
                b["_obsw"] = torch.as_tensor(b["obsw"], dtype=_dt, device=_dev)
            if b.get("dgt") is not None:
                b["_dgt"] = torch.as_tensor(b["dgt"], dtype=_dt, device=_dev)
                b["_dgt_ok"] = torch.as_tensor(np.ascontiguousarray(b["dgt_ok"]),
                                               dtype=torch.bool, device=_dev)
        _mb = torch.cuda.memory_allocated(_dev) / 1024**2
        print(f"[fuse-gpu] {len(VB)} view buffers -> {_dev} {args.fuse_dtype} "
              f"({_mb:.0f}MB, {time.time()-_t0g:.1f}s)")

    lin = np.linspace(-1, 1, G, dtype=np.float32)
    Fo = np.zeros((G, G, G), np.float32)      # observed TSDF, weighted sum
    Wo = np.zeros((G, G, G), np.float32)      # observation weight (view count)
    FRc = np.zeros((G, G, G), np.uint16)      # votes for observed empty space (consensus)
    VOb = np.zeros((G, G, G), np.uint16)      # votes: this object, near a scene surface
    VOt = np.zeros((G, G, G), np.uint16)      # votes: another object, near a scene surface
    NFR = np.zeros((G, G, G), np.uint16)      # [visual hull] views with this voxel in frustum
    NIN = np.zeros((G, G, G), np.uint16)      # [visual hull] views projecting inside the mask
    SG = np.empty((G, G, G), np.float32)      # generated SDF (truncated)
    for k0 in range(0, G, 8):
        k1 = min(k0 + 8, G)
        gx, gy, gz = np.meshgrid(lin, lin, lin[k0:k1], indexing="ij")
        Xw = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float64) * scale + center
        SG[:, :, k0:k1] = np.clip(sd_fn(Xw), -trunc, trunc).reshape(G, G, k1 - k0)
        if _gpu:
            # [GPU] Same formulas in the same order as the CPU path; accumulation stays
            # float32 so the association order matches too. Results agree to within 1e-6
            # (verify against --fuse_device cpu).
            Xt = torch.as_tensor(Xw, dtype=_dt, device=_dev)
            N = Xt.shape[0]
            f_ = torch.zeros(N, dtype=_dt, device=_dev)
            w_ = torch.zeros(N, dtype=_dt, device=_dev)
            fr_ = torch.zeros(N, dtype=torch.int32, device=_dev)
            vo_ = torch.zeros(N, dtype=torch.int32, device=_dev)
            vt_ = torch.zeros(N, dtype=torch.int32, device=_dev)
            nfr_ = torch.zeros(N, dtype=torch.int32, device=_dev)
            nin_ = torch.zeros(N, dtype=torch.int32, device=_dev)
            for b in VB:
                Xc = Xt @ b["_R"].T + b["_t"]
                z = Xc[:, 2]; zz = z.clamp_min(1e-6)
                u = b["fx"] * Xc[:, 0] / zz + b["cx"]
                v = b["fy"] * Xc[:, 1] / zz + b["cy"]
                infr = (z > 0.05) & (u >= 0) & (u < b["W"]) & (v >= 0) & (v < b["H"])
                ui = u.clamp(0, b["W"] - 1).long()
                vi = v.clamp(0, b["H"] - 1).long()
                di = b["_depth"][vi, ui]; mi = b["_mask"][vi, ui]
                nfr_ += infr.int()
                nin_ += (infr & mi).int()
                sdf = di - z
                hit = infr & mi & (di > 0) & (sdf > -trunc)
                if "_obsw" in b:
                    wp = b["_obsw"][vi, ui]
                    hit = hit & (wp > 0)
                    hm = hit.to(_dt)
                    f_ += sdf.clamp(-trunc, trunc) * wp * hm
                    w_ += wp * hm
                else:
                    hm = hit.to(_dt)
                    f_ += sdf.clamp(-trunc, trunc) * hm
                    w_ += hm
                if "_dgt" in b:
                    dgt = b["_dgt"][vi, ui]
                    vgt = infr & b["_dgt_ok"][vi, ui]
                    fr_ += (vgt & (z < dgt - margin)).int()
                    near = vgt & ((z - dgt).abs() < 2 * margin)
                    vo_ += (near & mi).int()
                    vt_ += (near & ~mi).int()
                else:
                    fr_ += (infr & (~mi) & ((di <= 0) | (z < di - margin))).int()
            f = f_.cpu().numpy().astype(np.float32)
            w = w_.cpu().numpy().astype(np.float32)
            fr = fr_.cpu().numpy().astype(np.uint16)
            vo = vo_.cpu().numpy().astype(np.uint16); vt = vt_.cpu().numpy().astype(np.uint16)
            nfr = nfr_.cpu().numpy().astype(np.uint16); nin = nin_.cpu().numpy().astype(np.uint16)
            sh = (G, G, k1 - k0)
            Fo[:, :, k0:k1] += f.reshape(sh); Wo[:, :, k0:k1] += w.reshape(sh)
            FRc[:, :, k0:k1] += fr.reshape(sh)
            VOb[:, :, k0:k1] += vo.reshape(sh); VOt[:, :, k0:k1] += vt.reshape(sh)
            NFR[:, :, k0:k1] += nfr.reshape(sh); NIN[:, :, k0:k1] += nin.reshape(sh)
            if (k0 // 8) % 8 == 0:
                print(f"  [grid-fuse/gpu] slab {k0}/{G}", flush=True)
            continue

        f = np.zeros(len(Xw), np.float32); w = np.zeros(len(Xw), np.float32)
        fr = np.zeros(len(Xw), np.uint16)
        vo = np.zeros(len(Xw), np.uint16); vt = np.zeros(len(Xw), np.uint16)
        nfr = np.zeros(len(Xw), np.uint16); nin = np.zeros(len(Xw), np.uint16)
        for b in VB:
            Xc = Xw @ b["R"].T + b["t"]; z = Xc[:, 2]; zz = np.maximum(z, 1e-6)
            u = b["fx"] * Xc[:, 0] / zz + b["cx"]
            v = b["fy"] * Xc[:, 1] / zz + b["cy"]
            infr = (z > 0.05) & (u >= 0) & (u < b["W"]) & (v >= 0) & (v < b["H"])
            ui = np.clip(u, 0, b["W"] - 1).astype(int)
            vi = np.clip(v, 0, b["H"] - 1).astype(int)
            di = b["depth"][vi, ui]; mi = b["mask"][vi, ui]
            # [visual hull] Intersection of mask cones; no occlusion test needed. Voxels on
            # the floor or on a neighbouring object project outside the mask in most views
            # and are dropped, while the object's unobserved back stays inside the mask.
            nfr += infr.astype(np.uint16)
            nin += (infr & mi).astype(np.uint16)
            sdf = di - z                                   # positive = in front of the surface
            hit = infr & mi & (di > 0) & (sdf > -trunc)    # beyond trunc behind the surface: skip (occlusion)
            # [obs confidence] silhouette/grazing pixels get low or zero weight.
            if "obsw" in b:
                wp = b["obsw"][vi, ui]
                hit &= wp > 0
                f[hit] += np.clip(sdf[hit], -trunc, trunc) * wp[hit]; w[hit] += wp[hit]
            else:
                f[hit] += np.clip(sdf[hit], -trunc, trunc); w[hit] += 1
            dg = b.get("dgt")
            if dg is not None:
                # [GT-depth carve] Mask-independent, against real scene geometry:
                #   z < d_gt - margin  -> vote "observed empty" (consensus, edges excluded)
                #   near a scene surface -> vote this-object / other-object (robust to
                #   mask noise)
                dgt = dg[vi, ui]
                vgt = infr & b["dgt_ok"][vi, ui]
                fr += (vgt & (z < dgt - margin)).astype(np.uint16)
                near = vgt & (np.abs(z - dgt) < 2 * margin)
                vo += (near & mi).astype(np.uint16)
                vt += (near & ~mi).astype(np.uint16)
            else:
                # Legacy silhouette carve, only without GT depth. The rendered depth has
                # no legs, so its occlusion test is incomplete and legs can be cut.
                fr += (infr & (~mi) & ((di <= 0) | (z < di - margin))).astype(np.uint16)
        sh = (G, G, k1 - k0)
        Fo[:, :, k0:k1] += f.reshape(sh); Wo[:, :, k0:k1] += w.reshape(sh)
        FRc[:, :, k0:k1] += fr.reshape(sh)
        VOb[:, :, k0:k1] += vo.reshape(sh); VOt[:, :, k0:k1] += vt.reshape(sh)
        NFR[:, :, k0:k1] += nfr.reshape(sh); NIN[:, :, k0:k1] += nin.reshape(sh)
        if (k0 // 8) % 8 == 0:
            print(f"  [grid-fuse] slab {k0}/{G}")
    if _gpu:   # release the view buffers once the slab loop ends; the rest is CPU-only
        for b in VB:
            for k in ("_R", "_t", "_depth", "_mask", "_obsw", "_dgt", "_dgt_ok"):
                b.pop(k, None)
        torch.cuda.empty_cache()
        print(f"[fuse-gpu] buffers freed ({torch.cuda.memory_allocated(_dev)/1024**2:.0f}MB left)")

    Fobs = Fo / np.maximum(Wo, 1e-6)
    alpha = np.clip(Wo / args.grid_wcap, 0, 1)             # observation confidence, from view count
    # [seam] Blur alpha, not F: this creates a blend band at the observed/generated
    # boundary and removes the step, without introducing a discontinuity of its own.
    if getattr(args, "alpha_smooth", 0) > 0:
        alpha = ndimage.gaussian_filter(alpha, sigma=args.alpha_smooth)
    FREE = FRc >= args.free_min_views                      # consensus: >= N views voted empty
    OTH = VOt > VOb                                        # another object's surface (majority vote)
    step = 2.0 / (G - 1)

    # [sign-fix] A non-watertight generated mesh (multi-material glb) has a broken
    # winding-number sign, so its interior is not negative: the surface is kept by the
    # classifier yet never appears in marching cubes. Treat |SG| as unsigned, then
    # flood-fill from the grid border across the surface shell: reached = outside.
    def _fix_sign(SGv):
        vox = 2 * scale / (G - 1)
        UD = np.abs(SGv)
        # (1) flood-fill sign: recovers the interior of closed parts (leaks through holes)
        shell = UD <= 1.5 * vox
        openv = ~shell
        lab, _ = ndimage.label(openv)
        bl = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(),
                                       lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        outside = np.isin(lab, bl[bl > 0]) & openv
        inside = openv & ~outside
        flood = np.where(inside, -UD, UD)
        # (2) Adaptive offset shell UD - d(x): a zero-thickness sheet becomes a 2d volume,
        #     with d driven by distance to the observed surface -- d_min near observations
        #     (table rim) to avoid inflation and double surfaces, d_max deep in unobserved
        #     space (legs) so they come out solid.
        dmin = max(1.5 * vox, args.shell_delta_min)
        dmax = max(dmin, args.shell_delta)
        Dobs = ndimage.distance_transform_edt(~(alpha > 0.25)).astype(np.float32) * vox
        dmap = np.clip(dmin + (dmax - dmin) * (Dobs / max(args.shell_ramp, 1e-6)),
                       dmin, dmax)
        out = np.minimum(flood, UD - dmap).astype(np.float32)    # volume union
        print(f"[sign-fix] flood inside {inside.mean()*100:.2f}%  final SG<0 {(out < 0).mean()*100:.2f}%  "
              f"(before {(SGv < 0).mean()*100:.2f}%, d {dmin*1000:.0f}->{dmax*1000:.0f}mm "
              f"ramp {args.shell_ramp}m)")
        return out

    need_sign_fix = getattr(args, "grid_sign_fix", False) or not getattr(args, "prior_watertight", True)
    if need_sign_fix:
        SG = _fix_sign(SG)

    # [carve-align] 9-DoF re-optimisation so the generated geometry avoids observed free
    # space and settles into unknown space -- the only region where unseen geometry can
    # actually exist. Anchor: generated surface touching observation-dominated voxels is
    # held on the observed TSDF zero-set.
    if getattr(args, "carve_align", False) and debug_pts is not None:
        from scipy.optimize import minimize as _pmin
        from scipy.spatial.transform import Rotation as _Rot
        freef = np.minimum(FRc.astype(np.float32) / max(args.free_min_views, 1), 1.0)
        pn_all = ((debug_pts - center) / scale).astype(np.float64)
        sel = np.random.default_rng(1).choice(len(pn_all), min(20000, len(pn_all)), replace=False)
        pn = pn_all[sel]

        def _interp(vol, q):
            idx = np.clip(np.round((q + 1) / step), 0, G - 1).astype(int)
            return vol[idx[:, 0], idx[:, 1], idx[:, 2]]

        anch = _interp(alpha, pn) > 0.5
        c0 = pn.mean(0)

        def _unpack(x):
            return _Rot.from_rotvec(x[:3]).as_matrix(), x[3:6], np.exp(x[6:9])

        def _loss(x):
            R_, t_, s_ = _unpack(x)
            q = ((pn - c0) * s_) @ R_.T + c0 + t_
            L_free = _interp(freef, q).mean()                       # penalty for occupying free space
            if anch.any():
                qa = q[anch]
                ai = _interp(alpha, qa) > 0.25
                fo = np.abs(_interp(Fobs, qa))
                L_anch = float(np.where(ai, fo, trunc).mean()) / trunc   # penalty for leaving the observed surface
            else:
                L_anch = 0.0
            return L_free + args.carve_align_w * L_anch + 0.05 * float(np.abs(x).sum())

        x0 = np.zeros(9); l0 = _loss(x0)
        res = _pmin(_loss, x0, method="Powell", options={"maxiter": 250, "xtol": 1e-4})
        R_, t_, s_ = _unpack(res.x)
        print(f"[carve-align] loss {l0:.4f}->{res.fun:.4f}  dt={np.round(res.x[3:6]*scale, 3)}m  "
              f"scale={np.round(s_, 3)}  rot={np.rad2deg(np.linalg.norm(res.x[:3])):.1f}deg")
        sm = float(s_.mean())
        for k0 in range(0, G, 8):                          # recompute the generated SDF with the correction
            k1 = min(k0 + 8, G)
            gx, gy, gz = np.meshgrid(lin, lin, lin[k0:k1], indexing="ij")
            Xn = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float64)
            q = ((Xn - c0 - res.x[3:6]) @ R_) / s_ + c0
            SG[:, :, k0:k1] = np.clip(sd_fn(q * scale + center) * sm,
                                      -trunc, trunc).reshape(G, G, k1 - k0)
        if need_sign_fix:
            SG = _fix_sign(SG)                             # restore the sign of the recomputed SG
        debug_pts = (((pn_all - c0) * s_) @ R_.T + c0 + res.x[3:6]) * scale + center

    # [visual hull] Restrict the prior to the intersection of the object's own mask cones.
    # The carve only deletes space IN FRONT of an observed surface, so generated geometry
    # that burrows into the floor or a neighbour survives and seen accuracy explodes
    # (measured obj20: 2.1 -> 107mm). The hull blocks that without an occlusion test.
    HULL = np.ones((G, G, G), bool)
    if args.hull_min_frac > 0:
        HULL = (NIN >= args.hull_min_frac * np.maximum(NFR, 1)) & (NFR >= args.hull_min_views)
        print(f"[hull] inside the mask cones {HULL.mean()*100:.1f}%  "
              f"(frac>={args.hull_min_frac}, min {args.hull_min_views} views)  "
              f"-> removed {(np.abs(SG) < 2*2*scale/(G-1))[~HULL].mean()*100 if (~HULL).any() else 0:.1f}% of the generated surface")

    # [apply gate] An object with almost no unobserved region has nothing to gain from the
    # prior and everything to lose (measured: objects with baseline unseen completion ~15mm
    # collapsed from unseen F@2cm 0.68 to 0.21). Decide without GT, from the fraction of the
    # generated SURFACE that sits in unknown space.
    # Do NOT measure interior volume: an object's interior is unobserved in every case, so
    # even a fully observed object scores high (67% fully observed vs 91% half unobserved --
    # no discriminative power).
    vox_g = 2 * scale / (G - 1)
    prior_surf = np.abs(SG) < 1.5 * vox_g
    unknown = (alpha < 0.25) & ~FREE & ~OTH
    ufrac = float((prior_surf & unknown).sum()) / max(int(prior_surf.sum()), 1)
    print(f"[gate] generated surface in unknown space {ufrac*100:.1f}% "
          f"(threshold {args.min_unknown_frac*100:.1f}%)")
    prior_applied = True

    # [obs gate] The gate above asks whether the GENERATED surface sits in unknown space.
    # It does not ask whether there is enough observation to constrain the prior at all.
    # With almost nothing observed the prior invents the object: measured obj31 at 1.0%
    # observed voxels went 2.16 -> 69.88mm seen accuracy and obj28 at 2.1% went 3.84 ->
    # 77.55mm, while obj22 at 7.7% improved 0.971 -> 0.992 seen F@1. The distribution has
    # a clean gap at 4.6-7.7%. Both failures pass min_unknown_frac (46.4% / 67.3% vs a 12.5%
    # threshold), so this is a second, independent condition.
    obs_frac = float((Wo > 0).mean())
    if obs_frac < args.min_obs_frac:
        print(f"[gate] observed voxels {obs_frac*100:.1f}% < {args.min_obs_frac*100:.0f}% "
              f"-> prior skipped (observation cannot constrain it)")
        SG = np.full_like(SG, trunc)
        prior_applied = False
    elif ufrac < args.min_unknown_frac:
        print("  -> not enough unobserved space: prior skipped, observation only")
        SG = np.full_like(SG, trunc)
        prior_applied = False

    # Without a prior there must be no alpha blend. In F = alpha*Fobs + (1-alpha)*base
    # with base = trunc (empty), the zero crossing moves to Fobs = -(1-alpha)/alpha*trunc,
    # i.e. the surface is pushed inward; at alpha=0.5 the shift is a full trunc (50mm) and
    # thin objects vanish entirely. Measured: obj16 (picture frame) seen F@1 0.914 -> 0.647,
    # obj35 0.984 -> 0.868. Both had the prior blocked by a low ufrac, so the loss came from
    # this blend, not from the prior. With nothing to blend, use the observation as is.
    # [passthrough] If the prior is not applied, return the observed reconstruction as is.
    #   The claim of this method is that it completes UNOBSERVED regions. With nothing to
    #   complete, a no-op is the honest result and re-fusing only costs quality.
    #   Measured on the 4 gate-blocked objects:
    #     obj16 seen F@1 0.914->0.647   obj35 0.984->0.980
    #     obj10 free 5.36%->27.12%      obj8  free 7.55%->22.33%
    #   The prior contributed nothing in all four, so re-fusion only lowered the metrics.
    #   (Its one benefit, the carve, moved obj16 free from 2.3% to 1.9% -- negligible.)
    if not prior_applied and args.passthrough_mesh:
        _pm = os.path.expanduser(args.passthrough_mesh)
        if os.path.isfile(_pm):
            _m = o3d.io.read_triangle_mesh(_pm)
            if len(_m.vertices):
                print(f"[passthrough] prior skipped -> returning the observed mesh "
                      f"({os.path.basename(_pm)}, {len(_m.vertices)} verts)")
                return np.asarray(_m.vertices), np.asarray(_m.triangles)
        print(f"[passthrough] WARN missing file: {_pm} -- continuing with fusion")

    #   Exclude carve (FREE) and other-object (OTH) voxels. Setting alpha=1 there makes
    #   F = Fobs and the carve is ignored outright. Measured when that was done: obj8 sanity
    #   violation 27.9% -> 50.9%, obj10 free 21.9% -> 27.6%, and 84% of the violation was
    #   observation-dominated. The object's rendered depth says "there is a surface here"
    #   while the GT scene depth says "this is empty"; the latter is a multi-view consensus
    #   and is the more trustworthy of the two.
    if not prior_applied and not args.no_alpha_full_wo_prior:
        lift = (Wo > 0) & ~FREE & ~OTH
        n_lift = int((lift & (alpha < 1.0)).sum())
        alpha = np.where(lift, np.float32(1.0), alpha)
        print(f"  -> no prior: alpha=1 on observed voxels ({n_lift} voxels, "
              f"carve/other-object excluded). Pulling toward empty space with "
              f"nothing to blend erodes the surface.")

    # free / other-object / outside hull = +trunc;  unobserved and inside hull = generated
    base = np.where(FREE | OTH | ~HULL, trunc, SG)
    F = alpha * Fobs + (1 - alpha) * base                  # priority blend

    # [opening] Remove spikes in unobserved regions. Behind a sofa no camera sees through,
    # so there is no carve constraint and generated spikes survive. Erosion then dilation
    # deletes only protrusions thinner than the structuring element and keeps the body and
    # legs (observed regions are untouched). The radius is in metres: the shell offset (2d)
    # inflates a spike of true thickness T to T + 2d, and removal needs r > T/2 + d, so d is
    # folded in automatically.
    if getattr(args, "unseen_open", 0) > 0:
        vox = 2 * scale / (G - 1)
        k = max(1, int(round(args.unseen_open / vox)))
        neg = F < 0
        st = ndimage.generate_binary_structure(3, 1)
        op = ndimage.binary_dilation(
            ndimage.binary_erosion(neg, st, iterations=k), st, iterations=k)
        rm = neg & ~op & (alpha < 0.5)
        F = np.where(rm, trunc, F)
        print(f"[opening] removed {int(rm.sum())} spike voxels "
              f"(r={args.unseen_open*1000:.0f}mm -> k={k} voxels, "
              f"protrusions up to {2*args.unseen_open*1000:.0f}mm thick)")

    if args.grid_smooth > 0:
        F = ndimage.gaussian_filter(F, sigma=args.grid_smooth)

    # [free-hard] Re-apply the carve constraint after smoothing. base = where(FREE|OTH|
    #   ~HULL, trunc, SG) is a hard constraint, but the gaussian blur on the next line
    #   pushes the zero crossing into empty space wherever the prior body (-trunc) meets
    #   carved free space (+trunc). Measured (obj22): 71% of free violations came from the
    #   prior, 20% from observation and only 3.7% from a lenient carve -- i.e. smoothing for
    #   surface quality was overwriting a geometric constraint. Voxels where observation
    #   dominates (high alpha) are left alone; there observation wins.
    if args.free_hard:
        # FREE and OTH are not equally strong evidence.
        #   FREE = a camera saw THROUGH this point to something farther (GT depth, strong)
        #   OTH  = more views voted "another object" (mask vote, noisy)
        # Measured: forcing both flipped obj2's seen F@1 from +0.042 to -0.062, apparently
        # because the OTH vote wavers at a neighbour boundary and deletes real surface.
        # FREE only by default; use --free_hard_oth to include OTH.
        hard = (FREE | OTH if args.free_hard_oth else FREE) & (alpha < args.free_hard_alpha)
        nneg = int((hard & (F < 0)).sum())
        nobs = int((hard & (F < 0) & (alpha > 0.5)).sum())
        F = np.where(hard, np.maximum(F, trunc), F)
        print(f"[free-hard] carve re-applied after smoothing: cleared "
              f"{nneg} negative free/other voxels ({nobs} observation-dominated) "
              f"[alpha<{args.free_hard_alpha}]")
        if nneg == 0:
            print("  WARN nothing was cleared -- free_hard_alpha may be too low to fire; "
                  "violations sit mostly in observation-dominated voxels with alpha>0.5.")

    # [keep-connected] Keep only the negative components connected to this object's
    # observed voxels. A prior generated as one blob over several objects leaves debris in
    # a neighbour's occluded (unknown) space; this removes it structurally. Legs survive
    # because they connect to the observed part through the top or bottom plate.
    if getattr(args, "keep_connected", False):
        neg = F < 0
        lab, ncomp = ndimage.label(neg)
        seeds = np.unique(lab[neg & (alpha > 0.5)])
        seeds = seeds[seeds > 0]
        keepm = np.isin(lab, seeds)
        removed = int(neg.sum() - keepm.sum())
        F = np.where(neg & ~keepm, trunc, F)
        print(f"[keep-connected] {ncomp} negative components -> kept {len(seeds)} touching "
              f"observation, removed {removed} voxels "
              f"({removed/max(neg.sum(),1)*100:.1f}%)")

    # [probe] Voxel-class statistics inside a world-coordinate box, to answer "why is this
    # leg missing" locally. Usage: --probe_box "x0,y0,z0,x1,y1,z1" (read the coords off a
    # viewer, around the missing part).
    if getattr(args, "probe_box", ""):
        try:
            v = [float(x) for x in args.probe_box.split(",")]
            assert len(v) == 6
        except (ValueError, AssertionError):
            print(f"[probe] bad format: '{args.probe_box}' -- needs 6 numbers "
                  f'(e.g. --probe_box "1.2,-0.5,0.0,1.5,-0.2,0.6"). Skipping probe.')
            v = None
    else:
        v = None
    if v is not None:
        lo_n = (((np.array(v[:3]) - center) / scale) + 1) / step
        hi_n = (((np.array(v[3:]) - center) / scale) + 1) / step
        i0, j0, k0p = np.clip(np.floor(np.minimum(lo_n, hi_n)), 0, G - 1).astype(int)
        i1, j1, k1p = np.clip(np.ceil(np.maximum(lo_n, hi_n)), 0, G - 1).astype(int)
        sub = np.s_[i0:i1 + 1, j0:j1 + 1, k0p:k1p + 1]
        nvox = FREE[sub].size
        print(f"[probe] {v} ({nvox} voxels): FREE {FREE[sub].mean()*100:.0f}%  "
              f"FRc>0 {(FRc[sub] > 0).mean()*100:.0f}%  obs {(alpha[sub] > 0.5).mean()*100:.0f}%  "
              f"OTH {OTH[sub].mean()*100:.0f}%  SG<0 {(SG[sub] < 0).mean()*100:.0f}%  "
              f"final F<0 {(F[sub] < 0).mean()*100:.0f}%")
    print(f"[grid-fuse] observed {(Wo > 0).mean()*100:.1f}%  "
          f"free {(FREE & (Wo == 0)).mean()*100:.1f}% ({args.free_min_views}-view consensus, "
          f"any-view {((FRc > 0) & (Wo == 0)).mean()*100:.1f}%)  "
          f"other-obj {OTH.mean()*100:.2f}%  "
          f"generated-interior {((SG < 0) & (Wo == 0) & ~FREE & ~OTH).mean()*100:.2f}%")
    step = 2.0 / (G - 1)
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(step,) * 3)
    verts = (verts - 1.0) * scale + center

    # [open boundary] Without a prior, do not close the unobserved region.
    #   Observed voxels hold F = Fobs (can be negative) while unobserved ones are forced to
    #   F = +trunc, so a zero crossing appears at their boundary and builds a fake wall: a
    #   face at trunc behind the object that does not exist. On a thin object seen only from
    #   the front (a picture frame) that wall becomes most of the mesh.
    #   Measured obj16: the output was 30% seen (baseline 87%), seen F@1 0.914 -> 0.642.
    #   obj8: 84% of the sanity violation was observation-dominated -- this same surface.
    #   Standard TSDF leaves weight-0 voxels undefined and does not mesh them. Close the
    #   region only when there is evidence to fill it (= the prior applied). Carve and
    #   other-object boundaries are kept: there we DO know the space is empty.
    if not prior_applied and len(faces):
        # Dilation is required: a fake-wall vertex lies BETWEEN an observed and an
        # unobserved voxel, so rounding puts it on the observed side and it is missed.
        # Synthetic check: 0% detected without dilation, 50% with (one of the two sheets).
        unk = ndimage.binary_dilation((Wo == 0) & ~FREE & ~OTH)
        vg = np.clip(np.round(((verts - center) / scale + 1.0) / step), 0, G - 1).astype(int)
        bad_v = unk[vg[:, 0], vg[:, 1], vg[:, 2]]
        keep_f = ~bad_v[faces].any(axis=1)
        n_drop = int((~keep_f).sum())
        faces = faces[keep_f]
        used = np.unique(faces)
        remap = np.full(len(verts), -1, np.int64); remap[used] = np.arange(len(used))
        verts, faces = verts[used], remap[faces]
        print(f"[open-boundary] no prior -> dropped {n_drop} fake faces at the unobserved "
              f"boundary ({n_drop + len(faces)} -> {len(faces)}). The mesh is left open.")

    # [sanity] Does the output contradict the observation?
    # Measured in a batch: obj0 went from 3.85mm to 1165mm (1.2m!) seen accuracy -- a sound
    # reconstruction destroyed. That is not a quality drop but a shape placed in the wrong
    # location, which is detectable without GT. Failing is better than shipping garbage.
    #
    # Do NOT test "a vertex lies in a FREE voxel": marching-cubes vertices sit BETWEEN
    #   voxels, so vertices on the legitimate object/empty boundary round into FREE. A
    #   synthetic check scores 50% for a perfect sphere inside a carve. This criterion
    #   actually killed a healthy obj28 as a false positive (25.8% by rounding vs 5.7% by
    #   depth, and the gate had blocked the prior so its contribution was 0%).
    #   Count only vertices that penetrate the free space by more than a margin.
    if not args.no_sanity:
        vi_ = np.clip(((verts - center) / scale + 1.0) / step, 0, G - 1).astype(int)
        sx, sy, sz = vi_[:, 0], vi_[:, 1], vi_[:, 2]
        k = max(1, int(np.ceil(args.sanity_free_depth / vox_g)))
        FREE_deep = ndimage.binary_erosion(FREE, iterations=k)   # k voxels inside the boundary
        vdeep = FREE_deep[sx, sy, sz]
        s_free = float(vdeep.mean())
        obsv = Wo[sx, sy, sz] > 0
        s_disp = float(np.median(np.abs(Fobs[sx, sy, sz][obsv]))) if obsv.any() else 0.0
        nd = max(int(vdeep.sum()), 1)
        va = alpha[sx, sy, sz] > 0.5          # observation-dominated voxels
        vgen = (SG[sx, sy, sz] < 0) & ~va     # surface created by the prior
        vfr = FRc[sx, sy, sz]
        print(f"[sanity] of {len(verts)} output vertices, {s_free*100:.1f}% penetrate free "
              f"space by more than {args.sanity_free_depth*1000:.0f}mm (={k} voxels) "
              f"(threshold {args.sanity_free_max*100:.0f}%) / median displacement from the "
              f"observed surface {s_disp*1000:.1f}mm "
              f"(threshold {args.sanity_disp_max*1000:.0f}mm, "
              f"{int(obsv.sum())} vertices on observed surface)")
        print(f"[free-split] violation is {float((vdeep & va).sum())/nd*100:.0f}% "
              f"observation-dominated / "
              f"prior {float((vdeep & vgen).sum())/nd*100:.0f}%  |  "
              f"voted empty by >=1 view but not FREE: "
              f"{float(((vfr > 0) & ~FREE[sx, sy, sz]).sum())/max(len(verts),1)*100:.1f}% "
              f"-- if large, our carve is more lenient than the evaluator "
              f"(eval min_views=1, ours {args.free_min_views}-view consensus)")
        bad = []
        if s_free > args.sanity_free_max:
            bad.append(f"{s_free*100:.0f}% of the surface penetrates observed free space "
                       f"by more than {args.sanity_free_depth*1000:.0f}mm")
        if s_disp > args.sanity_disp_max:
            bad.append(f"{s_disp*1000:.0f}mm displaced from the observed surface")
        if bad:
            print("\n" + "!" * 70)
            print("[sanity] the fused result contradicts the observation -- aborting.")
            for b_ in bad:
                print(f"  - {b_}")
            if prior_applied:
                print("  check, in order: 1) npz center/scale/R_align belong to this object")
                print("                   2) pkl T_model_world and bounds "
                      "(dump_shaper_points.py prints the world bbox)")
                print("                   3) the SAM3 instance does not span several "
                      "GT objects")
            else:
                print("  NOTE the gate blocked the prior, so this output contains no "
                      "generated geometry.")
                print("    The cause is on the observation side: check the mask, the poses, "
                      "or GT-depth alignment.")
            print("  disable this check with --no_sanity")
            print("!" * 70, flush=True)
            sys.exit(2)

    # [debug] Colour the generated surface samples by class, to see why a leg was cut:
    # green = unknown (kept), blue = observation-dominated, yellow = other object,
    # red = free-carve
    if getattr(args, "debug_class_ply", ""):
        # the --prior_field path has no mesh, so sample points at the field's zero crossing
        if debug_pts is None:
            # Extract the zero level with marching cubes. A |SG| < threshold test is
            # sensitive to field scale and selected the whole grid on small-valued fields,
            # producing a solid cube.
            if SG.min() < 0 < SG.max():
                dv, _, _, _ = marching_cubes(SG, level=0.0, spacing=(step,) * 3)
                debug_pts = (dv - 1.0) * scale + center
                if len(debug_pts) > 300000:
                    debug_pts = debug_pts[np.random.choice(len(debug_pts), 300000,
                                                           replace=False)]
                print(f"[debug] {len(debug_pts)} points on the prior zero-level "
                      f"(SG range [{SG.min():.4f}, {SG.max():.4f}])")
            else:
                print(f"  [debug] the prior field has no zero crossing "
                      f"(SG range [{SG.min():.4f}, {SG.max():.4f}]) -- generation failed "
                      f"or the gate disabled it. Skipping the point cloud.")
                debug_pts = np.zeros((0, 3))
    if getattr(args, "debug_class_ply", "") and len(debug_pts):
        idx = np.clip(np.round(((debug_pts - center) / scale + 1) / step), 0, G - 1).astype(int)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        cls = np.zeros(len(debug_pts), int)
        cls[~HULL[i, j, k]] = 4                      # outside the hull (mask-cone violation)
        cls[FREE[i, j, k]] = 3
        cls[OTH[i, j, k] & ~FREE[i, j, k]] = 2
        cls[alpha[i, j, k] > 0.5] = 1
        pal = np.array([[0.1, 0.8, 0.1], [0.2, 0.4, 1.0], [1.0, 0.8, 0.1],
                        [1.0, 0.15, 0.15], [0.7, 0.1, 0.9]])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(debug_pts)
        pc.colors = o3d.utility.Vector3dVector(pal[cls])
        dp = os.path.expanduser(args.debug_class_ply)
        os.makedirs(os.path.dirname(dp) or ".", exist_ok=True)
        o3d.io.write_point_cloud(dp, pc)
        print(f"[debug] class cloud: {dp}  keep {(cls == 0).mean()*100:.0f}%  "
              f"obs {(cls == 1).mean()*100:.0f}%  oth {(cls == 2).mean()*100:.0f}%  "
              f"free {(cls == 3).mean()*100:.0f}%  out-of-hull {(cls == 4).mean()*100:.0f}%  "
              f"(green=keep, blue=observed, yellow=other, red=carve, purple=out-of-hull)")
    return verts, faces


def main():
    parser = ArgumentParser(description="Depth-based SDF distillation (TSDF replacement)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # mapping of the render.py TSDF options
    parser.add_argument("--depth_trunc", default=6.0, type=float, help="max depth (back-projection cutoff)")
    parser.add_argument("--voxel_size", default=0.005, type=float, help="marching-cubes voxel size (sets the grid)")
    parser.add_argument("--sdf_trunc", default=0.04, type=float, help="reference only; unused on the SDF path")
    parser.add_argument("--num_cluster", default=10000, type=int, help="clusters to keep in post-processing (clamped)")
    # SDF options
    parser.add_argument("--alpha_thr", default=0.5, type=float, help="drop pixels with alpha below this (background / floaters)")
    parser.add_argument("--pts_per_view", default=40000, type=int)
    parser.add_argument("--n_pts", default=1500000, type=int, help="cap on surface points used for fitting (subsampled)")
    parser.add_argument("--pts_seed", default=0, type=int,
                        help="seed for point-cloud subsampling. Without it the same command "
                             "produces a different center/scale each run and results drift "
                             "(measured: gate 34.6%% vs 38.4%%). Always compare settings at "
                             "the same seed.")
    parser.add_argument("--sdf_iters", default=10000, type=int)
    parser.add_argument("--batch", default=16384, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--pe_L", default=0, type=int, help="positional-encoding level (0 = off, recommended)")
    parser.add_argument("--w_normal", default=1.0, type=float)
    parser.add_argument("--w_sign", default=1.0, type=float)
    parser.add_argument("--w_eik", default=0.5, type=float)
    parser.add_argument("--w_free", default=1.0, type=float,
                        help="weight for near-surface free-space carving (0 = off)")
    parser.add_argument("--free_range", default=0.5, type=float,
                        help="max carve distance from an observed point toward its camera "
                             "(normalised coords)")
    parser.add_argument("--w_empty", default=1.0, type=float,
                        help="weight for empty-ray carving (0 = off). Forces SDF >= 0 along "
                             "the bbox chord of alpha~0 rays: keeps real holes, removes "
                             "inflation")
    parser.add_argument("--empty_per_view", default=4096, type=int)
    parser.add_argument("--empty_alpha", default=0.1, type=float,
                        help="treat pixels below this alpha as empty rays. Relax to 0.3-0.5 "
                             "when junk gaussians raise alpha everywhere")
    parser.add_argument("--tile", default=0, type=int,
                        help="tiled marching block size (e.g. 128). 0 = one volume. Required "
                             "for high-resolution whole-scene runs")
    parser.add_argument("--extra_points", default="", type=str,
                        help="generated-view point cloud ply with normals; injects unseen "
                             "surface evidence (make_gen_points.py)")
    parser.add_argument("--prior_repeat", default=1.0, type=float,
                        help="< 1 subsamples the prior points by this ratio (lower trust)")
    parser.add_argument("--prior_weight", default=1.0, type=float,
                        help="surface-loss weight for extra_points (prior). < 1 makes it a "
                             "soft prior dominated by observation")
    parser.add_argument("--prior_field", default="", type=str,
                        help="recommended: ShapeR signed-SDF grid npz (shaper_field.py). "
                             "It skips the mesh, so sign-fix, shell_delta and alignment are "
                             "all unnecessary")
    parser.add_argument("--prior_field_rescale", default=1, type=int,
                        help="rescale the prior field's saturation value to --prior_trunc "
                             "(1 = on). Prevents fake surfaces from a step at the carve "
                             "boundary; the zero crossing is unchanged")
    parser.add_argument("--prior_sigma_w", default=0.0, type=float,
                        help="ensemble sigma weighting (0 = off). Larger values suppress "
                             "generation where seeds disagree: precision up, recall down")
    parser.add_argument("--prior_sigma_ref", default=0.0, type=float,
                        help="sigma reference (m). 0 = median sigma near the zero crossing")
    parser.add_argument("--prior_mesh", default="", type=str,
                        help="aligned watertight generated mesh (*_gen_aligned.ply from "
                             "fuse_generated_mesh --save_aligned). Surface + volume SDF "
                             "distillation; supersedes extra_points")
    parser.add_argument("--w_prior_sdf", default=0.5, type=float,
                        help="loss weight for volume SDF distillation (0 = off)")
    parser.add_argument("--prior_surf_n", default=150000, type=int,
                        help="surface samples on the prior mesh")
    parser.add_argument("--prior_band", default=0.08, type=float,
                        help="half-width of the volume shell samples (world m). Use 2-3x the "
                             "thinnest leg")
    parser.add_argument("--prior_trunc", default=0.05, type=float,
                        help="target SDF truncation (world m)")
    parser.add_argument("--prior_unseen_dist", default=0.03, type=float,
                        help="drop prior samples within this distance (m) of an observed "
                             "point: observation always wins")
    parser.add_argument("--prior_gate", default=0.02, type=float,
                        help="drop carve/empty samples within this distance (m) of the prior "
                             "mesh surface")
    parser.add_argument("--prior_uniform_n", default=200000, type=int,
                        help="uniform far-field volume samples; stops inflation in unseen "
                             "space outside the shell")
    parser.add_argument("--prior_carve_views", default=150, type=int,
                        help="evenly spaced views buffered for carving prior hallucinations "
                             "(0 = off). grid_fuse also builds the observed TSDF from them")
    parser.add_argument("--prior_carve_margin", default=0.015, type=float,
                        help="carve depth margin (m): this far in front of an observed "
                             "surface counts as a free-space violation")
    parser.add_argument("--grid_fuse", action="store_true",
                        help="deterministic grid TSDF fusion instead of the MLP, with "
                             "priority observation > carve > generated. Requires "
                             "--prior_mesh; use --prior_carve_views 120+. Removes inflation "
                             "and sponge artefacts by construction")
    parser.add_argument("--passthrough_mesh", default="", type=str,
                        help="mesh returned verbatim when a gate blocks the prior, normally "
                             "the observed reconstruction fuse_post.ply. Re-fusing an object "
                             "with nothing to complete only costs quality: measured obj16 "
                             "seen F@1 0.914 -> 0.647, obj10 free 5.4%% -> 27.1%%. Empty "
                             "string re-fuses")
    parser.add_argument("--no_alpha_full_wo_prior", action="store_true",
                        help="keep the alpha blend even when the prior is blocked (legacy). "
                             "The only blend partner is trunc, so the surface erodes inward "
                             "by up to trunc. A/B use only")
    parser.add_argument("--fuse_dtype", default="float64", type=str,
                        choices=["float32", "float64"],
                        help="GPU fusion precision. The CPU path is float64, so use float64 "
                             "to compare. float32 differs by 1e-7, which the gate "
                             "(alpha<0.25) and keep_connected amplify because both are "
                             "threshold statistics (measured gate 34.6%% vs 38.8%%). FP64 is "
                             "only 2x slower than FP32 on A100, hence the default")
    parser.add_argument("--fuse_device", default="auto", type=str,
                        help="device for grid fusion: auto|cuda|cpu. The GPU path uses the "
                             "same formulas and is 50-100x faster (28 min -> tens of "
                             "seconds). The full grid stays on CPU and only slabs are "
                             "uploaded, so the GPU holds ~1.5GB. Force cpu to cross-check")
    parser.add_argument("--grid_wcap", default=8.0, type=float,
                        help="view count at which observation confidence saturates: above it "
                             "a voxel is 100%% observed TSDF. Lower gives observation more "
                             "authority; higher hands the thin boundary band seen by only a "
                             "few views over to the prior")
    # [wcap evidence] Over 5 objects (6/1/22/2/16), 5 -> 8 won clearly: seen acc -0.49mm and
    #   free violation -0.97%p (22x and 11x the noise floor). Extending to 8/16/32 on 3
    #   objects (6/22/2) shows it flattens at 8:
    #                      8      16      32
    #     seen acc      4.250   4.190   4.128   <- 0.06mm per doubling, negligible
    #     uns comp      23.35   25.10   25.96   <- worse
    #     free viol     10.40   10.66   10.73   <- worse
    #   With 200 views a well-observed voxel has Wo in the tens to hundreds, so raising wcap
    #   only hands the thin boundary band to the prior. Observation weighting DELETED that
    #   band and lost (seen F@1 0.959 -> 0.923); wcap DELEGATES it and wins.
    #   obj6 alone made 5-8 look like a plateau -- the classic single-object tuning trap.
    # -- observation-confidence weighting (off by default) --------------------
    # Added to fix edge bleed on obj6, where it worked (free violation 4.93 -> 1.65%), but a
    # 5-object sweep (6/1/22/2/16) showed a net loss, so the default is off.
    #
    #   prior ON, median over 5 objects   baseline   full  noerode   none(=off)
    #     seen F@1cm                    0.924   0.923    0.933   0.959
    #     unseen F@2cm                  0.214   0.344    0.351   0.371
    #     unseen completion(mm)         125.5   23.29    23.44   23.18
    #     free violation (%)                  3.81    4.42     4.49    5.80
    #
    # The same held with the prior off (nothing to fill the discarded observations):
    #     seen F@1cm  full 0.857 / nocos 0.866 / noerode 0.895 / none 0.960
    #   Per-component: erosion +0.038, cos gate +0.009, all off +0.103 -- more than the
    #   sum, so |cos| weighting itself dominates (it halves Wo and thus lowers alpha).
    #
    # The one axis it improves is free violation. If that is all you need, --obs_erode 0
    # --obs_cos_min 0.2 (noerode) is the compromise: best seen acc at 4.381mm and free
    # violation 4.49% against 5.80% for none.
    parser.add_argument("--obs_erode", default=0, type=int,
                        help="erode the object mask by N pixels for TSDF integration, "
                             "excluding border pixels that flicker between fg and bg. 2-3")
    parser.add_argument("--obs_cos_min", default=0.0, type=float,
                        help="exclude grazing pixels with |cos(view, normal)| below this. "
                             "Depth-discontinuity pixels have cos ~ 0 and are caught too. "
                             "0.15-0.3")
    parser.add_argument("--obs_cos_weight", dest="obs_cos_weight", action="store_true",
                        default=False,
                        help="weight by |cos| so head-on views beat grazing ones. Off by "
                             "default: it cost the most seen F@1cm in the 5-object sweep")
    parser.add_argument("--no_obs_cos_weight", dest="obs_cos_weight", action="store_false",
                        help="disable cos weighting (default)")
    # -- re-apply the hard free-space constraint ------------------------------
    parser.add_argument("--free_hard", dest="free_hard", action="store_true",
                        default=True,
                        help="re-apply the carve (FREE) constraint after the blend (on by "
                             "default). In the blend base only gets a (1-alpha) share, so "
                             "the carve is ignored wherever observation is strong. Measured "
                             "on 4 objects: free violation +2.23%%p -> +0.33%%p, and obj10 "
                             "went 21.1%% -> 5.8%% while unseen F@2 rose 0.593 -> 0.679")
    parser.add_argument("--no_free_hard", dest="free_hard", action="store_false",
                        help="do not re-apply the carve; observation always wins (legacy)")
    parser.add_argument("--free_hard_oth", action="store_true",
                        help="apply free_hard to other-object (OTH) voxels too. OTH is a mask "
                             "vote, so it wavers at a neighbour boundary and can delete real "
                             "surface (measured obj2: seen F@1 +0.042 -> -0.062). FREE only "
                             "by default")
    parser.add_argument("--free_hard_alpha", default=0.95, type=float,
                        help="re-apply the carve only where alpha is below this. 0.95 = skip "
                             "only fully saturated voxels (Wo >= grid_wcap), the default")
    # [free_hard_alpha evidence] 4-object sweep. With observation weighting off, Wo is an
    #   integer view count and alpha = Wo/wcap, so the threshold means "up to how many views
    #   may the carve override".
    #     thr    applies to       seen acc  seen F@1  uns F@2  free%
    #     0.50   Wo <= 3 views      3.836    0.947    0.527   6.350
    #     0.80   Wo <= 6 views      3.855    0.947    0.529   5.915
    #     0.95   Wo <= 7 views      3.908    0.947    0.530   5.275   <- chosen
    #     1.01   Wo <= 8 (all)      4.655    0.923    0.532   4.505   <- cliff
    #   From 0.5 to 0.95 free falls monotonically at zero cost to seen. Only 1.01 collapses,
    #   and entirely on one object (obj2, 0.924 -> 0.864): when 8+ views say "surface here",
    #   they are right. alpha is discrete in {0, 1/8, ..., 1}, so 0.95 and 0.99 are the same.
    #
    #   Rule: the carve beats observation, except where observation is fully saturated
    #   (Wo >= wcap). Changing grid_wcap changes what this threshold means -- they are tied.
    # -- sanity checks for a misplaced prior (batch safety net) ----------------
    parser.add_argument("--sanity_free_max", default=0.25, type=float,
                        help="max fraction of the output surface allowed to penetrate free "
                             "space by more than sanity_free_depth; abort above it. Measured "
                             "by depth: obj22 0.2%%, obj28 5.7%% -- healthy is single digits. "
                             "The old rounding test read over 25%% even when healthy and "
                             "killed obj28 as a false positive")
    parser.add_argument("--sanity_disp_max", default=0.05, type=float,
                        help="max median displacement (m) of the output surface from the "
                             "observed TSDF surface, inside observed regions. Measured: "
                             "healthy 4-8mm, obj20 116mm")
    parser.add_argument("--sanity_free_depth", default=0.015, type=float,
                        help="only count surface penetrating free space by at least this (m). "
                             "Must match --margin in eval_seen_unseen.py to be comparable")
    parser.add_argument("--no_sanity", action="store_true",
                        help="disable the sanity checks; a misplaced prior is then shipped as is")
    parser.add_argument("--obs_max_reject", default=0.35, type=float,
                        help="max fraction of observed pixels that may be rejected; above it "
                             "--obs_erode drops by 1 (protects small, thin objects)")
    parser.add_argument("--grid_smooth", default=0.7, type=float,
                        help="gaussian smoothing sigma (voxels) on the fused grid. 0 = off")
    parser.add_argument("--gt_depth_dir", default="", type=str,
                        help="GT depth folder (e.g. nice-slam results). With it the carve runs "
                             "against real scene depth instead of the mask, which stops legs "
                             "being cut and edges being ragged")
    parser.add_argument("--gt_depth_scale", default=6553.5, type=float,
                        help="GT depth PNG scale (pixel value / scale = metres). Replica nice-slam = 6553.5")
    parser.add_argument("--prior_carve_ds", default=1, type=int,
                        help="view-buffer downscale. Keep at 1: grid_fuse builds the "
                             "OBSERVED TSDF from these buffers, not just the prior carve. "
                             "At ds=2 the fused surface sits ~8.8mm from the observed mesh "
                             "everywhere (only 8%% matches within 2mm) and seen F@1 falls "
                             "0.824 -> 0.680; at ds=1 it is 5.6mm and -0.015. No other knob "
                             "moves this: prior on/off, free_hard, grid_wcap, depth_ratio, "
                             "prior_trunc and voxel_size all measured flat, including voxel "
                             "and trunc set to render.py's own values. View COUNT does not "
                             "matter (150 vs 554 views: identical); resolution does.")
    parser.add_argument("--free_min_views", default=2, type=int,
                        help="views that must agree before a voxel counts as free. 1 is an OR "
                             "(aggressive); 3+ keeps depth-edge noise from eating thin parts")
    parser.add_argument("--gt_edge_thr", default=0.0, type=float,
                        help="GT-depth discontinuity threshold (m/px); pixels above it cast no "
                             "free vote. 0 = off. Enabling it narrows the carve and raises "
                             "free violations (obj22 7.5%% -> 10.8%%). Try 0.1 only on data "
                             "where edges genuinely bleed, e.g. nearest-resized depth")
    parser.add_argument("--debug_class_ply", default="", type=str,
                        help="path to save the class-coloured surface cloud, for diagnosing why parts are cut")
    parser.add_argument("--carve_align", action="store_true",
                        help="9-DoF re-optimisation so the generated shape avoids free space "
                             "and settles into unknown space, keeping observed anchors. "
                             "Absorbs shape mismatch between generation and observation")
    parser.add_argument("--carve_align_w", default=1.0, type=float,
                        help="weight of the observed anchor in carve-align; higher aligns the top plate more strictly")
    parser.add_argument("--probe_box", default="", type=str,
                        help='diagnostic world box "x0,y0,z0,x1,y1,z1"; prints voxel-class stats inside it')
    parser.add_argument("--grid_sign_fix", action="store_true",
                        help="force-restore the generated SDF sign by flood fill (automatic when watertight is False)")
    parser.add_argument("--shell_delta", default=0.02, type=float,
                        help="max offset-shell half-thickness d_max (m), used deep in unobserved space (legs)")
    parser.add_argument("--shell_delta_min", default=0.006, type=float,
                        help="min offset-shell half-thickness d_min (m), used next to the observed surface (rims)")
    parser.add_argument("--shell_ramp", default=0.10, type=float,
                        help="transition distance from d_min to d_max (m, measured from the observed surface)")
    parser.add_argument("--alpha_smooth", default=1.0, type=float,
                        help="gaussian sigma (voxels) applied to alpha, creating an "
                             "observed/generated transition band that removes the seam step. "
                             "0 = off")
    parser.add_argument("--unseen_open", default=0.0, type=float,
                        help="morphological opening radius (m) in unobserved regions; removes "
                             "protrusions up to 2r thick. Off by default because it also "
                             "deletes thin structure such as table legs -- check they survive")
    parser.add_argument("--no_color_match", action="store_true",
                        help="disable matching the generated colour to observed colour statistics")
    parser.add_argument("--color_blend_ramp", default=0.05, type=float,
                        help="seam colour blend distance (m): blend toward the observed colour "
                             "up to this distance from the observed surface. 0 = off")
    parser.add_argument("--hull_min_frac", default=0.0, type=float,
                        help="allow the prior only in voxels where at least this fraction of "
                             "in-frustum views project inside the object mask. 0 = off. "
                             "Stops generated geometry leaking into the floor or a neighbour")
    parser.add_argument("--hull_min_views", default=5, type=int,
                        help="minimum in-frustum views before the hull test applies (drops voxels with too little evidence)")
    parser.add_argument("--view_stride", default=1, type=int,
                        help="use every Nth training view (back-projection and carve cost scale "
                             "with view count). The point cloud is subsampled anyway, so 2-4 "
                             "costs little")
    # [gate] Validated on a 21-object batch. It was once disabled (0) and then restored.
    #
    #   Why it was disabled: the statistic is noisy. unknown = (alpha<0.25)&~FREE&~OTH is
    #   aggregated over the thin shell |SG| < 1.5 vox, so boundary voxels with Wo near 2
    #   decide the outcome. Three runs of the same command gave 38.4 / 38.8 / 40.0%.
    #
    #   Why it was restored: with it off, exactly the objects it used to block collapsed.
    #     obj16 (picture frame)  unseen F@2 0.358 -> 0.073,  seen F@1 0.914 -> 0.646
    #     obj8  (vase)           unseen F@2 0.070 -> 0.025,  free 7.6 -> 33.6%
    #     obj10                  unseen F@2 0.584 -> 0.564,  free 5.4 -> 21.9%
    #   The decision was right; only the statistic was noisy.
    #
    #   Threshold, re-measured on prior_carve_ds=1 inputs (gate_stat_check.py, 23 objects).
    #   The earlier 0.10 was chosen on ds=2 data; fixing the view buffers shifted every
    #   ufrac, so the threshold had to be re-picked.
    #     thr    applied-ok  wasted  blocked-ok  missed  net d(unsF2)
    #     0.10      13      3      0       0      +1.240
    #     0.125     13      2      1       0      +1.525   <- chosen
    #     0.15      12      2      1       1      +1.345
    #     0.30      10      2      1       3      +0.877
    #   The decision rests on two adjacent objects:
    #     obj16 (picture frame)  ufrac 10.2%  d unsF2 -0.287  d seenF1 -0.210   block it
    #     obj14                  ufrac 14.7%  d unsF2 +0.182  d seenF1 +0.011   keep it
    #   0.10 lets obj16 through and 0.15 sits 0.3%p from obj14 -- both inside the +-1.6%p
    #   run-to-run drift, i.e. a coin flip. 0.125 is the midpoint and leaves 2.2%p on each
    #   side. Effect on the 23-object mean: seen F@1 +0.010 -> +0.020, seen accuracy
    #   -0.021 -> -0.173mm, unseen F@2 +0.052 -> +0.065, free +0.455 -> +0.520%p.
    #
    #   The alternative statistic, "unobserved fraction of the generated INTERIOR volume",
    #   is worse at every threshold (best net +0.713 vs +1.525) and is not even printed for
    #   gate-blocked objects, since fusion returns before that line.
    #   Remaining errors, not separable by any threshold: obj24 (ufrac 55.7%, d unsF2
    #   -0.073) and obj12 (67.3%, -0.022).
    parser.add_argument("--min_obs_frac", default=0.05, type=float,
                        help="minimum fraction of observed voxels for the prior to apply. "
                             "Below it the prior is unconstrained and invents the object: "
                             "obj31 at 1.0%% and obj28 at 2.1%% blew up to 69.9 / 77.6mm seen "
                             "accuracy, while obj22 at 7.7%% improved. 0 disables the gate; "
                             "a blocked object falls through to --passthrough_mesh.")
    parser.add_argument("--min_unknown_frac", default=0.125, type=float,
                        help="skip the prior when the unknown fraction of the generated surface "
                             "is below this (the object is already well observed). "
                             "0 = always apply")
    parser.add_argument("--keep_connected", dest="keep_connected", action="store_true",
                        default=True,
                        help="keep only the negative components connected to observed voxels, "
                             "removing another object's debris from a single-blob prior "
                             "(on by default)")
    parser.add_argument("--no_keep_connected", dest="keep_connected", action="store_false",
                        help="disable the connected-component filter (for objects with detached parts)")
    parser.add_argument("--carve_depth_dir", default="", type=str,
                        help="dump_scene_depth.py output folder. Carves free space from whole-scene depth; takes precedence over empty-ray")
    parser.add_argument("--offsurf_delta", default=0.01, type=float, help="off-surface offset in normalised coordinates")
    parser.add_argument("--grid", default=0, type=int, help="marching-cubes resolution (0 = derived from voxel_size)")
    parser.add_argument("--max_grid", default=512, type=int)
    parser.add_argument("--mask_dist", default=0.0, type=float,
                        help="drop mesh vertices farther than this (world) from the observed "
                             "cloud; 0 = off. Trades box removal against hole filling. To keep "
                             "unseen completion (sides and back), use 0 or a large value "
                             "together with an ROI crop")
    parser.add_argument("--roi_mesh", default="", type=str,
                        help="observed anchor mesh (e.g. the TSDF fuse_post.ply). Points beyond roi_dist from it are dropped from the SDF input")
    parser.add_argument("--roi_dist", default=0.15, type=float)
    parser.add_argument("--mask_dir", default="auto", type=str,
                        help="per-view object mask folder. 'auto' = <source_path>/masks when present, '' = none")
    parser.add_argument("--require_mask", dest="require_mask", action="store_true",
                        default=True,
                        help="skip training views that have no mask (required for per-object extraction, on by default)")
    parser.add_argument("--no_require_mask", dest="require_mask", action="store_false",
                        help="also use views without a mask (whole-scene extraction)")
    parser.add_argument("--extra_poses", default="", type=str,
                        help="extra novel poses npz (render_hole_novel soft_out poses.npz), to "
                             "include See3D-refined unseen bands in the extraction")
    parser.add_argument("--extra_mask_npy", default="", type=str,
                        help="per-gaussian object label npy. Used to render a label buffer at "
                             "extra poses and build object masks there (blocks floor bleed). "
                             "Without it the extra views are unmasked")
    parser.add_argument("--extra_mask_thr", default=0.3, type=float)
    parser.add_argument("--out", default="", type=str)
    args = get_combined_args(parser)

    # -- reproducibility -------------------------------------------------------
    # Subsampling the observed cloud P without a seed changes the percentile-derived
    # center/scale every run, so the whole voxel grid lands in a slightly different place.
    # Measured (obj6, same settings twice): [gate] 34.6% vs 38.4%, 405 vertices apart,
    # Chamfer 0.09mm. Objects near the gate threshold flip on that drift alone (obj28 sat at
    # 18.1%, just below). In an A/B, an unequal seed mixes the setting difference with the
    # sampling difference.
    np.random.seed(args.pts_seed)

    # -- settled settings ------------------------------------------------------
    # Everything settled by tuning lives in the argparse defaults. Only the few that are
    # awkward to express on the command line are handled here, and every value actually in
    # force is printed. (A silent default once changed results and cost days -- unseen_open
    # 0.015 -- so nothing is hidden: the full table is printed every run.)
    _given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}

    if "--data_device" not in _given:
        args.data_device = "cpu"              # saves GPU memory, does not change results
    if args.prior_field and not args.grid_fuse and "--no_grid_fuse" not in _given:
        args.grid_fuse = True                 # passing prior_field already states the intent
    if not args.gt_depth_dir and "--no_gt_depth" not in _given:
        _d = os.environ.get("REFINEGS_GT_DEPTH", DEFAULT_GT_DEPTH_DIR)
        if os.path.isdir(_d):
            args.gt_depth_dir = _d

    print("+- [config] values in force this run " + "-" * 30)
    for _k, _v, _note in [
        ("prior_field",     args.prior_field,      "generated prior field npz"),
        ("prior_sigma_w",   args.prior_sigma_w,    "sigma weighting (0=off, ensemble npz only)"),
        ("grid_fuse",       args.grid_fuse,        "deterministic grid fusion"),
        ("obs_erode",       args.obs_erode,        "mask erosion px [seen quality]"),
        ("obs_cos_min",     args.obs_cos_min,      "grazing rejection threshold [seen quality]"),
        ("obs_cos_weight",  args.obs_cos_weight,   "cos weighting [seen quality]"),
        ("grid_wcap",       args.grid_wcap,        "views at which observation saturates"),
        ("unseen_open",     args.unseen_open,      "WARN >0 deletes thin structure"),
        ("free_min_views",  args.free_min_views,   "carve consensus views"),
        ("min_unknown_frac", args.min_unknown_frac, "prior gate: unobserved surface"),
        ("min_obs_frac",    args.min_obs_frac,     "prior gate: observed voxels"),
        ("hull_min_frac",   args.hull_min_frac,    "visual hull gate (0=off)"),
        ("keep_connected",  args.keep_connected,   "connected-component filter"),
        ("free_hard",       args.free_hard,        "re-apply carve after the blend"),
        ("free_hard_alpha", args.free_hard_alpha,  "skip fully saturated observation (0.95)"),
        ("voxel_size",      args.voxel_size,       "voxel size (m)"),
        ("prior_carve_ds",  args.prior_carve_ds,   "view-buffer downscale (keep 1)"),
        ("gt_depth_dir",    args.gt_depth_dir,     "GT depth (carve reference)"),
        ("pts_seed",        args.pts_seed,         "point sampling seed (reproducibility)"),
        ("fuse_device",     args.fuse_device,      "fusion device"),
    ]:
        _src = "set" if f"--{_k}" in _given or f"--no_{_k}" in _given else "default"
        print(f"| {_k:<17} = {str(_v):<28} [{_src}] {_note}")
    print("+" + "-" * 62)
    if args.unseen_open > 0:
        print("[config] WARN unseen_open > 0: morphological opening will delete thin "
              "structure in unobserved regions (table legs). Confirm this is intended.")

    _T0 = time.time(); _tk = _T0

    def _lap(msg):
        now = time.time()
        print(f"[time] {msg}: {now - _lap.prev:.1f}s (total {now - _T0:.1f}s)", flush=True)
        _lap.prev = now
    _lap.prev = _tk

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    _lap(f"scene load ({len(scene.getTrainCameras())} training views)")
    gaussians.active_sh_degree = 0  # diffuse only, same as the render.py mesh path
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    # 1) oriented point cloud (pixels outside the object mask dropped, as in the TSDF path)
    mask_dir = None
    if args.mask_dir == "auto":
        cand = os.path.join(dataset.source_path, "masks")
        mask_dir = cand if os.path.isdir(cand) else None
    elif args.mask_dir:
        mask_dir = args.mask_dir

    extra_cams = []
    if args.extra_poses:
        from scene.cameras import MiniCam
        recs = np.load(os.path.expanduser(args.extra_poses), allow_pickle=True)["records"]
        for r in recs:
            wvt = torch.tensor(np.asarray(r["world_view_transform"]), dtype=torch.float32).cuda()
            fpt = torch.tensor(np.asarray(r["full_proj_transform"]), dtype=torch.float32).cuda()
            mc = MiniCam(int(r["width"]), int(r["height"]),
                         float(r["FoVy"]), float(r["FoVx"]), 0.01, 100.0, wvt, fpt)
            mc.image_name = f"extra{int(r['idx']):04d}"
            extra_cams.append(mc)

    extra_masks = None
    if extra_cams and args.extra_mask_npy:
        lab_np = np.load(os.path.expanduser(args.extra_mask_npy)).astype(np.float32)
        assert len(lab_np) == gaussians.get_xyz.shape[0], \
            f"extra_mask_npy {len(lab_np)} != gaussians {gaussians.get_xyz.shape[0]}"
        extra_masks = render_extra_masks(extra_cams, gaussians, pipe, background,
                                         torch.from_numpy(lab_np).cuda(), thr=args.extra_mask_thr)
        cov = np.mean([m.float().mean().item() for m in extra_masks.values()])
        print(f"[extra] built masks for {len(extra_masks)} views (mean cover {cov*100:.1f}%)")

    print("back-projecting depth and aligning normals ...")
    P, N, C, O, EO, ED, VB = collect_oriented_points(scene, gaussians, pipe, background, args,
                                                     mask_dir=mask_dir,
                                                     require_mask=args.require_mask,
                                                     extra_cams=extra_cams,
                                                     extra_masks=extra_masks)
    _lap("depth back-projection")
    print(f"[points] {len(P)} observed surface points")
    if len(P) < 1000:
        raise SystemExit(f"[abort] only {len(P)} valid surface points -- check the mask value "
                         f"convention or alpha_thr (test without masks: --mask_dir '', "
                         f"relax alpha: --alpha_thr 0.5)")

    # 1b) ROI crop: keep only points near a trusted observed mesh (the TSDF fuse_post),
    #     removing background and mask-border junk before the SDF fit.
    if args.roi_mesh:
        from scipy.spatial import cKDTree as _KD
        rm = o3d.io.read_triangle_mesh(args.roi_mesh)
        rv = np.asarray(rm.vertices)
        assert len(rv) > 0, f"ROI mesh is empty: {args.roi_mesh}"
        d, _ = _KD(rv).query(P, workers=-1)
        keep = d < args.roi_dist
        print(f"[roi] kept {int(keep.sum())}/{len(P)} (dist<{args.roi_dist}, mesh={args.roi_mesh})")
        P, N, C, O = P[keep], N[keep], C[keep], O[keep]

    if len(P) > args.n_pts:
        idx = np.random.choice(len(P), args.n_pts, replace=False)
        P, N, C, O = P[idx], N[idx], C[idx], O[idx]

    # robust normalisation to [-1,1]: drop points outside the percentile bbox so floaters
    # cannot inflate scale
    lo = np.percentile(P, 0.5, axis=0); hi = np.percentile(P, 99.5, axis=0)
    pad = 0.05 * (hi - lo)
    keep = np.all((P >= lo - pad) & (P <= hi + pad), axis=1)
    n_drop = int((~keep).sum())
    P, N, C, O = P[keep], N[keep], C[keep], O[keep]
    center = (lo + hi) / 2
    scale = np.abs(P - center).max() * 1.1
    print(f"[bbox] dropped {n_drop} outliers, scale={scale:.3f} world (bbox {np.round(hi-lo,3)})")
    # Pn/On are computed AFTER the prior is injected. Computing them here fed a stale Pn to
    # train(), so prior points never reached the surface loss -- legs went unsupervised and
    # inflated.
    EOn = (EO - center) / scale if len(EO) else EO   # directions (ED) are invariant to normalisation
    if len(EOn):
        t0 = -(EOn * ED).sum(-1)
        dmin = np.linalg.norm(EOn + ED * t0[:, None], axis=-1)
        print(f"[empty-ray] median t0 (forward distance) {np.median(t0):.2f} (should be "
              f"positive), dmin (closest approach to centre) min/median "
              f"{dmin.min():.2f}/{np.median(dmin):.2f} (normalised units, bbox ~ 1)")
        keep_e = (t0 > 0) & (dmin < 1.2)             # keep only rays that actually pass near the object bbox
        EOn, ED = EOn[keep_e], ED[keep_e]
        print(f"[empty-ray] kept {int(keep_e.sum())}/{len(keep_e)} rays crossing the bbox")

    n_obs = len(P)
    OBS = np.ones(n_obs, bool)          # True = observed point; prior points are False and excluded from l_free

    # 1b-2) prior, point-cloud path: inject the generated-view cloud (make_gen_points.py /
    #       fuse --export_points). Marked OBS=False so their fake origin (O = center) cannot
    #       free-carve the interior.
    if args.extra_points:
        pp = o3d.io.read_point_cloud(os.path.expanduser(args.extra_points))
        Pe = np.asarray(pp.points)
        Ne = np.asarray(pp.normals) if pp.has_normals() else None
        Ce = np.asarray(pp.colors) if pp.has_colors() else np.tile([0.6, 0.6, 0.6], (len(Pe), 1))
        assert Ne is not None and len(Ne) == len(Pe), "extra_points needs normals (use make_gen_points.py)"
        if 0 < args.prior_repeat < 1.0:
            sel = np.random.choice(len(Pe), max(int(len(Pe) * args.prior_repeat), 1), replace=False)
            Pe, Ne, Ce = Pe[sel], Ne[sel], Ce[sel]
        P = np.concatenate([P, Pe]); N = np.concatenate([N, Ne])
        C = np.concatenate([C, Ce]); O = np.concatenate([O, np.tile(center, (len(Pe), 1))])
        OBS = np.concatenate([OBS, np.zeros(len(Pe), bool)])
        print(f"[prior] injected {len(Pe)} points ({len(P)} total)")
        n_extra = len(Pe)
    else:
        n_extra = 0

    # 1b-3) prior, mesh path: an aligned watertight generated mesh (*_gen_aligned.ply from
    #       fuse_generated_mesh --save_aligned). Regresses a target SDF from surface samples
    #       (face normals, unseen-gated) plus volume shell samples.
    PV = PS = None
    rs_prior = None
    _sd = None
    prior_dbg = None

    # 1b-3') prior, field path: inject the ShapeR decoder's SIGNED SDF grid directly.
    #   Skipping the mesh removes three problems:
    #     - no sign-fix needed (the field is already signed)
    #     - no shell_delta needed (there is no zero-thickness sheet: a ShapeR UDF mesh is
    #       extracted at |f| = iso and gains a shell on both sides; that step is skipped)
    #     - no alignment needed (ShapeR is metric and outputs a world transform)
    #   grid_fuse then fills SG from this field by trilinear interpolation.
    if args.prior_field:
        z = np.load(os.path.expanduser(args.prior_field))
        Ffield = z["field"].astype(np.float32)
        f_center = z["center"]; f_R = z["R_align"]; f_scale = float(z["scale"])
        Gf = Ffield.shape[0]
        print(f"[prior-field] {os.path.basename(args.prior_field)}  G={Gf}  "
              f"voxel={float(z['vox_world'])*1000:.2f}mm  "
              f"inside {(Ffield < 0).mean()*100:.2f}%  "
              f"range [{Ffield.min():.4f}, {Ffield.max():.4f}]m")

        # [truncation rescale] The decoder saturates at a different value per object (obj1
        # +-27mm, obj20 +-6mm), while fusion fills carved voxels with +prior_trunc (50mm).
        # The smaller the saturation, the sharper the jump at the carve boundary, and
        # smoothing then creates an ARTIFICIAL zero crossing (a fake surface) there. The
        # zero crossing is invariant to scaling, so rescale the saturation to prior_trunc
        # and make the two fields commensurate.
        if args.prior_field_rescale:
            sat = float(np.percentile(np.abs(Ffield), 99.5))
            if sat > 1e-9 and abs(sat - args.prior_trunc) / args.prior_trunc > 0.2:
                Ffield = (Ffield * (args.prior_trunc / sat)).astype(np.float32)
                print(f"  -> truncation rescale: saturation {sat*1000:.1f}mm -> "
                      f"{args.prior_trunc*1000:.0f}mm (x{args.prior_trunc/sat:.2f})")
        args.prior_watertight = True          # disable sign-fix; the field is already signed

        # [ensemble sigma weighting] Past ~50% unobserved there is no single right answer.
        # Trust the prior only where seeds agree; where they diverge (large sigma), fall back
        # toward "no surface" (+trunc) and let the natural extension of the observed surface
        # (eikonal, smoothing) take over.
        Fsig = None
        if "field_std" in z.files and args.prior_sigma_w > 0:
            Fsig = z["field_std"].astype(np.float32)
            near = np.abs(Ffield) < 3 * float(z["vox_world"])
            # The field is metric, so sigma0 is an absolute bound on acceptable surface
            # position uncertainty. Default = prior_trunc: sigma << sigma0 gives w ~ 1 (full
            # trust), sigma ~ sigma0 gives w = 0.5. Do NOT set sigma0 to the median sigma --
            # that suppresses half the voxels by definition.
            s0 = max(args.prior_sigma_ref if args.prior_sigma_ref > 0
                     else args.prior_trunc, 1e-6)
            Wsig = 1.0 / (1.0 + args.prior_sigma_w * (Fsig / s0) ** 2)
            sm = float(np.median(Fsig[near])) if near.any() else float("nan")
            print(f"[prior-field] sigma weighting on: sigma0={s0*1000:.1f}mm  "
                  f"median sigma near surface {sm*1000:.2f}mm  "
                  f"median w {float(np.median(Wsig[near])) if near.any() else float('nan'):.3f}  "
                  f"(w<0.5 in {(Wsig < 0.5).mean()*100:.1f}% of voxels)")

        def _sd(q):
            """World point -> approximate metric SDF (negative inside); +trunc outside the grid."""
            n = ((np.asarray(q, np.float64) - f_center) @ f_R.T) * f_scale
            idx = (n + 1.0) * (Gf - 1) / 2.0
            out = np.full(len(idx), args.prior_trunc, np.float64)
            ok = np.all((idx >= 0) & (idx <= Gf - 1.001), axis=1)
            if not ok.any():
                return out
            p = idx[ok]
            i0 = np.floor(p).astype(np.int64); w = p - i0
            i1 = i0 + 1

            def _interp(vol):                 # trilinear interpolation
                v = np.zeros(len(p), np.float64)
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            ww = ((w[:, 0] if dx else 1 - w[:, 0])
                                  * (w[:, 1] if dy else 1 - w[:, 1])
                                  * (w[:, 2] if dz else 1 - w[:, 2]))
                            v += ww * vol[(i1 if dx else i0)[:, 0],
                                          (i1 if dy else i0)[:, 1],
                                          (i1 if dz else i0)[:, 2]]
                return v

            v = _interp(Ffield)
            if Fsig is not None:              # consensus weighting: fall back to "no surface" where uncertain
                wg = _interp(Wsig)
                v = wg * v + (1 - wg) * args.prior_trunc
            out[ok] = v
            return out
    if args.prior_mesh:
        import open3d.core as o3c
        from scipy.spatial import cKDTree as _KDp
        pm_path = os.path.expanduser(args.prior_mesh)
        gm = o3d.io.read_triangle_mesh(pm_path)
        assert len(gm.vertices), f"failed to load prior mesh: {args.prior_mesh}"
        # Bake UV textures (glb) into vertex colours, otherwise the samples come out white.
        # Open3D's "more than 1 material" warning only means materials are dropped when
        # converting to a RaycastingScene; it does not affect the SDF geometry.
        if not (gm.has_vertex_colors() and len(gm.vertex_colors) == len(gm.vertices)):
            try:
                import trimesh
                tm = trimesh.load(pm_path, process=False, force="mesh")
                vc = np.asarray(tm.visual.to_color().vertex_colors)[:, :3] / 255.0
                if len(vc) == len(tm.vertices):
                    gm = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(np.asarray(tm.vertices, np.float64)),
                        o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
                    gm.vertex_colors = o3d.utility.Vector3dVector(np.clip(vc, 0, 1))
                    print(f"[prior] baked texture into vertex colours ({len(vc)} verts)")
            except Exception as e:
                print(f"[prior] colour bake failed ({e}) -- keeping grey")
        wt = gm.is_watertight()
        args.prior_watertight = bool(wt)
        print(f"[prior] mesh verts {len(gm.vertices)} watertight={wt}"
              + ("" if wt else "  WARN unstable signed-distance sign -- grid_fuse applies sign-fix"))
        rs_prior = o3d.t.geometry.RaycastingScene()
        rs_prior.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gm))

        def _sd(q):     # signed distance in world coords (positive outside)
            return rs_prior.compute_signed_distance(
                o3c.Tensor(q.astype(np.float32))).numpy().astype(np.float64)

        obs_tree_w = _KDp(P[:n_obs])
        band, trunc = args.prior_band, args.prior_trunc
        # surface samples: use face normals (generated-mesh vertex normals are unreliable)
        # and verify the sign
        sp = gm.sample_points_uniformly(args.prior_surf_n, use_triangle_normal=True)
        Ps_w = np.asarray(sp.points); Ns_w = np.asarray(sp.normals).copy()
        prior_dbg = Ps_w.copy()                      # [debug] pre-gate copy, for the class visualisation
        flip = _sd(Ps_w + 0.5 * band * Ns_w) < 0     # if p + eps*n is inside, the normal is flipped
        Ns_w[flip] = -Ns_w[flip]
        print(f"[prior] flipped {int(flip.sum())}/{len(Ns_w)} surface normals")
        # [hallucination carve] Validate prior samples against the depth+mask view buffers.
        #   freespace violation:  inside the mask but IN FRONT of the observed surface
        #                         (a blob floating above a table top)
        #   silhouette violation: projects unoccluded onto a pixel outside the mask (a tail)
        def _carve_viol(Xw, margin=args.prior_carve_margin):
            viol = np.zeros(len(Xw), bool)
            for b in VB:
                Xc = Xw @ b["R"].T + b["t"]
                z = Xc[:, 2]
                zz = np.maximum(z, 1e-6)
                u = b["fx"] * Xc[:, 0] / zz + b["cx"]
                v = b["fy"] * Xc[:, 1] / zz + b["cy"]
                infr = (z > 0.05) & (u >= 0) & (u < b["W"]) & (v >= 0) & (v < b["H"])
                ui = np.clip(u, 0, b["W"] - 1).astype(int)
                vi = np.clip(v, 0, b["H"] - 1).astype(int)
                di = b["depth"][vi, ui]; mi = b["mask"][vi, ui]
                front = (di > 0) & (z < di - margin)             # in front of the observed surface
                sil = (~mi) & ((di <= 0) | front)                # outside the silhouette and unoccluded
                viol |= infr & ((mi & front) | sil)
            return viol

        # unseen gate: drop surface samples within tau of an observed point. Observation
        # always wins, which prevents double surfaces.
        du, _ = obs_tree_w.query(Ps_w, workers=-1)
        ku = du > args.prior_unseen_dist
        Cs = (np.asarray(sp.colors)[ku] if len(sp.colors) == len(ku)
              else np.tile([0.6, 0.6, 0.6], (int(ku.sum()), 1)))
        Ps_w, Ns_w = Ps_w[ku], Ns_w[ku]
        # carve surface samples that contradict the observation (hallucinations)
        vv = _carve_viol(Ps_w)
        Ps_w, Ns_w, Cs = Ps_w[~vv], Ns_w[~vv], Cs[~vv]
        print(f"[prior] unseen surface samples {len(Ps_w)}/{len(ku)} "
              f"(tau={args.prior_unseen_dist}m, carved {int(vv.sum())})")
        P = np.concatenate([P, Ps_w]); N = np.concatenate([N, Ns_w])
        C = np.concatenate([C, Cs]);  O = np.concatenate([O, np.tile(center, (len(Ps_w), 1))])
        OBS = np.concatenate([OBS, np.zeros(len(Ps_w), bool)])
        n_extra += len(Ps_w)
        # Volume samples = shell (surface +- band) plus a uniform far field.
        #   shell:   supervises the empty space on BOTH sides of a thin structure as positive
        #   uniform: supervises the rest of the unseen ROI with a truncated SDF, removing the
        #            residual inflation that grew in the unsupervised gaps between shells
        rng = np.random.default_rng(0)
        Xs = [Ps_w + rng.standard_normal(Ps_w.shape) * band * f for f in (0.25, 1.0)]
        Xu = rng.uniform(-1, 1, (args.prior_uniform_n, 3)) * scale + center
        X = np.concatenate(Xs + [Xu])
        sd_x = _sd(X)
        tgt = np.clip(sd_x, -trunc, trunc)
        # carve-violating volume samples: force target=+trunc instead of dropping them
        # (observed empty space is definitively outside)
        vx = _carve_viol(X)
        tgt[vx] = trunc
        dxo, _ = obs_tree_w.query(X, workers=-1)
        kx = dxo > args.prior_unseen_dist            # exclude samples near observations; the observation term owns those
        Xn_ = (X[kx] - center) / scale
        kin = np.all(np.abs(Xn_) < 1.0, axis=1)      # keep only what is inside the normalised cube
        PV = Xn_[kin].astype(np.float64)
        PS = (tgt[kx][kin] / scale).astype(np.float64)
        print(f"[prior] volume distill samples {len(PV)} (shell {len(X)-len(Xu)} + uniform {len(Xu)}, "
              f"carve override {int(vx.sum())}, band={band}m trunc={trunc}m)")

    # normalised coords, computed AFTER the prior injection (the stale-Pn fix)
    Pn = (P - center) / scale
    On = (O - center) / scale

    Wp = np.ones(len(P), np.float32)
    if n_extra and args.prior_weight != 1.0:
        Wp[len(P) - n_extra:] = args.prior_weight
        print(f"[prior] weight {args.prior_weight}: seen {len(P)-n_extra} : prior {n_extra}")

    # 1c) carve samples from whole-scene depth; takes precedence over empty-ray
    CV = None
    if args.carve_depth_dir:
        CV = load_carve_points(args.carve_depth_dir, center, scale)
        print(f"[carve] {len(CV)} samples (whole-scene depth, inside the bbox)")

    # 1d) drop carve/empty samples near the prior mesh (s <= prior_gate), so "force SDF>=0"
    #     and "regress l_prior negative" stop fighting over the same voxel.
    if _sd is not None:
        gate = args.prior_gate
        if CV is not None and len(CV):
            keep_cv = _sd(CV * scale + center) > gate
            print(f"[prior] carve gate: kept {int(keep_cv.sum())}/{len(CV)} (dropped s<={gate}m)")
            CV = CV[keep_cv]
        if len(EOn):
            # turn empty rays into fixed samples on the chord, drop those near the prior,
            # merge into CV and disable the chord path
            K = 4
            t0 = -(EOn * ED).sum(-1, keepdims=True)
            cp = EOn + ED * t0
            half = np.sqrt(np.clip(1.44 - (cp * cp).sum(-1, keepdims=True), 0.0, None))
            ts = np.clip(t0 + (np.random.rand(len(EOn), K) * 2 - 1) * half, 0.05, None)
            Xe = (EOn[:, None, :] + ED[:, None, :] * ts[..., None]).reshape(-1, 3)
            Xe = Xe[np.all(np.abs(Xe) < 1.2, axis=1)]
            Xe = Xe[_sd(Xe * scale + center) > gate]
            CV = Xe if (CV is None or not len(CV)) else np.concatenate([CV, Xe])
            EOn = np.zeros((0, 3)); ED = np.zeros((0, 3))
            print(f"[prior] empty-ray -> {len(Xe)} fixed samples (gated, chord path off)")

    # 2) build the SDF: grid_fuse (deterministic, no MLP) or an IGR MLP fit
    net = None
    if args.grid_fuse:
        assert _sd is not None, "--grid_fuse needs --prior_field or --prior_mesh"
        assert VB, "--grid_fuse needs view buffers (--prior_carve_views > 0 and a mask_dir)"
        verts, faces = grid_fuse_tsdf(VB, _sd, center, scale, args, debug_pts=prior_dbg)
        _lap("grid_fuse (TSDF integration + carve + marching cubes)")
    else:
        print("fitting the IGR SDF ...")
        net = train_sdf(Pn, N, On, EOn, ED, args, CV=CV, W=Wp, OBS=OBS, PV=PV, PS=PS)

    # 3) grid evaluation + marching cubes; tiling supports high-resolution whole-scene runs
    G = args.grid if args.grid > 0 else int(round(2 * scale / args.voxel_size))
    G = int(min(G, args.max_grid))
    from skimage.measure import marching_cubes
    if net is not None:
        net.eval()
    lin = np.linspace(-1, 1, G, dtype=np.float32)
    step = 2.0 / (G - 1)

    if net is None:
        pass                                    # grid_fuse path: verts/faces already built
    elif args.tile <= 0 or G <= args.tile:
        print(f"grid evaluation (G={G}, voxel~{2*scale/(G-1):.4f} world) + marching cubes ...")
        vol = np.empty((G, G, G), np.float32)
        with torch.no_grad():
            gx, gy = np.meshgrid(lin, lin, indexing="ij")
            for k in range(G):
                pts = np.stack([gx, gy, np.full_like(gx, lin[k])], -1).reshape(-1, 3)
                s = net(torch.tensor(pts, dtype=torch.float32, device="cuda")).cpu().numpy().reshape(G, G)
                vol[:, :, k] = s
        verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(step,) * 3)
        verts = (verts - 1.0) * scale + center
    else:
        # Tiled marching: split G into blocks, march each, then merge. Blocks overlap by one
        # voxel so there is no seam. Memory is O(tile^3).
        T = int(args.tile)
        nb = int(np.ceil((G - 1) / (T - 1)))
        print(f"tiled marching (G={G}, voxel~{2*scale/(G-1):.4f} world, "
              f"tile={T}, blocks={nb}^3={nb**3}) ...")
        vs_all, fs_all, voff = [], [], 0
        with torch.no_grad():
            for bi in range(nb):
                i0 = bi * (T - 1); i1 = min(i0 + T, G)
                for bj in range(nb):
                    j0 = bj * (T - 1); j1 = min(j0 + T, G)
                    for bk in range(nb):
                        k0 = bk * (T - 1); k1 = min(k0 + T, G)
                        xs, ys, zs = lin[i0:i1], lin[j0:j1], lin[k0:k1]
                        if min(len(xs), len(ys), len(zs)) < 2:
                            continue
                        sub = np.empty((len(xs), len(ys), len(zs)), np.float32)
                        gx, gy = np.meshgrid(xs, ys, indexing="ij")
                        for kk, zv in enumerate(zs):
                            pts = np.stack([gx, gy, np.full_like(gx, zv)], -1).reshape(-1, 3)
                            sub[:, :, kk] = net(torch.tensor(pts, dtype=torch.float32, device="cuda")
                                                ).cpu().numpy().reshape(len(xs), len(ys))
                        if sub.min() > 0 or sub.max() < 0:      # skip blocks with no zero crossing
                            continue
                        v, f, _, _ = marching_cubes(sub, level=0.0, spacing=(step,) * 3)
                        v = v + np.array([xs[0], ys[0], zs[0]]) + 1.0   # block origin -> [-1,1] coords
                        vs_all.append((v - 1.0) * scale + center)
                        fs_all.append(f + voff)
                        voff += len(v)
                print(f"  block row {bi+1}/{nb} done ({voff} verts so far)")
        assert vs_all, "no block contains a zero crossing -- check the fit and the scale"
        verts = np.concatenate(vs_all); faces = np.concatenate(fs_all)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    if args.mask_dist > 0:
        d, _ = tree.query(verts, workers=-1)
        far = d > args.mask_dist
        mesh.remove_vertices_by_mask(far)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        print(f"[trim] d>{args.mask_dist}: removed {int(far.sum())}/{len(verts)} vertices")

    # Colour: 1) match generated colour statistics to observed ones (removes the tone gap)
    #         2) distance-weighted blend across the seam (removes the hard cut)
    verts2 = np.asarray(mesh.vertices)
    Cw = np.clip(C, 0, 1).copy()
    n_obs = int(OBS.sum()); n_pri = int((~OBS).sum())
    if (not args.no_color_match) and n_obs > 100 and n_pri > 100:
        mo, so = Cw[OBS].mean(0), Cw[OBS].std(0) + 1e-6
        mp, sp = Cw[~OBS].mean(0), Cw[~OBS].std(0) + 1e-6
        Cw[~OBS] = np.clip((Cw[~OBS] - mp) / sp * so + mo, 0, 1)
        print(f"[colour] matched statistics: mean {np.round(mp,3)} -> {np.round(mo,3)}")
    if args.color_blend_ramp > 0 and n_obs > 100 and n_pri > 0:
        t_obs = cKDTree(P[OBS]); Cobs = Cw[OBS]
        d_o, i_o = t_obs.query(verts2, workers=-1)
        _, i_a = tree.query(verts2, workers=-1)
        w = np.clip(1.0 - d_o / args.color_blend_ramp, 0, 1)[:, None]
        col = w * Cobs[i_o] + (1 - w) * Cw[i_a]
        print(f"[colour] seam blend ramp {args.color_blend_ramp*1000:.0f}mm "
              f"({int(((w > 0) & (w < 1)).sum())}/{len(verts2)} vertices in transition)")
    else:
        _, ni = tree.query(verts2, workers=-1)
        col = Cw[ni]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
    mesh.compute_vertex_normals()

    # 5) safe_post_process_mesh(num_cluster): same logic as the TSDF path, plus the clamp

    out = os.path.expanduser(args.out)
    if not out:
        train_dir = os.path.join(args.model_path, "train", f"ours_{scene.loaded_iter}")
        os.makedirs(train_dir, exist_ok=True)
        out = os.path.join(train_dir, "sdf_fuse.ply")
    # Open3D picks the format from the extension: a missing or unsupported one only warns
    # ("unknown file extension") and fails silently. Fix it up and verify the write.
    if os.path.splitext(out)[1].lower() not in (".ply", ".obj", ".stl", ".off", ".gltf", ".glb"):
        print(f"[warn] missing or unsupported output extension ('{out}') -> appending '.ply'")
        out = out + ".ply"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(out, mesh)
    assert ok, f"[abort] failed to write mesh: {os.path.abspath(out)}"
    print(f"mesh saved at {os.path.abspath(out)}  verts {len(verts)} faces {len(faces)}")

    mesh_post = safe_post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    out_post = os.path.splitext(out)[0] + "_post.ply"
    ok = o3d.io.write_triangle_mesh(out_post, mesh_post)
    assert ok, f"[abort] failed to write post-processed mesh: {os.path.abspath(out_post)}"
    print(f"mesh post processed saved at {os.path.abspath(out_post)}")


if __name__ == "__main__":
    main()