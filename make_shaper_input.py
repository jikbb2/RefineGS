#!/usr/bin/env python3
"""RefineGS object -> ShapeR input pkl.

ShapeR (Meta FAIR, arXiv 2601.11514) conditions on posed multi-view images plus a METRIC
sparse point cloud, which anchors the generated shape to the observed geometry, outputs an
SDF directly (no sign-fix / shell_delta / flood-fill), and keeps the seam attached.

Schema from ShapeR `dataset/shaper_dataset.py` and `dataset/image_processor.py`.
Required keys (SLAM path, strategy="cluster"):
  points_model            (N,3) torch  metric points in the object frame
  bounds                  (3,)  torch  half-extent -> scale = 0.9/max(bounds)
  inv_dist_std, dist_std  (N,)  torch  per-point uncertainty (smaller = trusted more)
  image_data              list[bytes]  encoded images (opened by PIL, converted to "L")
  Ts_camera_model         list[(4,4) torch]  model->camera
  camera_params           list[(3,3)]  pinhole K. Upstream expects Fisheye624, so
                                        infer_shape_pinhole.py patches this
  object_point_projections list[(M,2) torch]  uv of the object points, in crop coords
  visible_points_model    list[(M,3)]  visible points per view (drives view selection)
  T_model_world           (4,4) torch  world -> model (for --do_transform_to_world)
  caption / category      str
Optional:
  mesh_vertices, mesh_faces  GT, for evaluation only

  python make_shaper_input.py --gid 1 \
    --recon output/replica_room0_v2/refinegs_full/1/train/ours_7000/fuse_post.ply \
    --colmap data/replica_room0_v2/sparse/0 \
    --images data/replica_room0_v2/images \
    --masks_root data/replica_room0_v2/masks \
    --stems ~/See3D/dataset/stage6/clean_stems/1.txt \
    --caption "a sofa" --out ~/ShapeR/data/refinegs_obj1.pkl
"""
import os
import io
import glob
import pickle
import argparse

import numpy as np
import open3d as o3d
import torch
from PIL import Image

try:
    from warp_gt_to_pose import read_colmap, cam_center
except Exception:                                     # running outside the repo
    read_colmap = None

    def cam_center(R, t):
        return -R.T @ t


def load_mask(masks_root, gid, stem):
    p = os.path.join(masks_root, str(gid), "masks", stem + ".png")
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p))
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., 3]
    elif a.ndim == 3:
        a = np.array(Image.open(p).convert("L"))
    if a.max() <= 1:
        return a > 0
    if (a == 188).any():
        return a == 188                               # amodal convention: 188 = visible
    return a > 127


def load_depth_map(depth_dir, stem, scale):
    """stem -> depth in metres. Naming: frameNNNN->depthNNNN, same name, or _depth."""
    for c in (stem.replace("frame", "depth"), stem, stem + "_depth"):
        for ext in (".png", ".npy"):
            p = os.path.join(os.path.expanduser(depth_dir), c + ext)
            if not os.path.exists(p):
                continue
            if ext == ".npy":
                return np.load(p).astype(np.float32)
            return np.array(Image.open(p)).astype(np.float32) / scale
    return None


def _count_seen(P_w, stems, cams, args, margin):
    """Per point: in how many views is it the first surface inside the object mask."""
    n_seen = np.zeros(len(P_w), np.int32)
    n_dep = n_msk = 0
    for s in stems:
        D = load_depth_map(args.depth_dir, s, args.depth_scale)
        if D is None:
            continue
        n_dep += 1
        # The mask is needed too: a depth match alone also passes points lying on the
        # floor or on a neighbouring object, and those corrupt the generation.
        M = load_mask(args.masks_root, args.gid, s) if args.masks_root else None
        if M is not None:
            n_msk += 1
        c = cams[s]
        Hd, Wd = D.shape
        sx, sy = Wd / c["W"], Hd / c["H"]
        Xc = P_w @ c["R"].T + c["t"]
        z = Xc[:, 2]
        zz = np.maximum(z, 1e-6)
        u = (c["fx"] * Xc[:, 0] / zz + c["cx"]) * sx
        v = (c["fy"] * Xc[:, 1] / zz + c["cy"]) * sy
        ok = (z > 0.05) & (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd)
        if not ok.any():
            continue
        ui = np.clip(u, 0, Wd - 1).astype(int); vi = np.clip(v, 0, Hd - 1).astype(int)
        d = D[vi, ui]
        hit = ok & (d > 0.01) & (np.abs(z - d) < margin)
        if M is not None:                                  # must be inside the mask
            if M.shape != (Hd, Wd):
                M = np.array(Image.fromarray(M.astype(np.uint8))
                             .resize((Wd, Hd), Image.NEAREST)) > 0
            hit &= M[vi, ui]
        n_seen += hit.astype(np.int32)
    return n_seen, n_dep, n_msk


def filter_observed(P_w, stems, cams, args):
    """Keep points that are the first surface inside the mask in >= seen_min_views views.

    ShapeR anchors to these points, so recon junk is reproduced in the generation. The old
    version fell back to the UNFILTERED cloud below 200 survivors, which silently disabled
    the filter on the worst objects (34-object batch: 7 kept 0 points, 2 kept under 5%).
    Relax the margin instead, and refuse an object rather than poison it.
    """
    tried = []
    for mv in (args.seen_min_views, 1):
        for mul in (1.0, 2.0, 4.0):
            margin = args.seen_margin * mul
            n_seen, n_dep, n_msk = _count_seen(P_w, stems, cams, args, margin)
            keep = n_seen >= mv
            tried.append((margin, mv, int(keep.sum())))
            if keep.sum() >= args.min_kept:
                print(f"[filter] gid {args.gid}: {int(keep.sum())}/{len(P_w)} verified "
                      f"({keep.mean()*100:.1f}%)  depth {n_dep} / mask {n_msk} views"
                      + ("" if len(tried) == 1 else
                         f"  RELAXED |z-d|<{margin*1000:.0f}mm min_views={mv}"))
                return P_w[keep]
    print(f"[filter] FAILED to verify any observation for gid {args.gid}")
    for margin, mv, n in tried:
        print(f"    |z-d|<{margin*1000:.0f}mm  min_views={mv}  ->  {n} points")
    if args.allow_unfiltered:
        print("  --allow_unfiltered: passing the RAW cloud (it contains the junk)")
        return P_w
    raise SystemExit(
        f"[abort] gid {args.gid}: no reconstructed point lands on GT depth inside the mask.\n"
        f"  The reconstruction and the GT are not in the same place, or the mask belongs to\n"
        f"  a different object. Feeding this to ShapeR produces a prior built from junk.\n"
        f"  Inspect the recon against the GT object, or drop this gid. "
        f"Override with --allow_unfiltered.")


def sample_free_points(stems, cams, center, R_align, bounds, args, n_target=6000):
    """Observed free-space samples in the object frame.

    Space between a camera and the observed surface is known empty. Carrying it in the pkl
    blocks hallucination under a table earlier than the fusion-stage carve could remove it.
    """
    rng = np.random.default_rng(0)
    per = max(64, n_target // max(1, len(stems)))
    out = []
    for s in stems:
        D = load_depth_map(args.depth_dir, s, args.depth_scale)
        if D is None:
            continue
        c = cams[s]
        Hd, Wd = D.shape
        sx, sy = Wd / c["W"], Hd / c["H"]
        C = -c["R"].T @ c["t"]
        vs, us = np.nonzero(D > 0.05)
        if len(vs) == 0:
            continue
        k = min(per * 3, len(vs))
        sel = rng.choice(len(vs), k, replace=False)
        v_, u_ = vs[sel], us[sel]
        d = D[v_, u_]
        x = ((u_ / sx) - c["cx"]) / c["fx"]
        y = ((v_ / sy) - c["cy"]) / c["fy"]
        dirs = np.stack([x, y, np.ones_like(x)], 1) @ c["R"]      # world dirs, z = 1
        tau = rng.uniform(0.25, 0.95, k) * np.maximum(d - 2 * args.seen_margin, 1e-3)
        P = C[None] + dirs * tau[:, None]
        Pm = (R_align @ (P - center).T).T
        keep = (np.abs(Pm) <= bounds).all(1)                      # inside the object bbox
        if keep.any():
            out.append(Pm[keep])
    if not out:
        print("[free] no samples -- check the depth path")
        return np.zeros((0, 3), np.float32)
    F = np.concatenate(out)
    if len(F) > n_target:
        F = F[rng.choice(len(F), n_target, replace=False)]
    print(f"[free] {len(F)} free-space samples in the object bbox")
    return F.astype(np.float32)


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="RefineGS object -> ShapeR input pkl")
    ap.add_argument("--gid", required=True)
    ap.add_argument("--recon", required=True, help="observed object mesh (fuse_post.ply)")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks_root", default="")
    ap.add_argument("--stems", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption", default="", help="empty falls back to 'a 3D object'")
    ap.add_argument("--n_points", type=int, default=20000,
                    help="conditioning point count. Domain-gap A/B: dense 20000 vs SLAM-like 1500")
    ap.add_argument("--n_views", type=int, default=32,
                    help="candidate views stored in the pkl; ShapeR picks 16 of them")
    ap.add_argument("--world_up", default="z", choices=["x", "y", "z"])
    ap.add_argument("--bounds_margin", type=float, default=1.15,
                    help="half-extent margin. At 1.0 the unobserved extension can clip at |p|>1")
    ap.add_argument("--point_std", type=float, default=1e-3,
                    help="point uncertainty (inv_dist_std/dist_std). Our depth is better "
                         "than the semi-dense SLAM this expects, so keep it small or the "
                         "upstream filter discards the points")
    ap.add_argument("--img_max_side", type=int, default=640,
                    help="downscale images to this long side; ShapeR infers at 280px")
    ap.add_argument("--seed", type=int, default=0,
                    help="point sampling seed. Fix it or the same command yields a "
                         "different pkl, which invalidates any A/B comparison")
    ap.add_argument("--gt_mesh", default="", help="optional GT mesh, evaluation only")
    ap.add_argument("--depth_dir", default="",
                    help="depth folder for the observation filter. With it, only verified "
                         "points condition the generation, so junk in the unobserved part "
                         "of the recon cannot corrupt it")
    ap.add_argument("--depth_scale", type=float, default=6553.5)
    ap.add_argument("--min_kept", type=int, default=200,
                    help="minimum verified points before the margin is relaxed")
    ap.add_argument("--allow_unfiltered", action="store_true",
                    help="pass the raw cloud when nothing verifies, instead of aborting. "
                         "The old behaviour, and it poisoned the prior silently")
    ap.add_argument("--n_pool", type=int, default=200000,
                    help="points sampled from the mesh BEFORE the observation filter. Fixed "
                         "so --n_points does not move the object frame")
    ap.add_argument("--frame_pct", type=float, default=0.5,
                    help="percentile used for the object bbox (0 = raw min/max). The frame "
                         "sets ShapeR's normalisation, so one floater costs real resolution")
    ap.add_argument("--seen_margin", type=float, default=0.02,
                    help="tolerance (m) for calling a point observed: |z - depth| < margin")
    ap.add_argument("--seen_min_views", type=int, default=2,
                    help="views that must confirm a point")
    ap.add_argument("--free_points", type=int, default=6000,
                    help="observed free-space samples to carry in the pkl (0 = off); "
                         "used by shaper_field.py --guide_free_w to constrain generation")
    args = ap.parse_args()

    assert read_colmap is not None, "cannot import warp_gt_to_pose -- run from the RefineGS root"

    # ---- 1) observed points (world, metric) ----
    m = o3d.io.read_triangle_mesh(os.path.expanduser(args.recon))
    assert len(m.vertices), f"failed to load recon: {args.recon}"
    # Unseeded sampling changes the conditioning points, and ShapeR follows them closely.
    # This, not ShapeR's own sampling, was the main source of run-to-run variation.
    _seeded = False
    try:
        o3d.utility.random.seed(args.seed)             # Open3D >= 0.16
        _seeded = True
    except Exception:
        pass
    # Sample a fixed larger pool, filter, then subsample to --n_points: sampling n_points
    # directly spends the budget on junk the filter deletes. The pool is fixed so changing
    # --n_points cannot move the object frame (which would break ensemble averaging).
    n_pool = max(args.n_pool, args.n_points)
    try:
        pc = m.sample_points_uniformly(n_pool, seed=args.seed)
        _seeded = True
    except TypeError:
        pc = m.sample_points_uniformly(n_pool)
    P_w = np.asarray(pc.points, np.float64)
    if not _seeded:
        print("[points] WARN this Open3D ignores the sampling seed; runs will differ")

    # ---- 2) object frame: gravity aligned, AABB centre ----
    R_align = np.eye(3)
    if args.world_up != "z":                          # ShapeR assumes a z-up object frame
        ax = {"x": 0, "y": 1}[args.world_up]
        perm = [0, 1, 2]; perm[ax], perm[2] = perm[2], perm[ax]
        R_align = np.eye(3)[perm]

    # ---- 3) cameras / views ----
    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    if args.stems and os.path.exists(os.path.expanduser(args.stems)):
        stems = [l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()]
    else:
        stems = sorted(cams)
    stems = [s for s in stems if s in cams and find_image(args.images, s)]
    assert stems, "no usable view -- check --images / --stems"

    # ---- 3b) observation filter, THEN the frame ----
    # The frame was previously taken from raw mesh vertices: one floater doubled `bounds`,
    # halved `scale`, and the object shrank inside ShapeR's cube, costing resolution on
    # exactly the worst reconstructions (measured raw/robust 2.73, 1.69, 1.62, 1.52).
    # Use the filtered points and a percentile bbox.
    F_m = np.zeros((0, 3), np.float32)
    if args.depth_dir:
        P_w = filter_observed(P_w, stems, cams, args)

    q = args.frame_pct
    lo, hi = np.percentile(P_w, q, axis=0), np.percentile(P_w, 100 - q, axis=0)
    center = (lo + hi) / 2
    keep = np.all((P_w >= lo) & (P_w <= hi), axis=1)
    ref = P_w[keep] if keep.sum() >= 200 else P_w
    bounds = np.abs((R_align @ (ref - center).T).T).max(0) * args.bounds_margin
    scale = 0.9 / bounds.max()
    T_model_world = np.eye(4)                          # world -> model
    T_model_world[:3, :3] = R_align
    T_model_world[:3, 3] = -R_align @ center

    V_w = np.asarray(m.vertices, np.float64)
    raw_b = np.abs((R_align @ (V_w - center).T).T).max(0) * args.bounds_margin
    if args.depth_dir and args.free_points > 0:
        F_m = sample_free_points(stems, cams, center, R_align, bounds, args,
                                 n_target=args.free_points)
    if len(P_w) > args.n_points:                       # budget now lands on real surface
        rng = np.random.default_rng(args.seed)
        P_w = P_w[rng.choice(len(P_w), args.n_points, replace=False)]
    P_m = (R_align @ (P_w - center).T).T
    clipped = int((np.abs(P_m * scale) > 1.0).any(1).sum())
    ratio = raw_b.max() / bounds.max()
    print(f"[frame] gid {args.gid}: {len(P_w)} pts  "
          f"half-extent={np.round(bounds, 3)}m  scale={scale:.3f}  clipped={clipped}  "
          f"raw/robust={ratio:.2f}x"
          + ("  <- junk was setting the frame" if ratio > 1.3 else ""))
    if len(stems) > args.n_views:                      # uniform subsample
        idx = np.unique(np.linspace(0, len(stems) - 1, args.n_views).round().astype(int))
        stems = [stems[i] for i in idx]

    image_data, Ts_cm, cam_params, obj_uv, vis_pts = [], [], [], [], []
    n_infr, n_vis = [], []
    n_mask = 0
    for s in stems:
        c = cams[s]
        img = Image.open(find_image(args.images, s)).convert("L")
        W0, H0 = img.size
        sc = min(1.0, args.img_max_side / max(W0, H0))
        if sc < 1.0:
            img = img.resize((int(round(W0 * sc)), int(round(H0 * sc))), Image.LANCZOS)
        W, H = img.size
        fx, fy = c["fx"] * sc, c["fy"] * sc
        cx, cy = c["cx"] * sc, c["cy"] * sc

        # model -> camera : T_cm = T_cw @ T_wm
        T_cw = np.eye(4); T_cw[:3, :3] = c["R"]; T_cw[:3, 3] = c["t"]
        T_cm = T_cw @ np.linalg.inv(T_model_world)

        # project the object points (pinhole): crop coords plus the view-selection score
        Xc = P_m @ T_cm[:3, :3].T + T_cm[:3, 3]
        z = Xc[:, 2]
        ok = z > 1e-6
        u = np.full(len(P_m), -1.0); v = np.full(len(P_m), -1.0)
        u[ok] = fx * Xc[ok, 0] / z[ok] + cx
        v[ok] = fy * Xc[ok, 1] / z[ok] + cy
        infr = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if args.masks_root:                            # refine visibility with the mask
            mk = load_mask(args.masks_root, args.gid, s)
            if mk is not None:
                n_mask += 1
                if mk.shape != (H, W):
                    mk = np.array(Image.fromarray(mk.astype(np.uint8))
                                  .resize((W, H), Image.NEAREST)) > 0
                ui = np.clip(u, 0, W - 1).astype(int); vi = np.clip(v, 0, H - 1).astype(int)
                infr &= mk[vi, ui]
        # Occlusion test: infr only means "projects inside the image". Without it, points
        # on the far side counted as visible (obj10: 13133/14252 = 92% across all 32
        # views), telling ShapeR there is nothing to complete. Keep first-surface only.
        vis = infr.copy()
        if args.depth_dir:
            d = load_depth_map(args.depth_dir, s, args.depth_scale)
            if d is not None:
                if d.shape != (H, W):
                    d = np.array(Image.fromarray(d).resize((W, H), Image.NEAREST))
                ui = np.clip(u, 0, W - 1).astype(int); vi = np.clip(v, 0, H - 1).astype(int)
                dv = d[vi, ui]
                vis = infr & (dv > 0.01) & (np.abs(z - dv) < args.seen_margin)
        if infr.sum() < 20:                            # drop views that barely see it
            continue

        buf = io.BytesIO(); img.save(buf, format="PNG")
        image_data.append(buf.getvalue())
        Ts_cm.append(torch.tensor(T_cm, dtype=torch.float32))
        cam_params.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32))
        obj_uv.append(torch.tensor(np.stack([u[infr], v[infr]], 1), dtype=torch.float32))
        vis_pts.append(P_m[vis].astype(np.float32))
        n_infr.append(int(infr.sum())); n_vis.append(int(vis.sum()))
    assert image_data, "no valid view -- check the masks and poses"
    _mv, _mi = int(np.median(n_vis)), int(np.median(n_infr))
    vis = _mv / max(_mi, 1)
    print(f"[views] {len(image_data)} ({n_mask} masked)  visible {_mv}/{_mi} "
          f"({vis*100:.0f}%)  occlusion {'on' if args.depth_dir else 'OFF'}"
          + ("  <- too high; nothing left for ShapeR to complete" if vis > 0.8 else ""))

    # ---- 4) assemble the pkl ----
    N = len(P_m)
    sample = {
        "points_model": torch.tensor(P_m, dtype=torch.float32),
        "bounds": torch.tensor(bounds, dtype=torch.float32),
        "inv_dist_std": torch.full((N,), args.point_std, dtype=torch.float32),
        "dist_std": torch.full((N,), args.point_std, dtype=torch.float32),
        "image_data": image_data,
        "Ts_camera_model": torch.stack(Ts_cm),
        "camera_params": np.stack(cam_params),         # 3x3 K (needs the pinhole patch)
        "object_point_projections": obj_uv,
        "visible_points_model": vis_pts,
        "T_model_world": torch.tensor(T_model_world, dtype=torch.float32),
        "caption": args.caption or "a 3D object",
        "is_ariagen2": False,
        "pinhole": True,                               # the patch skips rectification on this
    }
    if len(F_m):                                       # free-space constraint for generation
        sample["free_points_model"] = torch.tensor(F_m, dtype=torch.float32)
    if args.gt_mesh:
        gm = o3d.io.read_triangle_mesh(os.path.expanduser(args.gt_mesh))
        if len(gm.vertices):
            gv = (R_align @ (np.asarray(gm.vertices) - center).T).T
            sample["mesh_vertices"] = torch.tensor(gv, dtype=torch.float32)
            sample["mesh_faces"] = torch.tensor(np.asarray(gm.triangles), dtype=torch.int64)

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(sample, fh)
    print(f"[pkl] {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()