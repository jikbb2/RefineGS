#!/usr/bin/env python3
"""Novel poses that see the unobserved side, and the prior's depth from them.

This is the supervision signal for geometric fine-tuning. Raycasting the prior from the
TRAINING views is useless: the unobserved region is occluded there too, which is the same
reason the view-based TSDF never captured the injected gaussians. Poses that can see the
back are what makes the prior reachable by a rendering loss.

Each pixel gets three things:
  depth   the prior's surface distance along that ray
  valid   the ray hit the prior at all
  unseen  the 3D point at that depth is NOT observed by any training view
          -- the loss must be restricted to these, or prior and observation fight over the
          same surface and the observation loses

Consumed by the fine-tuning step as
    L_geom = | render(pose).depth - depth |   over (valid and unseen)

  python make_prior_depth.py --npz ~/prior_smoke/obj6_field.npz \
      --colmap DATA/sparse/0 --gt_depth_dir GTD --out /tmp/pd6 --n_poses 60
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warp_gt_to_pose import read_colmap                          # noqa: E402
from sdf_distill_depth import load_gt_depth                      # noqa: E402


def prior_mesh(npz, level=0.0):
    z = np.load(os.path.expanduser(npz))
    F = z["field"].astype(np.float32)
    assert F.min() < level < F.max(), "the prior field has no zero crossing"
    G = F.shape[0]
    v, f, _, _ = marching_cubes(F, level=level, spacing=(2.0 / (G - 1),) * 3)
    v = ((v - 1.0) / float(z["scale"])) @ z["R_align"] + z["center"]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                  o3d.utility.Vector3iVector(f))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    return m


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """world->camera R, t for a CV-convention camera (+Z forward)."""
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-9)
    if abs(float(f @ up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f); r /= max(np.linalg.norm(r), 1e-9)
    u = np.cross(f, r)
    R = np.stack([r, u, f])                       # rows: world->camera
    return R, -R @ eye


def orbit(center, radius, n, elev_deg=(-30, 0, 30, 60)):
    """Poses on a few elevation rings. Low rings matter: the underside of a table is the
    part no training view reached."""
    out = []
    per = max(1, n // len(elev_deg))
    for e in elev_deg:
        for i in range(per):
            a = 2 * np.pi * i / per + (0.5 * np.pi / per) * (e / 30.0)
            ce, se = np.cos(np.radians(e)), np.sin(np.radians(e))
            eye = center + radius * np.array([ce * np.cos(a), ce * np.sin(a), se])
            out.append(look_at(eye, center))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="prior field")
    ap.add_argument("--colmap", required=True, help="training poses, for intrinsics and "
                                                    "for the observed test")
    ap.add_argument("--gt_depth_dir", required=True)
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_poses", type=int, default=60)
    ap.add_argument("--radius_scale", type=float, default=2.2,
                    help="orbit radius as a multiple of the object's half-extent")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="|z - d_gt| tolerance for calling a point observed")
    ap.add_argument("--obs_views", type=int, default=150, help="training views to test against")
    ap.add_argument("--stems", default="")
    args = ap.parse_args()

    m = prior_mesh(args.npz)
    V = np.asarray(m.vertices)
    center = (V.min(0) + V.max(0)) / 2
    half = float(np.abs(V - center).max())
    radius = half * args.radius_scale
    print(f"[prior] {len(m.triangles):,} tris  centre {np.round(center,3)}  "
          f"half-extent {half:.3f}m  orbit radius {radius:.3f}m")

    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    cams = read_colmap(args.colmap)
    if args.stems and os.path.isfile(os.path.expanduser(args.stems)):
        keep = {l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()}
        cams = [c for c in cams if c["stem"] in keep]
    assert cams, "no training camera"
    c0 = cams[0]
    W, H = int(c0["W"]), int(c0["H"])
    K = np.array([[c0["fx"], 0, c0["cx"]], [0, c0["fy"], c0["cy"]], [0, 0, 1]])
    step = max(1, len(cams) // max(args.obs_views, 1))
    obs = cams[::step]
    print(f"[views] {W}x{H}  intrinsics from {c0['stem']}  "
          f"observed test against {len(obs)} training views")

    poses = orbit(center, radius, args.n_poses)
    D = np.zeros((len(poses), H, W), np.float32)
    U = np.zeros((len(poses), H, W), bool)
    n_hit = n_unseen = 0
    for i, (R, t) in enumerate(poses):
        E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
        r = sc.cast_rays(sc.create_rays_pinhole(
            o3d.core.Tensor(K), o3d.core.Tensor(E), W, H))
        d = r["t_hit"].numpy()
        hit = np.isfinite(d) & (d > 0)
        D[i] = np.where(hit, d, 0.0)
        n_hit += int(hit.sum())
        if not hit.any():
            continue
        # back-project the hits, then ask whether any training view already saw them
        vv, uu = np.nonzero(hit)
        z = d[vv, uu]
        x = (uu - K[0, 2]) / K[0, 0] * z
        y = (vv - K[1, 2]) / K[1, 1] * z
        P = (np.stack([x, y, z], 1) - t) @ R
        seen = np.zeros(len(P), bool)
        for c in obs:
            Dg = load_gt_depth(args.gt_depth_dir, c["stem"], c["H"], c["W"],
                               args.gt_depth_scale)
            if Dg is None:
                continue
            Hd, Wd = Dg.shape
            sx, sy = Wd / c["W"], Hd / c["H"]
            Xc = P @ c["R"].T + c["t"]
            zc = Xc[:, 2]
            zz = np.maximum(zc, 1e-6)
            ui = np.clip((c["fx"] * Xc[:, 0] / zz + c["cx"]) * sx, 0, Wd - 1).astype(int)
            vi = np.clip((c["fy"] * Xc[:, 1] / zz + c["cy"]) * sy, 0, Hd - 1).astype(int)
            ok = (zc > 0.05)
            dg = Dg[vi, ui]
            seen |= ok & (dg > 0.01) & (np.abs(zc - dg) < args.margin)
        u = np.zeros((H, W), bool)
        u[vv, uu] = ~seen
        U[i] = u
        n_unseen += int(u.sum())
        if (i + 1) % 20 == 0:
            print(f"  pose {i + 1}/{len(poses)}")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(
        out if out.endswith(".npz") else out + ".npz",
        R=np.stack([p[0] for p in poses]), t=np.stack([p[1] for p in poses]),
        K=K, W=W, H=H, depth=D, unseen=U, center=center, radius=radius)
    px = len(poses) * H * W
    print(f"\n[out] {out}   {len(poses)} poses")
    print(f"  prior hit   {n_hit / px * 100:5.1f}% of pixels")
    print(f"  supervised  {n_unseen / px * 100:5.1f}% (hit and not observed by training views)")
    if n_unseen < 0.02 * n_hit:
        print("  WARN almost nothing is unobserved -- either the orbit does not reach the "
              "hidden side, or this object was already fully seen")
