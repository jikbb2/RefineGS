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

import functools
import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData

print = functools.partial(print, flush=True)     # the heavy loops must show progress
from skimage.measure import marching_cubes

# Self-contained on purpose. Importing sdf_distill_depth pulls in torch, and torch loaded
# after open3d in the same process can segfault -- the script then exits with no output and
# no traceback, which is exactly what happened.


def _qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_colmap(path):
    """Minimal COLMAP text reader: [{stem,R,t,fx,fy,cx,cy,W,H}] sorted by name."""
    path = os.path.expanduser(path)
    cams = {}
    with open(os.path.join(path, "cameras.txt")) as f:
        for l in f:
            if l.startswith("#") or not l.strip():
                continue
            p = l.split()
            cid, model, W, H = int(p[0]), p[1], int(p[2]), int(p[3])
            v = [float(x) for x in p[4:]]
            if model in ("PINHOLE", "OPENCV"):
                fx, fy, cx, cy = v[0], v[1], v[2], v[3]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = v[0]; cx, cy = v[1], v[2]
            else:
                raise SystemExit(f"unsupported camera model {model}")
            cams[cid] = dict(fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H)
    out = []
    with open(os.path.join(path, "images.txt")) as f:
        lines = [l for l in f.read().split("\n")]
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    while i < len(lines):
        l = lines[i].strip()
        if not l:
            i += 1; continue
        p = l.split()
        q = [float(x) for x in p[1:5]]
        t = np.array([float(x) for x in p[5:8]])
        c = dict(cams[int(p[8])])
        c["stem"] = os.path.splitext(p[9])[0]
        c["R"] = _qvec2rot(q)                       # world -> camera
        c["t"] = t
        out.append(c)
        i += 2                                      # skip the POINTS2D line
    return sorted(out, key=lambda c: c["stem"])


def load_gt_depth(d, stem, H, W, scale):
    """depth in metres, or None. Naming: frameNNN -> depthNNN, same name, or _depth."""
    if not d:
        return None
    d = os.path.expanduser(d)
    for nm in (stem.replace("frame", "depth"), stem, stem + "_depth"):
        for ext in (".png", ".npy"):
            p = os.path.join(d, nm + ext)
            if not os.path.isfile(p):
                continue
            a = np.load(p) if ext == ".npy" else np.asarray(Image.open(p))
            if a.ndim == 3:
                a = a[..., 0]
            a = a.astype(np.float32) / (1.0 if ext == ".npy" else scale)
            if a.shape != (H, W):
                a = np.asarray(Image.fromarray(a).resize((W, H), Image.NEAREST))
            return a
    return None


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


def look_at(eye, target, up):
    """world->camera R, t for a CV-convention camera (+Z forward)."""
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-9)
    if abs(float(f @ up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(up, f); r /= max(np.linalg.norm(r), 1e-9)
    u = np.cross(f, r)
    R = np.stack([r, u, f])                       # rows: world->camera
    return R, -R @ eye


def orbit(center, radius, n, elev_deg=(-60, -45, -30, -15, 0, 30), up_axis=2,
          reach=None, obj_radius=0.0):
    """Poses on elevation rings, and which ring each belongs to.

    Low rings carry the supervision. Measured on a table, per ring: -30 gave up to 10.2%
    unobserved pixels while +60 gave 0.7-1.0%, because the top was already observed. Rings
    above the horizon are nearly wasted on furniture, so the default leans downward.
    """
    up = np.zeros(3); up[up_axis] = 1.0
    ax = [i for i in range(3) if i != up_axis]
    out, ring, blocked = [], [], 0
    per = max(1, n // len(elev_deg))
    for e in elev_deg:
        for i in range(per):
            a = 2 * np.pi * i / per + (0.5 * np.pi / per) * (e / 30.0)
            ce, se = np.cos(np.radians(e)), np.sin(np.radians(e))
            off = np.zeros(3)
            off[ax[0]], off[ax[1]], off[up_axis] = ce * np.cos(a), ce * np.sin(a), se
            eye = center + radius * off
            if reach is not None and not reachable(reach, eye, center, obj_radius):
                blocked += 1
                continue
            out.append(look_at(eye, center, up))
            ring.append(e)
    if reach is not None:
        print(f"[reach] {blocked} poses dropped as blocked, {len(out)} kept")
    return out, np.array(ring)


def load_mesh_any(path):
    """Open3D's PLY reader rejects Replica's mesh_semantic.ply: the per-face object_id
    property makes it fail with "unable to parse header", and it then returns an empty
    mesh that crashes RaycastingScene. Fall back to plyfile, which every other script here
    already uses for it. Quads are fan-triangulated."""
    path = os.path.expanduser(path)
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles):
        return m
    p = PlyData.read(path)
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)
    T = []
    for f in p["face"][p["face"].data.dtype.names[0]]:
        f = np.asarray(f)
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1]))
    assert T, f"no face in {path}"
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                     o3d.utility.Vector3iVector(np.asarray(T, np.int32)))


def reachable(rc, eye, center, obj_radius, slack=1.5):
    """False when something blocks the camera before it reaches the object.

    A pose inside a wall or a sofa renders nothing useful, and its prior depth would
    supervise the gaussians from a viewpoint that can never occur.
    """
    d = center - eye
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return False
    ray = o3d.core.Tensor([[*eye, *(d / dist)]], dtype=o3d.core.Dtype.Float32)
    t = float(rc.cast_rays(ray)["t_hit"].numpy()[0])
    return not (np.isfinite(t) and t < dist - obj_radius * slack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="prior field")
    ap.add_argument("--colmap", required=True, help="training poses, for intrinsics and "
                                                    "for the observed test")
    ap.add_argument("--gt_depth_dir", required=True)
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_poses", type=int, default=60)
    ap.add_argument("--up_axis", type=int, default=2,
                    help="world up: 2 = Z. On this data the +60 ring was almost fully "
                         "observed and the 0 ring covered the fewest pixels, which is what "
                         "a flat table looks like under Z-up")
    ap.add_argument("--occluder_mesh", default="",
                    help="scene mesh used to reject poses behind a wall or inside "
                         "furniture. The GT mesh works")
    ap.add_argument("--elev", default="-60,-45,-30,-15,0,30",
                    help="elevation rings in degrees. Negative looks up from below, which "
                         "is where an unobserved underside is")
    ap.add_argument("--radius_scale", type=float, default=1.8,
                    help="orbit radius as a multiple of the object's half-extent")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="|z - d_gt| tolerance for calling a point observed")
    ap.add_argument("--obs_views", type=int, default=80,
                    help="training views the observed test runs against")
    ap.add_argument("--obs_ds", type=int, default=2,
                    help="downscale the cached GT depth; a visibility test does not need "
                         "full resolution")
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

    # Cache the GT depth once. Reading it per pose meant n_poses x obs_views PNG loads
    # (60 x 150 = 9000) and the script looked hung.
    print("[cache] loading GT depth ...")
    OBS = []
    for c in obs:
        Dg = load_gt_depth(args.gt_depth_dir, c["stem"], c["H"], c["W"],
                           args.gt_depth_scale)
        if Dg is None:
            continue
        Dg = np.asarray(Dg, np.float32)
        if args.obs_ds > 1:
            Dg = Dg[::args.obs_ds, ::args.obs_ds]
        OBS.append((c, Dg))
    assert OBS, "no GT depth matched the training stems -- check --gt_depth_dir"
    print(f"[cache] {len(OBS)} depth maps, {sum(d.nbytes for _, d in OBS)/1e6:.0f} MB")

    reach = None
    if args.occluder_mesh and os.path.isfile(os.path.expanduser(args.occluder_mesh)):
        om = load_mesh_any(args.occluder_mesh)
        assert len(om.triangles), f"empty occluder mesh: {args.occluder_mesh}"
        reach = o3d.t.geometry.RaycastingScene()
        reach.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(om))
        print(f"[reach] occluder {len(om.triangles):,} tris")
    elev = tuple(float(x) for x in args.elev.split(","))
    poses, ring = orbit(center, radius, args.n_poses, elev, args.up_axis, reach, half)
    assert poses, "every pose was blocked -- check --occluder_mesh or raise --radius_scale"
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
        for c, Dg in OBS:
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
        print(f"  pose {i + 1}/{len(poses)}  hit {hit.mean()*100:4.1f}%  "
              f"unseen {u.mean()*100:4.1f}%")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(
        out if out.endswith(".npz") else out + ".npz",
        R=np.stack([p[0] for p in poses]), t=np.stack([p[1] for p in poses]),
        K=K, W=W, H=H, depth=D, unseen=U, center=center, radius=radius, ring=ring)
    px = len(poses) * H * W
    print(f"\n{'ring':>7}{'poses':>7}{'hit%':>8}{'unseen%':>9}")
    for e in sorted(set(ring.tolist())):
        m = ring == e
        print(f"{e:>7.0f}{int(m.sum()):>7}{(D[m] > 0).mean()*100:>8.1f}"
              f"{U[m].mean()*100:>9.1f}")
    print(f"\n[out] {out}   {len(poses)} poses")
    print(f"  prior hit   {n_hit / px * 100:5.1f}% of pixels")
    print(f"  supervised  {n_unseen / px * 100:5.1f}% (hit and not observed by training views)")
    if n_unseen < 0.02 * n_hit:
        print("  WARN almost nothing is unobserved -- either the orbit does not reach the "
              "hidden side, or this object was already fully seen")


if __name__ == "__main__":
    main()