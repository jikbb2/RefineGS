#!/usr/bin/env python3
"""Place the generated prior into the scene as gaussians, and re-mesh.

This is the cheapest test of "use ShapeR as material for the representation instead of as
a competing field": no optimiser, no pruning, no training loop. Points on the prior's zero
level set that the scene model does not already cover are appended to the ply as gaussians
oriented along the surface normal. render.py then meshes the result through the path we
already trust.

If unseen metrics improve here, the premise holds and fine-tuning is refinement. If they do
not, the idea is questionable and no amount of training loop will rescue it -- worth
knowing in an hour rather than a week.

Colour is copied from the nearest existing gaussian, which is wrong in general but keeps
the mesh readable; geometry is what this test measures.

  python inject_prior_gaussians.py -m OUT/scene --iteration 30000 \
      --fields ~/prior_v3/obj*_field.npz --out OUT/scene_inj
"""
import argparse
import glob
import os
import shutil

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def inv_sigmoid(x):
    return float(np.log(x / (1 - x)))


def quat_from_normal(N):
    """(M,3) unit normals -> (M,4) wxyz quaternions whose third axis is the normal."""
    n = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    t1 = np.cross(a, n)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9)
    t2 = np.cross(n, t1)
    R = np.stack([t1, t2, n], axis=2)                     # columns
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((len(R), 4))
    s = tr > 0
    if s.any():
        r = np.sqrt(1 + tr[s]) * 2
        q[s, 0] = 0.25 * r
        q[s, 1] = (R[s, 2, 1] - R[s, 1, 2]) / r
        q[s, 2] = (R[s, 0, 2] - R[s, 2, 0]) / r
        q[s, 3] = (R[s, 1, 0] - R[s, 0, 1]) / r
    for i, (a_, b_, c_) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        m = ~s & (R[:, a_, a_] >= R[:, b_, b_]) & (R[:, a_, a_] >= R[:, c_, c_])
        if not m.any():
            continue
        r = np.sqrt(1 + R[m, a_, a_] - R[m, b_, b_] - R[m, c_, c_]) * 2
        q[m, 0] = (R[m, c_, b_] - R[m, b_, c_]) / r
        q[m, 1 + a_] = 0.25 * r
        q[m, 1 + b_] = (R[m, b_, a_] + R[m, a_, b_]) / r
        q[m, 1 + c_] = (R[m, c_, a_] + R[m, a_, c_]) / r
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)


def prior_surface(npz, level=0.0):
    """World-space points and normals on the prior's zero level set."""
    z = np.load(os.path.expanduser(npz))
    F = z["field"].astype(np.float32)
    if not (F.min() < level < F.max()):
        return None, None
    G = F.shape[0]
    v, f, nrm, _ = marching_cubes(F, level=level, spacing=(2.0 / (G - 1),) * 3)
    # same transform as sdf_distill_depth's _sd, inverted
    v = ((v - 1.0) / float(z["scale"])) @ z["R_align"] + z["center"]
    nrm = nrm @ z["R_align"]
    # marching_cubes normals point along +gradient, i.e. outward for a signed field
    return v, -nrm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--fields", nargs="+", required=True, help="prior npz files or globs")
    ap.add_argument("--out", required=True, help="new model dir")
    ap.add_argument("--new_dist", type=float, default=0.02,
                    help="only inject where no gaussian is within this distance (m)")
    ap.add_argument("--max_new", type=int, default=40000, help="per prior field")
    ap.add_argument("--scale", type=float, default=0.006, help="disc radius (m)")
    ap.add_argument("--opacity", type=float, default=0.9)
    ap.add_argument("--level", type=float, default=0.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src = os.path.join(os.path.expanduser(args.model), "point_cloud",
                       f"iteration_{args.iteration}", "point_cloud.ply")
    ply = PlyData.read(src)
    el = ply["vertex"]
    V = np.stack([el[k] for k in ("x", "y", "z")], 1).astype(np.float64)
    names = list(el.data.dtype.names)
    tree = cKDTree(V)
    print(f"[scene] {len(V):,} gaussians   fields: ", end="")

    files = []
    for f in args.fields:
        files += sorted(glob.glob(os.path.expanduser(f)))
    print(len(files))
    if not files:
        raise SystemExit(
            "[abort] no prior field matched --fields.\n"
            "  Quote the pattern so the shell does not expand it: --fields '~/prior/obj*_field.npz'\n"
            "  NOTE the per-object and scene runs both write obj<gid>_field.npz, so one\n"
            "  overwrites the other unless PRIOR= points somewhere separate. Check the\n"
            "  timestamps to see which run produced the files you are about to use.")

    n_scale = sum(1 for n in names if n.startswith("scale_"))
    n_rest = sum(1 for n in names if n.startswith("f_rest_"))
    rng = np.random.default_rng(0)
    chunks, report = [], []
    for fp in files:
        P, N = prior_surface(fp, args.level)
        if P is None:
            report.append((os.path.basename(fp), 0, 0, "no zero crossing")); continue
        d, j = tree.query(P, workers=-1)
        keep = d > args.new_dist
        n_tot = len(P)
        P, N, j = P[keep], N[keep], j[keep]
        if len(P) > args.max_new:
            s = rng.choice(len(P), args.max_new, replace=False)
            P, N, j = P[s], N[s], j[s]
        report.append((os.path.basename(fp), n_tot, len(P), ""))
        if not len(P) or args.dry_run:
            continue

        rows = np.zeros(len(P), dtype=el.data.dtype)
        rows["x"], rows["y"], rows["z"] = P[:, 0], P[:, 1], P[:, 2]
        for k, c in zip(("nx", "ny", "nz"), range(3)):
            if k in names:
                rows[k] = N[:, c]
        for c in range(3):                                   # colour from the neighbour
            k = f"f_dc_{c}"
            if k in names:
                rows[k] = el[k][j]
        for c in range(n_rest):
            rows[f"f_rest_{c}"] = 0.0
        rows["opacity"] = inv_sigmoid(args.opacity)
        for c in range(n_scale):
            rows[f"scale_{c}"] = np.log(args.scale)
        q = quat_from_normal(N)
        for c in range(4):
            k = f"rot_{c}"
            if k in names:
                rows[k] = q[:, c]
        for k in names:                                      # id_*, desc_*: copy neighbour
            if k.startswith("id_") or k.startswith("desc_"):
                rows[k] = el[k][j]
        chunks.append(rows)

    print(f"\n{'field':<28}{'surface':>10}{'injected':>10}  note")
    for nm, a, b, note in report:
        print(f"{nm:<28}{a:>10,}{b:>10,}  {note}")
    n_new = sum(b for _, _, b, _ in report)   # not len(chunks): dry_run fills nothing
    print(f"\ninjected {n_new:,} gaussians "
          f"({n_new / max(len(V), 1) * 100:.1f}% of the scene)")
    if args.dry_run or not chunks:
        return

    out = os.path.expanduser(args.out)
    dst = os.path.join(out, "point_cloud", f"iteration_{args.iteration}")
    os.makedirs(dst, exist_ok=True)
    merged = np.concatenate([el.data] + chunks)
    PlyData([PlyElement.describe(merged, "vertex")]).write(
        os.path.join(dst, "point_cloud.ply"))
    for f in ("cfg_args", "cameras.json"):
        s = os.path.join(os.path.expanduser(args.model), f)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(out, f))
    print(f"-> {out}   ({len(merged):,} gaussians)")
    print("next: render.py -m this dir, then eval against the same GT")


if __name__ == "__main__":
    main()