#!/usr/bin/env python3
"""Bidirectional surface diff between the observed mesh and the fused mesh.

Compare SURFACES, not vertices. A and B come from different voxel grids (render.py
uses 0.004, the fusion 0.005), so marching-cubes puts their vertices in different
places even where the surface is identical -- a vertex-to-vertex test at a few mm
reports ~100% mismatch for two meshes that agree perfectly. Sample both surfaces
uniformly instead.

Reported:
  d(B->A)   how far the fused surface sits from the observed one
  d(A->B)   observed surface with nothing fused near it = genuinely lost
  bias      mean signed offset along A's normal. Negative = B sits INSIDE A.
            A systematic inward bias is the signature of the alpha blend:
            F = alpha*Fobs + (1-alpha)*base with base = +trunc pulls the zero
            crossing toward the interior wherever alpha < 1 (alpha = Wo/grid_wcap).
  area      total triangle area, B/A. Erosion shows up here independent of sampling.

  python diff_mesh.py --out OUT/objects_voted
"""
import argparse
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def _seed(v=0):
    """open3d >= 0.16 seeds globally; older builds take no seed at all."""
    try:
        o3d.utility.random.seed(v)
    except AttributeError:
        pass


def load(path, n):
    m = o3d.io.read_triangle_mesh(path)
    if not len(m.triangles):
        return None, None, 0.0
    m.compute_vertex_normals()
    _seed(0)
    pc = m.sample_points_uniformly(number_of_points=n)
    return (np.asarray(pc.points), np.asarray(pc.normals),
            float(m.get_surface_area()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gids", nargs="*", default=[])
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--a", default="fuse_post.ply")
    ap.add_argument("--b", default="fused_field_post.ply")
    ap.add_argument("--n", type=int, default=200000, help="surface samples per mesh")
    ap.add_argument("--tols", type=float, nargs="*", default=[0.002, 0.005, 0.01, 0.02, 0.04],
                    help="report unmatched %% of the observed surface at each radius (m). "
                         "A smooth fall-off means the surface is DISPLACED; a plateau "
                         "means it is genuinely DELETED.")
    args = ap.parse_args()

    od = os.path.expanduser(args.out)
    gids = args.gids or sorted(g for g in os.listdir(od) if g.isdigit())
    tm = [f"{t * 1000:g}mm" for t in args.tols]
    print(f"{args.n:,} surface samples/mesh\n")
    print(f"{'gid':>5}{'dAB_med':>9}{'dAB_p95':>9}"
          + "".join(f"{t:>8}" for t in tm)
          + f"{'bias_mm':>9}{'areaB/A':>9}")
    rows = []
    for gid in gids:
        d = os.path.join(od, str(gid), "train", f"ours_{args.iter}")
        A, AN, aA = load(os.path.join(d, args.a), args.n)
        B, _, aB = load(os.path.join(d, args.b), args.n)
        if A is None or B is None:
            continue
        tA, tB = cKDTree(A), cKDTree(B)
        dBA, iBA = tA.query(B, workers=-1)          # fused -> observed
        dAB = tB.query(A, workers=-1)[0]            # observed -> fused
        # signed offset of the fused surface along the observed normal
        bias = float(np.median(((B - A[iBA]) * AN[iBA]).sum(1))) * 1000
        unm = [(dAB > t).mean() * 100 for t in args.tols]
        print(f"{gid:>5}{np.median(dAB) * 1000:>9.2f}{np.percentile(dAB, 95) * 1000:>9.2f}"
              + "".join(f"{u:>8.1f}" for u in unm)
              + f"{bias:>9.2f}{(aB / aA if aA else np.nan):>9.2f}")
        rows.append([np.median(dAB) * 1000] + unm + [bias, aB / aA if aA else np.nan])

    if rows:
        r = np.array(rows, float)
        act = r[r[:, 0] > 0]                         # drop passthrough identity rows
        m = np.median(act, axis=0)
        print(f"\nmedian over {len(act)} fused objects (of {len(r)}):")
        print(f"  d(A->B) {m[0]:.2f}mm   bias {m[-2]:+.2f}mm   areaB/A {m[-1]:.2f}")
        print("  unmatched observed surface: "
              + "  ".join(f"{t} {v:.1f}%" for t, v in zip(tm, m[1:1 + len(tm)])))
        print("\nfalls off smoothly -> the surface is DISPLACED, not deleted: look at "
              "voxel size, truncation and any smoothing of the observed field.")
        print("plateaus above ~10%%  -> that fraction is genuinely missing: look at "
              "carve, boundary removal and connected-component pruning.")


if __name__ == "__main__":
    main()