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


def load(path, n):
    m = o3d.io.read_triangle_mesh(path)
    if not len(m.triangles):
        return None, None, 0.0
    m.compute_vertex_normals()
    pc = m.sample_points_uniformly(number_of_points=n, seed=0)
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
    ap.add_argument("--lost_tol", type=float, default=0.01,
                    help="A sample with no B surface within this is lost (m)")
    args = ap.parse_args()

    od = os.path.expanduser(args.out)
    gids = args.gids or sorted(g for g in os.listdir(od) if g.isdigit())
    print(f"{args.n:,} surface samples/mesh   lost_tol={args.lost_tol * 1000:.0f}mm\n")
    print(f"{'gid':>5}{'dBA_med':>9}{'dBA_p95':>9}{'dAB_med':>9}"
          f"{'lost%':>7}{'bias_mm':>9}{'areaB/A':>9}  note")
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
        lost = (dAB > args.lost_tol).mean() * 100
        note = []
        if abs(bias) > 2:
            note.append("INSIDE" if bias < 0 else "OUTSIDE")
        if lost > 10:
            note.append("lost surface")
        if aA > 0 and aB / aA < 0.85:
            note.append("eroded")
        print(f"{gid:>5}{np.median(dBA) * 1000:>9.2f}{np.percentile(dBA, 95) * 1000:>9.2f}"
              f"{np.median(dAB) * 1000:>9.2f}{lost:>7.1f}{bias:>9.2f}"
              f"{(aB / aA if aA else np.nan):>9.2f}  {', '.join(note)}")
        rows.append((np.median(dBA) * 1000, lost, bias, aB / aA if aA else np.nan))

    if rows:
        r = np.array(rows, float)
        print(f"\nmedian over {len(r)} objects:  d(B->A) {np.median(r[:, 0]):.2f}mm"
              f"   lost {np.median(r[:, 1]):.1f}%"
              f"   bias {np.median(r[:, 2]):+.2f}mm"
              f"   areaB/A {np.median(r[:, 3]):.2f}")
        print("bias clearly negative -> the alpha blend is pulling the isosurface in; "
              "raise grid_wcap or lift alpha where Wo > 0.")
        print("bias ~0 but lost high -> carve / boundary removal is deleting surface.")


if __name__ == "__main__":
    main()