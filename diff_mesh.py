#!/usr/bin/env python3
"""Bidirectional diff between the observed mesh and the fused mesh.

seen F1 can fall for three different reasons and the summary CSV cannot tell them
apart. This does, by measuring both directions against a tolerance:

  lost     A vertex with no B vertex within --tol   -> fusion DELETED surface
                                                       (carve / open-boundary /
                                                        largest-connected-component)
  new      B vertex with no A vertex within --tol   -> fusion INVENTED surface
  shifted  matched pairs, but the median distance is large
                                                       -> voxel quantisation

Measured context: with the prior fully disabled, seen F@1 still fell 0.821 -> 0.659,
so whatever this finds is in the grid machinery, not in ShapeR.

  python diff_mesh.py --out OUT/objects_voted --gids 0 5 7 6 --iter 30000
"""
import argparse
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def verts(path):
    m = o3d.io.read_triangle_mesh(path)
    return np.asarray(m.vertices), len(m.triangles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gids", nargs="*", default=[], help="default: every dir found")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--a", default="fuse_post.ply")
    ap.add_argument("--b", default="fused_field_post.ply")
    ap.add_argument("--tol", type=float, default=0.002, help="match radius (m)")
    args = ap.parse_args()

    od = os.path.expanduser(args.out)
    gids = args.gids or sorted(g for g in os.listdir(od) if g.isdigit())
    print(f"tol={args.tol * 1000:.0f}mm   A={args.a}  B={args.b}\n")
    print(f"{'gid':>5}{'|A|':>9}{'|B|':>9}{'dB/A':>8}"
          f"{'lost%':>8}{'new%':>7}{'shift_mm':>10}{'p95_mm':>8}  verdict")
    rows = []
    for gid in gids:
        d = os.path.join(od, str(gid), "train", f"ours_{args.iter}")
        pa, pb = os.path.join(d, args.a), os.path.join(d, args.b)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        A, fa = verts(pa)
        B, fb = verts(pb)
        if not len(A) or not len(B):
            continue
        dA = cKDTree(B).query(A, workers=-1)[0]      # A -> nearest B
        dB = cKDTree(A).query(B, workers=-1)[0]      # B -> nearest A
        lost = (dA > args.tol).mean() * 100
        new = (dB > args.tol).mean() * 100
        m = dB[dB <= args.tol]
        shift = float(np.median(dB)) * 1000
        p95 = float(np.percentile(dA, 95)) * 1000
        v = []
        if lost > 5:
            v.append("DELETED surface")
        if new > 5:
            v.append("INVENTED surface")
        if not v and shift > args.tol * 1000 * 0.5:
            v.append("quantised")
        print(f"{gid:>5}{len(A):>9,}{len(B):>9,}{len(B) / len(A):>8.2f}"
              f"{lost:>8.1f}{new:>7.1f}{shift:>10.2f}{p95:>8.1f}  {', '.join(v)}")
        rows.append((lost, new, shift))

    if rows:
        r = np.array(rows)
        print(f"\nmean  lost {r[:, 0].mean():.1f}%   new {r[:, 1].mean():.1f}%   "
              f"shift {r[:, 2].mean():.2f}mm")
        print("lost >> new  -> the carve / boundary logic is eating observed surface.")
        print("new  >> lost -> the grid is adding surface the observation never had.")
        print("both small   -> pure voxel quantisation; raise the grid resolution.")


if __name__ == "__main__":
    main()
