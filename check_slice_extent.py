#!/usr/bin/env python3
"""Compare a sliced gaussian set's extent against the mesh render.py built from it.

The fusion grid takes center/scale from the gaussian point cloud. Voting assigns a
label per gaussian, so a handful of strays far from the object still count: they
inflate the bbox, the grid covers dead space, and the surface gets fewer voxels
than render.py's TSDF gave it. That shows up as seen accuracy degrading even with
the prior fully disabled (measured: seen F@1 0.821 -> 0.659 with prior off, i.e.
the prior accounts for 0.006 of the 0.169 loss -- the grid accounts for the rest).

robust = 1st..99th percentile extent. bbox/robust >> 1 means outliers dominate.
mesh   = bbox of fuse_post.ply, i.e. what the object actually occupies.

  python check_slice_extent.py --out OUT/objects_voted --iter 30000
"""
import argparse
import os

import numpy as np
from plyfile import PlyData


def xyz_of(path):
    p = PlyData.read(path)["vertex"]
    return np.stack([p[k] for k in ("x", "y", "z")], 1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir holding per-object model dirs")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--lo", type=float, default=1.0, help="robust percentile")
    args = ap.parse_args()

    od = os.path.expanduser(args.out)
    rows = []
    for gid in sorted(os.listdir(od)):
        if not gid.isdigit():
            continue
        ply = os.path.join(od, gid, "point_cloud", f"iteration_{args.iter}",
                           "point_cloud.ply")
        msh = os.path.join(od, gid, "train", f"ours_{args.iter}", "fuse_post.ply")
        if not os.path.isfile(ply):
            continue
        g = xyz_of(ply)
        full = float((g.max(0) - g.min(0)).max())
        q = np.percentile(g, [args.lo, 100 - args.lo], axis=0)
        rob = float((q[1] - q[0]).max())
        mesh = np.nan
        if os.path.isfile(msh):
            m = xyz_of(msh)
            if len(m):
                mesh = float((m.max(0) - m.min(0)).max())
        rows.append((gid, len(g), full, rob, mesh))

    if not rows:
        raise SystemExit(f"no per-object dirs with iteration_{args.iter} under {od}")

    rows.sort(key=lambda r: -(r[2] / max(r[3], 1e-9)))
    print(f"{'gid':>5}{'gauss':>9}{'bbox':>8}{'robust':>8}{'mesh':>8}"
          f"{'bbox/rob':>10}{'bbox/mesh':>11}  note")
    for gid, n, full, rob, mesh in rows:
        r1 = full / max(rob, 1e-9)
        r2 = full / mesh if mesh == mesh else float("nan")
        note = []
        if r1 > 1.5:
            note.append("outliers inflate bbox")
        if r2 == r2 and r2 > 1.5:
            note.append("grid much larger than object")
        print(f"{gid:>5}{n:>9,}{full:>8.2f}{rob:>8.2f}"
              f"{mesh:>8.2f}{r1:>10.2f}{r2:>11.2f}  {', '.join(note)}")

    a = np.array([[r[2], r[3], r[4]] for r in rows], float)
    ok = np.isfinite(a[:, 2])
    print(f"\nobjects {len(rows)}   median bbox/robust {np.median(a[:, 0] / a[:, 1]):.2f}"
          + (f"   median bbox/mesh {np.median(a[ok, 0] / a[ok, 2]):.2f}" if ok.any() else ""))
    print("ratio near 1.0 = the grid fits the object. Above ~1.5 the fusion grid is "
          "spending its resolution on empty space.")


if __name__ == "__main__":
    main()
