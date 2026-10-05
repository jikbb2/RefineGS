"""Split an injected model into its observed and injected gaussians, as plain clouds.

A 3DGS ply opens in MeshLab as bare points, so the injected prior cannot be told apart
from the reconstruction it was added to. inject_prior_gaussians.py appends the new
gaussians after the originals and leaves the originals untouched, so a nearest-neighbour
test against the base model recovers the split exactly.

  python injected_only.py --base OUT/objects_voted/6 --inj OUT/objects_inj_0917/6 \
      --out /tmp/obj6
    -> /tmp/obj6_injected.ply   the prior surface that was added
       /tmp/obj6_observed.ply   what was already there
"""
import argparse, os
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree


def xyz(model, it):
    p = os.path.join(os.path.expanduser(model), "point_cloud", f"iteration_{it}",
                     "point_cloud.ply")
    v = PlyData.read(p)["vertex"]
    return np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64), p


def write(path, P, rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(np.tile(rgb, (len(P), 1)))
    assert o3d.io.write_point_cloud(path, pc), f"failed to write {path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="model before injection")
    ap.add_argument("--inj", required=True, help="model after injection")
    ap.add_argument("--iteration", default=30000, type=int)
    ap.add_argument("--out", required=True, help="prefix; _injected.ply and _observed.ply")
    args = ap.parse_args()

    A, pa = xyz(args.base, args.iteration)
    B, pb = xyz(args.inj, args.iteration)
    new = cKDTree(A).query(B)[0] > 1e-6

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    write(out + "_injected.ply", B[new], [1.0, 0.85, 0.0])   # yellow
    write(out + "_observed.ply", B[~new], [0.6, 0.6, 0.6])
    print(f"base {len(A):,}  injected {int(new.sum()):,}  "
          f"({new.mean()*100:.1f}% of the injected model)")
    print(f"  {out}_injected.ply   {out}_observed.ply")


if __name__ == "__main__":
    main()
