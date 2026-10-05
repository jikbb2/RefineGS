#!/usr/bin/env python3
"""How much geometry is lost going SDF -> gaussians -> surface, without any rendering?

The injected point cloud looks right, but the view-based TSDF cannot see it, so the
question is whether extracting a surface directly from the gaussians preserves the prior.
That sets the ceiling for treating the prior as material for the representation: the
grid-fusion path injects the decoder's SDF straight into the voxel grid and never makes
this round trip.

Two sources:
  --npz    the prior field itself, sampled as an ideal oriented point cloud
  --ply    the gaussians actually injected, normals taken from their rotations
           (this includes the losses the injection already introduced)

Two extractions: screened Poisson, and splatting the discs into a voxel grid.

  python roundtrip_test.py --npz ~/prior_smoke/obj6_field.npz \
      --ply OUT/objects_inj/6/point_cloud/iteration_30000/point_cloud.ply
"""
import argparse
import os

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def prior_mesh(npz, level=0.0):
    z = np.load(os.path.expanduser(npz))
    F = z["field"].astype(np.float32)
    G = F.shape[0]
    v, f, _, _ = marching_cubes(F, level=level, spacing=(2.0 / (G - 1),) * 3)
    v = ((v - 1.0) / float(z["scale"])) @ z["R_align"] + z["center"]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                  o3d.utility.Vector3iVector(f))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    m.compute_vertex_normals()
    return m


def gauss_points(ply):
    """xyz and the disc normal (third column of the rotation) from a 3DGS/2DGS ply."""
    v = PlyData.read(os.path.expanduser(ply))["vertex"]
    P = np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    w, x, y, z = q.T
    N = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    return P, N


def sample_oriented(mesh, n, seed=0):
    try:
        o3d.utility.random.seed(seed)
    except AttributeError:
        pass
    pc = mesh.sample_points_uniformly(n)
    return np.asarray(pc.points), np.asarray(pc.normals)


def poisson(P, N, depth=9, trim=0.02):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.normals = o3d.utility.Vector3dVector(N)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    d = np.asarray(dens)
    if trim > 0 and len(d):
        m.remove_vertices_by_mask(d < np.quantile(d, trim))
    return m


def splat(P, N, voxel=0.005, trunc=0.02):
    """Rasterise the oriented discs into a grid as a signed distance, then march.

    No camera anywhere: each voxel takes the signed distance to the nearest disc plane.
    """
    lo, hi = P.min(0) - 4 * trunc, P.max(0) + 4 * trunc
    dim = np.maximum(((hi - lo) / voxel).astype(int) + 1, 2)
    if dim.prod() > 3e8:
        return None
    g = np.stack(np.meshgrid(*[np.arange(d) for d in dim], indexing="ij"), -1)
    X = lo + g.reshape(-1, 3) * voxel
    tree = cKDTree(P)
    d, j = tree.query(X, workers=-1)
    s = ((X - P[j]) * N[j]).sum(1)                    # signed by the disc normal
    F = np.where(d < 3 * trunc, np.clip(s, -trunc, trunc), trunc).reshape(dim)
    if not (F.min() < 0 < F.max()):
        return None
    v, f, _, _ = marching_cubes(F, level=0.0, spacing=(voxel,) * 3)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v + lo),
                                  o3d.utility.Vector3iVector(f))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    return m


def compare(ref, got, n=200000, tol=(0.002, 0.005, 0.01)):
    if got is None or not len(got.triangles):
        return None
    try:
        o3d.utility.random.seed(0)
    except AttributeError:
        pass
    A = np.asarray(ref.sample_points_uniformly(n).points)
    B = np.asarray(got.sample_points_uniformly(n).points)
    dAB = cKDTree(B).query(A, workers=-1)[0]
    dBA = cKDTree(A).query(B, workers=-1)[0]
    return dict(med=np.median(dAB) * 1000,
                unmatched=[(dAB > t).mean() * 100 for t in tol],
                extra=[(dBA > t).mean() * 100 for t in tol],
                tris=len(got.triangles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ply", default="", help="injected gaussians (optional)")
    ap.add_argument("--n_sample", type=int, default=120000)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--poisson_depth", type=int, default=9)
    args = ap.parse_args()

    ref = prior_mesh(args.npz)
    print(f"[ref] prior mesh: {len(ref.triangles):,} tris  "
          f"extent {np.round(ref.get_max_bound() - ref.get_min_bound(), 3)} m")

    srcs = [("prior mesh sampled", *sample_oriented(ref, args.n_sample))]
    if args.ply:
        P, N = gauss_points(args.ply)
        srcs.append((f"injected gaussians ({len(P):,})", P, N))

    print(f"\n{'source':<28}{'method':<10}{'tris':>10}{'med':>8}"
          f"{'>2mm':>8}{'>5mm':>8}{'>10mm':>8}{'extra>10mm':>12}")
    for nm, P, N in srcs:
        for meth, fn in (("poisson", lambda: poisson(P, N, args.poisson_depth)),
                         ("splat", lambda: splat(P, N, args.voxel))):
            r = compare(ref, fn())
            if r is None:
                print(f"{nm:<28}{meth:<10}{'FAILED':>10}")
                continue
            print(f"{nm:<28}{meth:<10}{r['tris']:>10,}{r['med']:>8.2f}"
                  + "".join(f"{u:>8.1f}" for u in r["unmatched"])
                  + f"{r['extra'][2]:>12.1f}")

    print("\nmed / >Nmm: how much of the PRIOR surface the round trip fails to reproduce.")
    print("extra>10mm: surface the round trip invented that the prior does not have.")
    print("Grid fusion makes no round trip -- it injects the decoder SDF into the grid,")
    print("so these numbers are the ceiling for the gaussian-based path.")


if __name__ == "__main__":
    main()
