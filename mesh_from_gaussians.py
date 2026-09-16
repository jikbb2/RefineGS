#!/usr/bin/env python3
"""Mesh a gaussian model directly, with no camera anywhere.

render.py builds its TSDF from depth rendered at the training views, so geometry no
training view can see never enters the mesh. Measured: injecting 86,042 prior gaussians
into obj6 moved unseen completion by 3% (86.5 -> 83.8mm) while grid fusion moved it to
18.8mm -- the gaussians were present in the ply and absent from the mesh.

2DGS gaussians are oriented discs, so the surface can be extracted from them directly. A
round trip through that path costs about half a voxel (median 2.5mm, nothing beyond 10mm,
no invented surface), so the extraction itself is not the bottleneck.

This script exists to answer the remaining question: on OBSERVED geometry, where the
gaussians are well optimised, does direct extraction match the view-based TSDF? If it
does, views can be dropped entirely.

Round trip cost, SDF -> oriented points -> surface, over three objects:
    poisson   median 2.1-2.5mm,  beyond 10mm <= 0.4%,  invented 0.0%
    splat     median 2.5-2.9mm,  beyond 10mm  0.0%,    invented 2-43%
About half a voxel, and Poisson adds nothing that was not there.

  python mesh_from_gaussians.py -m OUT/objects_voted/6 --iteration 30000 \
      --out /tmp/g6.ply
"""
import argparse
import os

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


def disc_samples(P, T1, T2, N, S, k, seed=0):
    """Spread k points over each gaussian's disc.

    Using only the centres makes the input far sparser than what a view-based TSDF sees:
    9,815 centres for obj6 against hundreds of views of per-pixel depth. Poisson from that
    returned an accurate but incomplete surface (seen precision 0.98, recall 0.90; seen
    completion 2.88 -> 11.41mm). Each gaussian is an oriented disc, so its extent is known.
    """
    if k <= 1:
        return P, N
    rng = np.random.default_rng(seed)
    n = len(P)
    # tangents come from the rotation, not from an arbitrary basis built off the normal:
    # the disc is elliptical and its axes carry the two scales
    s1 = S[:, 0:1]
    s2 = S[:, 1:2] if S.shape[1] > 1 else S[:, 0:1]
    r = np.sqrt(rng.random((n, k, 1)))                 # uniform over the disc
    th = rng.random((n, k, 1)) * 2 * np.pi
    off = (r * np.cos(th)) * (s1 * T1)[:, None] + (r * np.sin(th)) * (s2 * T2)[:, None]
    Q = (P[:, None] + off).reshape(-1, 3)
    M = np.repeat(N, k, axis=0)
    print(f"[disc] {n:,} gaussians x {k} -> {len(Q):,} oriented points")
    return Q, M


def load_gaussians(path, min_opacity=0.1, max_scale=0.2):
    v = PlyData.read(os.path.expanduser(path))["vertex"]
    P = np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    w, x, y, z = q.T
    T1 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1)
    T2 = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1)
    N = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
    ns = sum(1 for n in v.data.dtype.names if n.startswith("scale_"))
    sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(ns)], 1).astype(np.float64))
    # A near-transparent or huge gaussian contributes nothing to the surface but drags the
    # nearest-neighbour field around, so drop both.
    keep = (op >= min_opacity) & (sc.max(1) <= max_scale)
    print(f"[gauss] {len(P):,} -> {int(keep.sum()):,} after opacity >= {min_opacity} "
          f"and scale <= {max_scale}m")
    return P[keep], T1[keep], T2[keep], N[keep], sc[keep]


def splat(P, N, voxel, trunc, band):
    """Signed distance to the nearest disc plane, on a grid. No camera involved."""
    lo, hi = P.min(0) - 4 * trunc, P.max(0) + 4 * trunc
    dim = np.maximum(((hi - lo) / voxel).astype(int) + 1, 2)
    print(f"[splat] grid {dim.tolist()} = {dim.prod() / 1e6:.1f}M voxels")
    assert dim.prod() < 4e8, "grid too large -- raise --voxel"
    tree = cKDTree(P)
    F = np.full(dim, trunc, np.float32)
    zs = np.arange(dim[2]) * voxel + lo[2]
    gx, gy = np.meshgrid(np.arange(dim[0]) * voxel + lo[0],
                         np.arange(dim[1]) * voxel + lo[1], indexing="ij")
    for k, zc in enumerate(zs):                       # slab by slab, bounded memory
        X = np.stack([gx, gy, np.full_like(gx, zc)], -1).reshape(-1, 3)
        d, j = tree.query(X, workers=-1, distance_upper_bound=band)
        m = np.isfinite(d)
        if not m.any():
            continue
        s = ((X[m] - P[j[m]]) * N[j[m]]).sum(1)       # signed by the disc normal
        sl = np.full(len(X), trunc, np.float32)
        sl[m] = np.clip(s, -trunc, trunc)
        F[:, :, k] = sl.reshape(dim[0], dim[1])
    assert F.min() < 0 < F.max(), "no zero crossing -- check normals or --band"
    v, f, _, _ = marching_cubes(F, level=0.0, spacing=(voxel,) * 3)
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v + lo),
                                     o3d.utility.Vector3iVector(f))


def poisson(P, N, depth, trim):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.normals = o3d.utility.Vector3dVector(N)
    m, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
    d = np.asarray(dens)
    if trim > 0 and len(d):
        m.remove_vertices_by_mask(d < np.quantile(d, trim))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="poisson", choices=["poisson", "splat"],
                    help="poisson solves a global indicator function and invents nothing "
                         "(measured 0.0%% extra surface on three objects). splat takes the "
                         "signed distance to the nearest disc, which creates spurious zero "
                         "crossings wherever facing surfaces are close -- between a chair "
                         "seat and its back, say: 43%% and 40%% invented surface on two "
                         "chairs against 2%% on a flat table")
    ap.add_argument("--voxel", type=float, default=0.004, help="match render.py's TSDF")
    ap.add_argument("--trunc", type=float, default=0.02)
    ap.add_argument("--band", type=float, default=0.03,
                    help="ignore voxels farther than this from any gaussian")
    ap.add_argument("--min_opacity", type=float, default=0.1)
    ap.add_argument("--max_scale", type=float, default=0.2)
    ap.add_argument("--disc_samples", type=int, default=20,
                    help="points sampled per gaussian disc; 1 uses the centre only")
    ap.add_argument("--poisson_depth", type=int, default=10)
    ap.add_argument("--poisson_trim", type=float, default=0.02,
                    help="drop vertices below this density quantile. Poisson closes holes, "
                         "which completes an object but can also invent surface where the "
                         "gaussians end; raise this if free violations climb")
    ap.add_argument("--num_cluster", type=int, default=1,
                    help="keep the N largest components, as render.py does")
    args = ap.parse_args()

    ply = os.path.join(os.path.expanduser(args.model), "point_cloud",
                       f"iteration_{args.iteration}", "point_cloud.ply")
    P, T1, T2, N, S = load_gaussians(ply, args.min_opacity, args.max_scale)
    P, N = disc_samples(P, T1, T2, N, S, args.disc_samples)
    m = (splat(P, N, args.voxel, args.trunc, args.band) if args.method == "splat"
         else poisson(P, N, args.poisson_depth, args.poisson_trim))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    if args.num_cluster > 0 and len(m.triangles):
        lab, cnt, _ = m.cluster_connected_triangles()
        lab, cnt = np.asarray(lab), np.asarray(cnt)
        keep = np.argsort(-cnt)[:args.num_cluster]
        m.remove_triangles_by_mask(~np.isin(lab, keep))
        m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    assert o3d.io.write_triangle_mesh(out, m), f"failed to write {out}"
    print(f"[out] {out}   {len(m.vertices):,} verts  {len(m.triangles):,} tris")


if __name__ == "__main__":
    main()
