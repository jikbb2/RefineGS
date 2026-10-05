#!/usr/bin/env python3
"""Whole-scene 메쉬 vs GT(mesh_semantic.ply 전체) 평가 — 여러 recon 나란히 비교.

  python eval_whole_scene.py --gt_mesh ~/room_0/habitat/mesh_semantic.ply \
    --recon output/replica_room0_v2/scene_whole_dense_reg/train/ours_15000/fuse_post.ply \
            output/replica_room0_v2/scene_mono_reg/train/ours_30000/fuse_post.ply \
    --names ours mono

Deps: numpy, open3d, plyfile, scipy.
"""
import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree


def load_gt_whole(gt_path):
    """semantic ply 전체 → o3d mesh (quad 자동 삼각화)."""
    ply = PlyData.read(os.path.expanduser(gt_path))
    v = ply["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    f = ply["face"]
    fname = [n for n in f.data.dtype.names if "vertex" in n][0]
    faces = np.vstack([np.asarray(x) for x in f[fname]])
    if faces.shape[1] == 4:
        faces = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(np.ascontiguousarray(faces.astype(np.int32))))
    m.remove_unreferenced_vertices()
    return m


def sample(mesh, n):
    pc = mesh.sample_points_uniformly(n, use_triangle_normal=True)
    return np.asarray(pc.points), np.asarray(pc.normals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--recon", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--n_pts", type=int, default=500000)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.01, 0.02, 0.05])
    args = ap.parse_args()

    names = args.names if args.names and len(args.names) == len(args.recon) \
        else [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in args.recon]

    gt = load_gt_whole(args.gt_mesh)
    gp, gn = sample(gt, args.n_pts)
    tg = cKDTree(gp)
    print(f"GT 샘플 {len(gp)}점\n")

    header = f"{'model':>24} | {'acc(mm)':>8} {'comp(mm)':>8} {'chamfer':>8} {'NC':>6}"
    for t in args.taus:
        header += f" {'F@'+str(int(t*1000))+'mm':>8}"
    print(header)
    print("-" * len(header))

    for name, path in zip(names, args.recon):
        m = o3d.io.read_triangle_mesh(os.path.expanduser(path))
        assert len(m.triangles) > 0, f"빈 메쉬: {path}"
        rp, rn = sample(m, args.n_pts)
        tr = cKDTree(rp)
        d_rg, i_rg = tg.query(rp, workers=-1)
        d_gr, _ = tr.query(gp, workers=-1)
        acc, comp = d_rg.mean(), d_gr.mean()
        nc = np.abs((rn * gn[i_rg]).sum(-1)).mean()
        row = f"{name:>24} | {acc*1000:8.1f} {comp*1000:8.1f} {(acc+comp)/2*1000:8.1f} {nc:6.3f}"
        for t in args.taus:
            p, r = (d_rg < t).mean(), (d_gr < t).mean()
            f1 = 2 * p * r / max(p + r, 1e-9)
            row += f" {f1:8.3f}"
        print(row)

    print("\nacc=recon→GT 평균거리(낮을수록 정확), comp=GT→recon(낮을수록 완전), F=조화평균")


if __name__ == "__main__":
    main()
