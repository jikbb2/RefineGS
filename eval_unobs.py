#!/usr/bin/env python3
"""미관측 영역 한정 completion 평가 — refinement 의 담당 구역 성적표.

GT 표면점 중 '어떤 학습 카메라에서도 보이지 않는' 점(가림/뒷면)을 raycast 가시성으로 골라,
그 부분집합에 대한 comp(GT→recon 거리)를 모델별 비교. whole-scene 평균에 희석되는
국소 개선을 분리 측정한다.

  python eval_unobs.py --gt_mesh ~/room_0/habitat/mesh_semantic.ply \
    --colmap data/replica_room0_v2/sparse/0 --n_cams 100 \
    --recon output/replica_room0_v2/scene_whole_dense_see3d_a7/train/ours_5000/fuse_post.ply \
            output/replica_room0_v2/scene_whole_dense_gtwarp_a7/train/ours_5000/fuse_post.ply \
            output/replica_room0_v2/scene_whole_dense_reg/train/ours_15000/fuse_post.ply \
            output/replica_room0_v2/scene_mono_reg/train/ours_30000/fuse_post.ply \
    --names see3d_a7 gtwarp_a7 stage5 mono

RefineGS repo 루트에서 실행. Deps: numpy, open3d, plyfile, scipy.
"""
import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from warp_gt_to_pose import read_colmap, cam_center


def load_gt_whole(gt_path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--n_cams", type=int, default=100, help="가시성 검사에 쓸 카메라 수(균등 샘플)")
    ap.add_argument("--n_pts", type=int, default=300000)
    ap.add_argument("--vis_eps", type=float, default=0.02, help="가시 판정 여유(m)")
    ap.add_argument("--recon", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()

    names = args.names if args.names and len(args.names) == len(args.recon) \
        else [os.path.basename(p) for p in args.recon]

    gt = load_gt_whole(args.gt_mesh)
    pc = gt.sample_points_uniformly(args.n_pts)
    P = np.asarray(pc.points).astype(np.float32)

    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))

    cams = read_colmap(args.colmap)
    cams = cams[:: max(len(cams) // args.n_cams, 1)][: args.n_cams]
    centers = [cam_center(c["R"], c["t"]).astype(np.float32) for c in cams]
    print(f"GT 점 {len(P)}, 카메라 {len(centers)}대로 가시성 검사 ...")

    visible = np.zeros(len(P), bool)
    for ci, cc in enumerate(centers):
        todo = ~visible
        if not todo.any():
            break
        Q = P[todo]
        d = Q - cc[None]
        dist = np.linalg.norm(d, axis=1)
        dirs = d / dist[:, None]
        rays = np.concatenate([np.broadcast_to(cc, Q.shape), dirs], 1).astype(np.float32)
        thit = rc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis = thit >= dist - args.vis_eps           # 카메라→점 사이 가림 없음
        idx = np.where(todo)[0]
        visible[idx[vis]] = True
        if ci % 20 == 0:
            print(f"  cam {ci}: 가시 {visible.mean()*100:.1f}%")
    unobs = ~visible
    print(f"미관측 GT 점: {unobs.sum()} ({unobs.mean()*100:.1f}%)  /  관측 {visible.sum()}")

    print(f"\n{'model':>14} | {'comp_all':>9} {'comp_obs':>9} {'comp_unobs':>10} (mm, GT→recon)")
    print("-" * 60)
    for name, path in zip(names, args.recon):
        m = o3d.io.read_triangle_mesh(os.path.expanduser(path))
        rp = np.asarray(m.sample_points_uniformly(args.n_pts).points)
        tr = cKDTree(rp)
        d_all, _ = tr.query(P, workers=-1)
        print(f"{name:>14} | {d_all.mean()*1000:9.1f} {d_all[visible].mean()*1000:9.1f} "
              f"{d_all[unobs].mean()*1000:10.1f}")

    print("\ncomp_unobs = refinement 담당 구역 성적. 여기서도 stage5 ≥ see3d 면 "
          "'dense 체제에서 생성 refinement 무익' 결론 확정, see3d 우위면 '국소 개선이 전역 노이즈에 희석' 서사.")


if __name__ == "__main__":
    main()
