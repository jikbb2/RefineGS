#!/usr/bin/env python3
"""RefineGS — depth 렌더 기반 screened Poisson reconstruction (TSDF/SDF-MLP 대체).

sdf_distill_depth.py와 동일한 입력(뷰별 surf_depth back-project + 카메라 정렬 법선)을 쓰되,
IGR MLP 대신 Open3D screened Poisson으로 메쉬를 뽑는다.

장점:
  - watertight + 작은 구멍 자동 채움 (Poisson의 본래 목적)
  - MLP smoothness prior가 없어 객체 간 접촉면이 덜 뭉개짐
  - 볼륨 마스킹이 없어 큐브/계단 아티팩트가 구조적으로 없음
  - 학습 불필요 — 전체 수 분

환각 표면(관측 없는 영역에 Poisson이 만드는 막)은 2단계로 제거:
  1) density 트리밍: 지지 점이 적은 정점 제거 (--density_quantile)
  2) 거리 트리밍: 관측 점군에서 --trim_dist 이상 떨어진 정점 제거 (0=off)

RefineGS repo 루트(render.py 옆)에 두고 실행:

  python poisson_depth.py -m output/replica_room0_v2/scene_whole_orbit -s data/replica_room0_v2 \
    --iteration 7000 --depth_ratio 0 --depth_trunc 6.0 --num_cluster 10000 \
    --poisson_depth 11 --density_quantile 0.02 --trim_dist 0.06
"""
import os
import copy
import numpy as np
import torch
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
import open3d as o3d


# ---------------------------------------------------------------------------
# 카메라 intrinsic/extrinsic — to_cam_open3d(mesh_utils.py)와 완전히 동일한 규약
# ---------------------------------------------------------------------------
def cam_intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2],
                            [0, H / 2, 0, (H - 1) / 2],
                            [0, 0, 0, 1]]).float().cuda().T
    intrins = (cam.projection_matrix @ ndc2pix)[:3, :3].T
    fx, fy = intrins[0, 0].item(), intrins[1, 1].item()
    cx, cy = intrins[0, 2].item(), intrins[1, 2].item()
    extrinsic = cam.world_view_transform.T  # world->camera (w2c), CV 규약(+Z forward)
    return fx, fy, cx, cy, W, H, extrinsic


# ---------------------------------------------------------------------------
# utils.mesh_utils.post_process_mesh 안전 버전(클러스터 수 클램프 — IndexError 방지)
# ---------------------------------------------------------------------------
def safe_post_process_mesh(mesh, cluster_to_keep=1000):
    print(f"post processing the mesh to keep {cluster_to_keep} clusters (clamped)")
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    keep = min(cluster_to_keep, len(cluster_n_triangles))
    n_cluster = np.sort(cluster_n_triangles.copy())[-keep]
    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_oriented_points(scene, gaussians, pipe, background, args):
    """뷰별 depth를 월드 점군으로 back-project. 법선은 카메라 방향으로 정렬."""
    views = scene.getTrainCameras()
    P_all, N_all, C_all = [], [], []
    for cam in views:
        pkg = render(cam, gaussians, pipe, background)
        depth = pkg["surf_depth"][0]                      # [H,W]
        alpha = pkg["rend_alpha"][0]                      # [H,W]
        rgb = pkg["render"].permute(1, 2, 0)              # [H,W,3]
        nrm = torch.nn.functional.normalize(pkg["rend_normal"], dim=0).permute(1, 2, 0)  # world

        fx, fy, cx, cy, W, H, extrinsic = cam_intrinsics(cam)
        c2w = torch.inverse(extrinsic)
        cam_center = c2w[:3, 3]

        vv, uu = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32),
                                torch.arange(W, device="cuda", dtype=torch.float32),
                                indexing="ij")
        x = (uu - cx) * depth / fx
        y = (vv - cy) * depth / fy
        pts_cam = torch.stack([x, y, depth], -1)
        pts_w = pts_cam @ c2w[:3, :3].T + cam_center

        valid = (depth > 0) & (depth < args.depth_trunc) & (alpha > args.alpha_thr)
        pts_w = pts_w[valid]
        n = nrm[valid]
        c = rgb[valid].clamp(0, 1)

        # 법선을 카메라 쪽으로 정렬(부호 일관) — Poisson은 일관된 orientation 필수
        view_dir = cam_center[None] - pts_w
        flip = (n * view_dir).sum(-1) < 0
        n[flip] = -n[flip]
        n = torch.nn.functional.normalize(n, dim=-1)

        if len(pts_w) > args.pts_per_view:
            sel = torch.randperm(len(pts_w), device="cuda")[:args.pts_per_view]
            pts_w, n, c = pts_w[sel], n[sel], c[sel]

        P_all.append(pts_w.cpu()); N_all.append(n.cpu()); C_all.append(c.cpu())

    P = torch.cat(P_all).numpy().astype(np.float64)
    N = torch.cat(N_all).numpy().astype(np.float64)
    C = torch.cat(C_all).numpy().astype(np.float64)
    return P, N, C


def main():
    parser = ArgumentParser(description="Depth-based screened Poisson reconstruction")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # render.py TSDF 옵션과 동일한 것들
    parser.add_argument("--depth_trunc", default=6.0, type=float, help="최대 depth (back-project cutoff)")
    parser.add_argument("--num_cluster", default=10000, type=int, help="후처리 유지 클러스터 수(클램프)")
    # 점군 수집
    parser.add_argument("--alpha_thr", default=0.5, type=float, help="이하 alpha 픽셀 제거(배경/floater)")
    parser.add_argument("--pts_per_view", default=60000, type=int)
    parser.add_argument("--n_pts", default=4000000, type=int, help="Poisson 입력 점 상한(subsample)")
    parser.add_argument("--outlier_nb", default=20, type=int, help="statistical outlier removal 이웃 수(0=off)")
    parser.add_argument("--outlier_std", default=2.0, type=float)
    # Poisson
    parser.add_argument("--poisson_depth", default=11, type=int,
                        help="octree depth. 10≈거칠, 11≈2-3cm 디테일, 12≈고해상(느림/메모리↑)")
    parser.add_argument("--density_quantile", default=0.02, type=float,
                        help="이 분위수 미만 density 정점 제거 — Poisson 환각 막 제거. 0=off")
    parser.add_argument("--trim_dist", default=0.06, type=float,
                        help="관측 점군에서 이 거리(world) 초과 정점 제거. 구멍 다시 생기면 키우고, 막 남으면 줄임. 0=off")
    parser.add_argument("--out", default="", type=str)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.active_sh_degree = 0  # diffuse만 — render.py mesh 경로와 동일
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    # 1) oriented point cloud
    print("뷰별 depth back-project + 법선 정렬 ...")
    P, N, C = collect_oriented_points(scene, gaussians, pipe, background, args)
    print(f"표면점 {len(P)}")
    if len(P) > args.n_pts:
        idx = np.random.choice(len(P), args.n_pts, replace=False)
        P, N, C = P[idx], N[idx], C[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    pcd.colors = o3d.utility.Vector3dVector(C)

    # 2) outlier 제거(떠다니는 depth 노이즈 → Poisson 혹 방지)
    if args.outlier_nb > 0:
        pcd, keep_idx = pcd.remove_statistical_outlier(nb_neighbors=args.outlier_nb,
                                                       std_ratio=args.outlier_std)
        print(f"outlier 제거 후 {len(pcd.points)}")

    # 3) screened Poisson
    print(f"screened Poisson (depth={args.poisson_depth}) ...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth)
    densities = np.asarray(densities)
    print(f"raw mesh: verts {len(mesh.vertices)} faces {len(mesh.triangles)}")

    # 4a) density 트리밍 — 지지 점이 적은(환각) 정점 제거
    if args.density_quantile > 0:
        thr = np.quantile(densities, args.density_quantile)
        mesh.remove_vertices_by_mask(densities < thr)
        print(f"density 트리밍(q={args.density_quantile}) 후: verts {len(mesh.vertices)}")

    # 4b) 거리 트리밍 — 관측 점군에서 먼 정점 제거(메쉬 단계 마스킹: 큐브 아티팩트 없음)
    if args.trim_dist > 0:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.asarray(pcd.points)).query(np.asarray(mesh.vertices), workers=-1)
        mesh.remove_vertices_by_mask(d > args.trim_dist)
        print(f"거리 트리밍(d={args.trim_dist}) 후: verts {len(mesh.vertices)}")

    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    out = args.out
    if not out:
        train_dir = os.path.join(args.model_path, "train", f"ours_{scene.loaded_iter}")
        os.makedirs(train_dir, exist_ok=True)
        out = os.path.join(train_dir, "poisson_fuse.ply")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"mesh saved at {out}  verts {len(mesh.vertices)} faces {len(mesh.triangles)}")

    # 5) 클러스터 후처리(TSDF 경로와 동일 로직, 클램프)
    mesh_post = safe_post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    out_post = out.replace(".ply", "_post.ply")
    o3d.io.write_triangle_mesh(out_post, mesh_post)
    print(f"mesh post processed saved at {out_post}")


if __name__ == "__main__":
    main()
