#!/usr/bin/env python3
"""SDF distillation 입력 점군 진단 덤프 — 학습 없이 back-project 점군만 ply로 저장.

sdf_distill_depth.py의 collect_oriented_points와 동일한 로직 + 픽셀 드랍 통계.
확인 목적:
  1) 벽/바닥이 점군에 존재하는가 (alpha_thr가 벽을 지우는지)
  2) 쿠션-소파 같은 접촉부가 점군 단계에서 이미 붙어있는가 (depth_ratio 0 vs 1 비교)
  3) 법선 방향 품질 (MeshLab에서 normal 시각화)

  # 현재 설정 그대로:
  python save_oriented_pcd.py -m output/replica_room0_v2/scene_whole_orbit -s data/replica_room0_v2 \
    --iteration 7000 --depth_ratio 0 --depth_trunc 6.0 --alpha_thr 0.5 --out pcd_dr0_a05.ply
  # depth_ratio 1 (median depth) 비교:
  python save_oriented_pcd.py -m ... -s ... --iteration 7000 --depth_ratio 1 --alpha_thr 0.5 --out pcd_dr1_a05.ply
  # alpha 필터 완화 비교(벽 회복 여부):
  python save_oriented_pcd.py -m ... -s ... --iteration 7000 --depth_ratio 1 --alpha_thr 0.1 --out pcd_dr1_a01.ply
"""
import numpy as np
import torch
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
import open3d as o3d


def cam_intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2],
                            [0, H / 2, 0, (H - 1) / 2],
                            [0, 0, 0, 1]]).float().cuda().T
    intrins = (cam.projection_matrix @ ndc2pix)[:3, :3].T
    fx, fy = intrins[0, 0].item(), intrins[1, 1].item()
    cx, cy = intrins[0, 2].item(), intrins[1, 2].item()
    extrinsic = cam.world_view_transform.T
    return fx, fy, cx, cy, W, H, extrinsic


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--depth_trunc", default=6.0, type=float)
    parser.add_argument("--alpha_thr", default=0.5, type=float)
    parser.add_argument("--pts_per_view", default=40000, type=int)
    parser.add_argument("--out", default="pcd_dump.ply", type=str)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.active_sh_degree = 0
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()
    P_all, N_all, C_all = [], [], []
    tot = dict(px=0, d0=0, dtrunc=0, alpha=0, kept=0)
    alpha_hist = torch.zeros(10)

    for cam in views:
        pkg = render(cam, gaussians, pipe, background)
        depth = pkg["surf_depth"][0]
        alpha = pkg["rend_alpha"][0]
        rgb = pkg["render"].permute(1, 2, 0)
        nrm = torch.nn.functional.normalize(pkg["rend_normal"], dim=0).permute(1, 2, 0)

        fx, fy, cx, cy, W, H, extrinsic = cam_intrinsics(cam)
        c2w = torch.inverse(extrinsic)
        cam_center = c2w[:3, 3]

        # 드랍 통계
        m_d0 = depth <= 0
        m_dt = depth >= args.depth_trunc
        m_a = alpha <= args.alpha_thr
        tot["px"] += depth.numel()
        tot["d0"] += int(m_d0.sum())
        tot["dtrunc"] += int((~m_d0 & m_dt).sum())
        tot["alpha"] += int((~m_d0 & ~m_dt & m_a).sum())
        alpha_hist += torch.histc(alpha.flatten().float(), bins=10, min=0, max=1).cpu()

        vv, uu = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32),
                                torch.arange(W, device="cuda", dtype=torch.float32),
                                indexing="ij")
        x = (uu - cx) * depth / fx
        y = (vv - cy) * depth / fy
        pts_w = torch.stack([x, y, depth], -1) @ c2w[:3, :3].T + cam_center

        valid = (~m_d0) & (~m_dt) & (~m_a)
        tot["kept"] += int(valid.sum())
        pts_w = pts_w[valid]; n = nrm[valid]; c = rgb[valid].clamp(0, 1)

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

    print("\n=== 픽셀 드랍 통계 (전체 뷰 합산) ===")
    print(f"총 픽셀        : {tot['px']}")
    print(f"depth<=0 드랍  : {tot['d0']} ({100*tot['d0']/tot['px']:.1f}%)")
    print(f"depth_trunc 드랍: {tot['dtrunc']} ({100*tot['dtrunc']/tot['px']:.1f}%)")
    print(f"alpha 드랍     : {tot['alpha']} ({100*tot['alpha']/tot['px']:.1f}%)  ← 벽 소실 의심 시 여기")
    print(f"유효(kept)     : {tot['kept']} ({100*tot['kept']/tot['px']:.1f}%)")
    print("alpha 분포(0~1, 10구간):", [int(v) for v in alpha_hist])
    print(f"저장 점 수(subsample 후): {len(P)}")
    bb = P.max(0) - P.min(0)
    print(f"bbox extent: {bb.round(3)}  center: {((P.max(0)+P.min(0))/2).round(3)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    pcd.colors = o3d.utility.Vector3dVector(C)
    o3d.io.write_point_cloud(args.out, pcd)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
