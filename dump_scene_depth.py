#!/usr/bin/env python3
"""전체 씬 모델의 뷰별 depth를 npz로 덤프 — per-object SDF의 free-space carving 증거용.

scene_mono_reg 같은 '정규화 잘 된 whole-scene 모델'의 depth를 200뷰 전부 저장하면,
sdf_distill_depth.py --carve_depth_dir 로 읽어 "카메라→표면 사이 = 빈 공간" 제약을
객체 bbox 안에 적용할 수 있다 (8개 객체 뷰로는 못 보는 테이블 옆/아래 공간 carve).

  python dump_scene_depth.py -m output/replica_room0_v2/scene_mono_reg -s data/replica_room0_v2 \
    --iteration 30000 --depth_ratio 1 --out_dir ~/carve_depth_mono
"""
import os
import numpy as np
import torch
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args


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
    parser.add_argument("--alpha_thr", default=0.5, type=float, help="이하 alpha 픽셀 depth=0(무효)")
    parser.add_argument("--out_dir", required=True, type=str)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    views = scene.getTrainCameras()
    for i, cam in enumerate(views):
        pkg = render(cam, gaussians, pipe, background)
        depth = pkg["surf_depth"][0]
        alpha = pkg["rend_alpha"][0]
        depth = torch.where(alpha > args.alpha_thr, depth, torch.zeros_like(depth))
        fx, fy, cx, cy, W, H, extrinsic = cam_intrinsics(cam)
        c2w = torch.inverse(extrinsic).cpu().numpy().astype(np.float32)
        np.savez_compressed(
            os.path.join(out_dir, f"{os.path.splitext(cam.image_name)[0]}.npz"),
            depth=depth.cpu().numpy().astype(np.float16),
            fx=np.float32(fx), fy=np.float32(fy), cx=np.float32(cx), cy=np.float32(cy),
            c2w=c2w)
        if i % 50 == 0:
            print(f"{i}/{len(views)} ...")
    print(f"done: {len(views)}뷰 → {out_dir}")


if __name__ == "__main__":
    main()
