#!/usr/bin/env python3
"""RefineGS — train + novel(orbit) 카메라 TSDF 융합 메쉬 추출 (미관측 gen 완성부까지 담음).

render.py 는 train 카메라 depth만 융합 → 미관측(gen 뒷면·가림)이 흰 구멍.
이 스크립트는 orbit novel pose(gen을 보는 각도)의 depth도 함께 융합 → 구멍을 gen geometry로 메움.
mesh_utils.py 는 안 건드리고, novel 카메라 객체(FusionCam)를 만들어 GaussianExtractor.reconstruction 에 함께 전달.

실행:
  python extract_mesh_novel.py -m output/replica_room0_v2/scene_whole_orbit -s data/replica_room0_v2 \
    --iteration 7000 --novel_poses ~/See3D/dataset/whole_orbit/poses/poses.npz --max_novel 300 \
    --depth_ratio 1 --depth_trunc 6.0 --voxel_size 0.006 --sdf_trunc 0.018 --num_cluster 10000 \
    --out fuse_novel.ply

Deps: render.py 와 동일 (Split&Splat 코드베이스).
"""
import os
import numpy as np
import torch
import open3d as o3d
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix
from utils.mesh_utils import GaussianExtractor, post_process_mesh


class FusionCam:
    """GaussianExtractor(reconstruction/to_cam_open3d/extract_mesh_bounded)가 요구하는 속성만 채운 카메라."""
    def __init__(self, W, H, FoVx, FoVy, wvt, fpt, znear=0.01, zfar=100.0):
        self.image_width = int(W); self.image_height = int(H)
        self.FoVx = float(FoVx); self.FoVy = float(FoVy)
        self.znear = znear; self.zfar = zfar
        self.world_view_transform = wvt
        self.full_proj_transform = fpt
        self.projection_matrix = getProjectionMatrix(znear, zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda()
        self.camera_center = wvt.inverse()[3, :3]
        self.alpha_mask = None            # mask_backgrond 가 None 이면 masking skip


def build_novel(poses_npz, max_novel):
    recs = list(np.load(poses_npz, allow_pickle=True)["records"])
    if max_novel > 0 and len(recs) > max_novel:
        idx = np.linspace(0, len(recs) - 1, max_novel).astype(int)
        recs = [recs[i] for i in idx]
    cams = []
    for r in recs:
        wvt = torch.tensor(np.asarray(r["world_view_transform"]), dtype=torch.float32).cuda()
        fpt = torch.tensor(np.asarray(r["full_proj_transform"]), dtype=torch.float32).cuda()
        cams.append(FusionCam(r["width"], r["height"], r["FoVx"], r["FoVy"], wvt, fpt))
    return cams


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--novel_poses", required=True, help="orbit_poses_objects poses.npz")
    parser.add_argument("--max_novel", default=300, type=int, help="융합할 novel 카메라 수(균등 subsample)")
    parser.add_argument("--voxel_size", default=0.006, type=float)
    parser.add_argument("--depth_trunc", default=6.0, type=float)
    parser.add_argument("--sdf_trunc", default=0.018, type=float)
    parser.add_argument("--num_cluster", default=10000, type=int)
    parser.add_argument("--out", default="fuse_novel.ply")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    ext = GaussianExtractor(gaussians, render, pipe, bg_color=bg)
    ext.gaussians.active_sh_degree = 0

    train_cams = scene.getTrainCameras()
    novel_cams = build_novel(args.novel_poses, args.max_novel)
    print(f"융합 카메라: train {len(train_cams)} + novel {len(novel_cams)} = {len(train_cams)+len(novel_cams)}")

    ext.reconstruction(list(train_cams) + novel_cams)
    mesh = ext.extract_mesh_bounded(voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc,
                                    depth_trunc=args.depth_trunc)

    train_dir = os.path.join(args.model_path, "train", f"ours_{scene.loaded_iter}")
    os.makedirs(train_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(train_dir, args.out), mesh)
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(train_dir, args.out.replace(".ply", "_post.ply")), mesh_post)
    print(f"→ {os.path.join(train_dir, args.out)} (+ _post)")
    print("train+novel 융합 → 미관측 gen 완성부까지 담긴 메쉬. 흰 구멍이 gen geometry로 메워졌는지 확인.")
