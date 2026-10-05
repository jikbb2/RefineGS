#!/usr/bin/env python3
"""모델 프리뷰 렌더 — 소수 뷰를 저용량 JPG로 (다운로드/디스크 부담 최소화).

  python render_preview.py -m output/replica_room0_v2/scene_b1_obj24_see3d_reg -s data/replica_room0_v2 \
    --iteration 3000 --train_stride 25 \
    --poses ~/See3D/dataset/obj24_v2/soft/poses.npz \
    --out ~/preview_see3d --quality 80

train_stride: 학습 카메라 N장마다 1장(전체 sanity). --poses: orbit 포즈(npz)도 렌더 —
See3D 정제된 unseen(테이블 하부)이 보이는 각도. 총 용량 ~2MB.
"""
import os
import numpy as np
import torch
from argparse import ArgumentParser
from PIL import Image

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args


def save_jpg(t, path, quality):
    a = (t.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(a).save(path, quality=quality)


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--train_stride", default=25, type=int, help="학습 카메라 N장당 1장(0=off)")
    parser.add_argument("--poses", default="", type=str, help="추가 novel 포즈 npz(render_hole_novel soft_out)")
    parser.add_argument("--quality", default=80, type=int)
    parser.add_argument("--out", required=True, type=str)
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    n = 0

    if args.train_stride > 0:
        for cam in scene.getTrainCameras()[::args.train_stride]:
            pkg = render(cam, gaussians, pipe, background)
            save_jpg(pkg["render"], os.path.join(out, f"train_{cam.image_name}.jpg"), args.quality)
            n += 1

    if args.poses:
        from scene.cameras import MiniCam
        recs = np.load(os.path.expanduser(args.poses), allow_pickle=True)["records"]
        for r in recs:
            i = int(r["idx"])
            wvt = torch.tensor(np.asarray(r["world_view_transform"]), dtype=torch.float32).cuda()
            fpt = torch.tensor(np.asarray(r["full_proj_transform"]), dtype=torch.float32).cuda()
            cam = MiniCam(int(r["width"]), int(r["height"]),
                          float(r["FoVy"]), float(r["FoVx"]), 0.01, 100.0, wvt, fpt)
            pkg = render(cam, gaussians, pipe, background)
            save_jpg(pkg["render"], os.path.join(out, f"novel_{i:04d}.jpg"), args.quality)
            n += 1

    print(f"→ {n}장 저장: {out}  (총 용량 확인: du -sh {out})")


if __name__ == "__main__":
    main()
