#!/usr/bin/env python3
"""RefineGS — 학습 뷰에서 RGB | depth | normal 렌더 저장 (메쉬 찢김 디버깅).

메쉬는 depth 로 만드므로, RGB가 깨끗해도 *depth가 노이즈면 메쉬가 찢긴다*.
학습 카메라 몇 개에서 render_pkg 의 render/depth/rend_normal 을 나란히 저장해 눈으로 확인.

깨끗한 표면이면: depth 가 부드러운 그라데이션, normal 이 면별로 일관.
찢긴 geometry면: depth 에 얼룩·계단·구멍, normal 이 지글지글.  → distortion/densify 부족 신호.

실행:
  python render_depth_check.py -m output/replica_room0_v2/scene_whole_orbit -s data/replica_room0_v2 \
    --iteration 7000 --n 4 --out_dir output/replica_room0_v2/scene_whole_orbit/depth_check

Deps: torch, torchvision, numpy, matplotlib(선택).
"""
import os
import numpy as np
import torch
import torchvision
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state


def colorize(x):
    x = x.detach().cpu().numpy()
    lo, hi = np.percentile(x[x > 0], 2) if (x > 0).any() else 0.0, np.percentile(x, 98)
    x = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    try:
        import matplotlib.cm as cm
        return torch.from_numpy(cm.turbo(x)[..., :3]).permute(2, 0, 1).float()
    except Exception:
        return torch.from_numpy(x)[None].repeat(3, 1, 1).float()


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--n", default=4, type=int, help="확인할 학습 뷰 수")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    cams = scene.getTrainCameras()
    idx = np.linspace(0, len(cams) - 1, args.n).astype(int)
    for k in idx:
        cam = cams[k]
        stem = os.path.splitext(getattr(cam, "image_name", f"v{k:04d}"))[0]
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg)
        rgb = pkg["render"].clamp(0, 1)
        dep = pkg["depth"];  dep = dep[0] if dep.dim() == 3 else dep
        nrm = pkg.get("rend_normal", None)
        cols = [rgb.cpu(), colorize(dep)]
        if nrm is not None:
            cols.append((nrm * 0.5 + 0.5).clamp(0, 1).cpu())      # normal 시각화
        row = torch.cat([c if c.shape[0] == 3 else c.repeat(3,1,1) for c in cols], dim=2)
        torchvision.utils.save_image(row, os.path.join(args.out_dir, f"{stem}_rgb_depth_normal.png"))
        # depth 단독도 크게 저장(얼룩 확인용)
        torchvision.utils.save_image(colorize(dep), os.path.join(args.out_dir, f"{stem}_depth.png"))
        print(f"{stem}: depth range [{float(dep[dep>0].min()) if (dep>0).any() else 0:.2f}, {float(dep.max()):.2f}]")

    print(f"→ {args.out_dir}  (*_rgb_depth_normal.png = RGB|depth|normal, *_depth.png)")
    print("판정: depth가 부드러우면 geometry OK(메쉬 문제는 추출법), 얼룩·계단·구멍이면 geometry 거침(distortion/densify 부족).")
