#!/usr/bin/env python3
"""RefineGS — 색 교체 2차 렌더 패스로 hole mask 생성 (S3-2, 능동적 hole authoring).

각 Gaussian 의 DC 색을 per-Gaussian hole 라벨(make_hole_labels.py 출력)로 임시 교체한 뒤
기존 gaussian_renderer.render() 를 그대로 호출 → 2DGS alpha-compositing 이 앞면 surfel 을 우선해
RGB 렌더와 픽셀 정합되는 'soft 라벨 이미지'를 만든다. threshold → 명시적 hole mask.

color-swap 수식 (active_sh_degree=0, features_rest=0):
    rendered_pixel ≈ C0 * dc + 0.5   (C0 = 0.2820948, 표준 3DGS SH0→RGB)
    label L 을 픽셀값 L 로 그리려면  dc = (L - 0.5) / C0,  3채널 동일, black bg.
    → 픽셀값 ≈ alpha-weighted 라벨.  ⚠️ 포크에 따라 색 변환이 다르면 _raw 이미지로 매핑 검증.

이 스크립트는 render.py 셋업을 미러링 → 동일한 -s / -m / --iteration 사용.
1차 테스트는 *학습 카메라*에서 렌더(novel pose 불필요) → hole 이 미관측 gen 영역에 정확히 떨어지는지 검수.

실행:
  python render_hole_masks.py \
    -m output/replica_room0_v2/scene_b1_obj29 -s data/replica_room0_v2 \
    --iteration 1 --hole_npy /tmp/hole_label.npy \
    --thr 0.5 --dilate 3 --out_dir output/replica_room0_v2/scene_b1_obj29/holes

출력 (out_dir):
  <stem>_label.png  soft 라벨(매핑 검증용)
  <stem>_hole.png   binary hole mask (white=hole)
  <stem>_overlay.png  RGB 렌더 위 hole 빨강 틴트 (정성 검수)

Deps: torch, torchvision, numpy, (선택) scipy[dilate], + Split&Splat 코드베이스(scene, gaussian_renderer).
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

C0 = 0.28209479177387814


def dilate_mask(m, k):
    if k <= 0:
        return m
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(m, iterations=int(k))
    except Exception:
        # scipy 없으면 maxpool 로 대체
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        t = torch.nn.functional.max_pool2d(t, kernel_size=2*k+1, stride=1, padding=k)
        return (t[0, 0].numpy() > 0.5)


def set_label_color(gaussians, label):
    """DC 색을 라벨로 교체. 원본 (features_dc, features_rest, active_sh_degree) 복원용 반환."""
    fdc = gaussians._features_dc       # (N,1,3)
    frest = gaussians._features_rest   # (N,R,3)
    saved = (fdc.detach().clone(), frest.detach().clone(), int(gaussians.active_sh_degree))
    dc_val = (label - 0.5) / C0        # (N,)
    with torch.no_grad():
        fdc[:, 0, 0] = dc_val
        fdc[:, 0, 1] = dc_val
        fdc[:, 0, 2] = dc_val
        frest.zero_()
    gaussians.active_sh_degree = 0
    return saved


def restore_color(gaussians, saved):
    fdc_s, frest_s, sh = saved
    with torch.no_grad():
        gaussians._features_dc.copy_(fdc_s)
        gaussians._features_rest.copy_(frest_s)
    gaussians.active_sh_degree = sh


if __name__ == "__main__":
    parser = ArgumentParser(description="hole mask via color-swap 2nd render pass")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--hole_npy", required=True, help="make_hole_labels.py 출력 (정점순서 정합)")
    parser.add_argument("--thr", default=0.5, type=float, help="hole 판정 임계")
    parser.add_argument("--dilate", default=0, type=int, help="hole mask dilate 픽셀")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_views", default=0, type=int, help=">0 이면 앞 N개 카메라만")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # black bg 필수 (빈 픽셀 = 0)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    label_np = np.load(args.hole_npy).astype(np.float32)
    n = gaussians.get_xyz.shape[0]
    if len(label_np) != n:
        raise SystemExit(f"hole_npy 길이 {len(label_np)} != gaussians {n}. "
                         f"동일 ply(iteration)인지 확인.")
    label = torch.from_numpy(label_np).cuda()

    os.makedirs(args.out_dir, exist_ok=True)
    cams = scene.getTrainCameras()
    if args.max_views > 0:
        cams = cams[:args.max_views]
    print(f"hole frac (per-Gaussian): {label.mean().item():.3f} | views: {len(cams)}")

    for i, cam in enumerate(cams):
        stem = getattr(cam, "image_name", f"view{i:04d}")
        stem = os.path.splitext(stem)[0]

        # (1) 원본 RGB 렌더 (overlay 용)
        with torch.no_grad():
            rgb = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)

        # (2) 색 교체 → 라벨 렌더
        saved = set_label_color(gaussians, label)
        with torch.no_grad():
            lab = render(cam, gaussians, pipe, background)["render"][0].clamp(0, 1)  # 채널0
        restore_color(gaussians, saved)

        lab_np = lab.detach().cpu().numpy()
        hole = lab_np >= args.thr
        hole = dilate_mask(hole, args.dilate)

        # 저장: 라벨(grayscale), hole(binary), overlay(red tint)
        torchvision.utils.save_image(lab.unsqueeze(0), os.path.join(args.out_dir, f"{stem}_label.png"))
        hole_t = torch.from_numpy(hole.astype(np.float32))[None]
        torchvision.utils.save_image(hole_t, os.path.join(args.out_dir, f"{stem}_hole.png"))
        ov = rgb.clone()
        hm = torch.from_numpy(hole.astype(np.float32)).cuda()
        ov[0] = torch.maximum(ov[0], hm)          # red 채널 강조
        ov[1] = ov[1] * (1 - 0.5 * hm)
        ov[2] = ov[2] * (1 - 0.5 * hm)
        torchvision.utils.save_image(ov, os.path.join(args.out_dir, f"{stem}_overlay.png"))

        if i < 3 or (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(cams)}] {stem}: hole px {hole.mean():.4f}")

    print(f"→ {args.out_dir} (*_label / *_hole / *_overlay)")
    print("검수 포인트: overlay 의 빨강이 *미관측 gen 영역*에만 떨어지는지, "
          "관측면/바닥/벽으로 새지 않는지 확인.")
