#!/usr/bin/env python3
"""RefineGS — novel pose 에서 hole 렌더 + reachability 진단 (S2+S3).

학습 카메라가 아니라 *novel pose* 에서 label-buffer(색 교체 2차 패스)로 hole 을 그린다.
목적: '미관측이지만 도달가능(reachable)' 표면(객체 윗면·옆면 등)이 어떤 novel pose 에서
보이는지 확인 = See3D 가 실제로 supervise 할 수 있는 영역인지 진단.

벽-인접 prior-bound 영역(예: cabinet 뒷면)은 어떤 novel pose 에서도 front-most 로 안 떠야 정상.
free-standing 객체라면 옆/아랫면 hole 이 일부 novel pose 에서 보여야 함.

pose 소스:
  --path generate : render.py 와 동일한 generate_path(학습카메라→smooth novel 궤적). 안전, 1순위.
  --path orbit    : 객체 중심 orbit(실험적; world_view 직접 구성). generate 가 커버 부족할 때.

색 교체 수식 등은 render_hole_masks.py 와 동일.

실행(1순위):
  python render_hole_novel.py \
    -m output/replica_room0_v2/scene_b1_<gid> -s data/replica_room0_v2 \
    --iteration 1 --hole_npy ~/tmp/hole_label.npy \
    --path generate --n_frames 120 --max_views 24 \
    --thr 0.5 --dilate 3 --out_dir output/replica_room0_v2/scene_b1_<gid>/holes_novel

Deps: torch, torchvision, numpy, (선택)scipy + Split&Splat 코드베이스.
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
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        t = torch.nn.functional.max_pool2d(t, 2*k+1, 1, k)
        return (t[0, 0].numpy() > 0.5)


def set_label_color(gaussians, label):
    fdc, frest = gaussians._features_dc, gaussians._features_rest
    saved = (fdc.detach().clone(), frest.detach().clone(), int(gaussians.active_sh_degree))
    dc = (label - 0.5) / C0
    with torch.no_grad():
        fdc[:, 0, 0] = dc; fdc[:, 0, 1] = dc; fdc[:, 0, 2] = dc
        frest.zero_()
    gaussians.active_sh_degree = 0
    return saved


def restore_color(gaussians, saved):
    fdc_s, frest_s, sh = saved
    with torch.no_grad():
        gaussians._features_dc.copy_(fdc_s)
        gaussians._features_rest.copy_(frest_s)
    gaussians.active_sh_degree = sh


def get_novel_cams(scene, mode, n_frames):
    train = scene.getTrainCameras()
    if mode == "generate":
        from utils.render_utils import generate_path
        return list(generate_path(train, n_frames=n_frames))
    elif mode == "orbit":
        # 실험적: 객체 중심(=장면 중심 근사) orbit. generate 가 부족할 때만.
        # train 카메라 평균 위치/거리를 반경으로, 고도 0~60° 스파이럴.
        import copy
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        centers = np.stack([c.camera_center.detach().cpu().numpy() for c in train])
        ctr = centers.mean(0)
        rad = float(np.linalg.norm(centers - ctr, axis=1).mean())
        base = train[0]
        out = []
        for i in range(n_frames):
            az = 2 * np.pi * i / n_frames
            el = np.deg2rad(60.0 * i / n_frames)        # 점진적 top-down
            cam_pos = ctr + rad * np.array([np.cos(az)*np.cos(el),
                                            np.sin(el),
                                            np.sin(az)*np.cos(el)])
            fwd = ctr - cam_pos; fwd /= (np.linalg.norm(fwd) + 1e-9)
            up = np.array([0., 1., 0.])
            right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
            up2 = np.cross(right, fwd)
            R = np.stack([right, -up2, fwd], 1)         # cam→world (col)
            Rw2c = R.T
            t = -Rw2c @ cam_pos
            cam = copy.copy(base)
            w2v = torch.tensor(getWorld2View2(Rw2c, t)).transpose(0, 1).float().cuda()
            proj = getProjectionMatrix(base.znear, base.zfar, base.FoVx, base.FoVy).transpose(0, 1).cuda()
            cam.world_view_transform = w2v
            cam.full_proj_transform = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            cam.camera_center = w2v.inverse()[3, :3]
            cam.image_name = f"orbit{i:04d}"
            out.append(cam)
        return out
    raise SystemExit(f"unknown path mode {mode}")


if __name__ == "__main__":
    parser = ArgumentParser(description="novel-pose hole render + reachability 진단")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--hole_npy", required=True)
    parser.add_argument("--path", default="generate", choices=["generate", "orbit"])
    parser.add_argument("--n_frames", default=120, type=int)
    parser.add_argument("--max_views", default=24, type=int, help="저장할 뷰 수(균등 subsample)")
    parser.add_argument("--thr", default=0.5, type=float)
    parser.add_argument("--dilate", default=0, type=int)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    label_np = np.load(args.hole_npy).astype(np.float32)
    n = gaussians.get_xyz.shape[0]
    if len(label_np) != n:
        raise SystemExit(f"hole_npy {len(label_np)} != gaussians {n}")
    label = torch.from_numpy(label_np).cuda()

    cams = get_novel_cams(scene, args.path, args.n_frames)
    if args.max_views > 0 and len(cams) > args.max_views:
        idx = np.linspace(0, len(cams) - 1, args.max_views).astype(int)
        cams = [cams[i] for i in idx]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"path={args.path} novel views={len(cams)}  per-Gaussian hole frac={label.mean().item():.3f}")
    fracs = []
    for i, cam in enumerate(cams):
        stem = getattr(cam, "image_name", f"nv{i:04d}"); stem = os.path.splitext(stem)[0]
        with torch.no_grad():
            rgb = render(cam, gaussians, pipe, background)["render"].clamp(0, 1)
        saved = set_label_color(gaussians, label)
        with torch.no_grad():
            lab = render(cam, gaussians, pipe, background)["render"][0].clamp(0, 1)
        restore_color(gaussians, saved)

        lab_np = lab.detach().cpu().numpy()
        hole = dilate_mask(lab_np >= args.thr, args.dilate)
        fracs.append(float(hole.mean()))

        torchvision.utils.save_image(lab.unsqueeze(0), os.path.join(args.out_dir, f"{stem}_label.png"))
        ov = rgb.clone(); hm = torch.from_numpy(hole.astype(np.float32)).cuda()
        ov[0] = torch.maximum(ov[0], hm); ov[1] = ov[1]*(1-0.5*hm); ov[2] = ov[2]*(1-0.5*hm)
        torchvision.utils.save_image(ov, os.path.join(args.out_dir, f"{stem}_overlay.png"))
        if i < 3 or (i+1) % 10 == 0:
            print(f"[{i+1}/{len(cams)}] {stem}: hole px {hole.mean():.4f}")

    fracs = np.array(fracs)
    print(f"\nreachability 진단: hole px  mean {fracs.mean():.4f}  max {fracs.max():.4f}  "
          f"(>0.5% 뷰 {int((fracs>0.005).sum())}/{len(fracs)})")
    print(f"→ {args.out_dir}")
    print("해석: 어떤 novel pose 의 overlay 에서 *관측 안 됐던 옆/아랫면*에 빨강이 뜨면 = reachable(See3D 타깃). "
          "모든 pose 에서 거의 안 뜨면 = prior-bound(예: 벽-인접 뒷면).")
