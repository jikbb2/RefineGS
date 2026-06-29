#!/usr/bin/env python3
"""RefineGS — render-vs-GT 품질 맵 (task 9, 'observed≠good' 교정).

핵심 동기: confidence_map 의 'observed' 는 *카메라가 봤는가*만 본다. 봤지만 *복원이 깨진* geometry
(저opacity/floater/잘못된 depth)는 못 잡는다. → 학습 포즈에서 조립 scene 을 렌더해 **GT 이미지와 L1 오차**
를 직접 측정 = broken-but-observed 정량화. 이게 hole=gen∧¬obs 가 놓친 큰 refinement 타깃.

출력:
  per-view : <stem>_err.png(오차 히트맵, 밝을수록 깨짐) + <stem>_overlay.png(GT|render|err 3분할)
  per-Gaussian : <out_dir>/quality.npy  (정점순서, 0=좋음 ~ 1=깨짐, 역투영 평균오차 정규화)
                 + quality_qa.ply (red=broken/green=good)  → 품질 기반 weight/ routing 소스
  요약 : 평균/중앙 오차, 오차>thr 픽셀 비율

routing 해석:
  관측 ∧ 고오차 ∧ (저opacity|floater) → prune
  관측 ∧ 고오차 ∧ 정상opacity         → confidence-weighted 재학습(실측으로 교정; 생성 불요)
  미관측                              → See3D/prior

실행:
  python render_quality_map.py -m output/replica_room0_v2/scene_b1_obj24 -s data/replica_room0_v2 \
    --iteration 1 --max_views 20 --err_thr 0.1 \
    --out_dir output/replica_room0_v2/scene_b1_obj24/quality

Deps: torch, torchvision, numpy, matplotlib(없으면 grayscale).
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


def colorize(e):  # 0..1 → RGB heat (matplotlib 있으면 inferno)
    try:
        import matplotlib.cm as cm
        return torch.from_numpy(cm.inferno(e.cpu().numpy())[..., :3]).permute(2, 0, 1).float()
    except Exception:
        return e[None].repeat(3, 1, 1).cpu()


if __name__ == "__main__":
    parser = ArgumentParser(description="render-vs-GT 품질 맵")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--max_views", default=20, type=int)
    parser.add_argument("--err_thr", default=0.1, type=float, help="broken 판정 픽셀 오차 임계")
    parser.add_argument("--use_mask", action="store_true", help="original_mask 내부만 평가(객체 한정)")
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
    if args.max_views > 0 and len(cams) > args.max_views:
        idx = np.linspace(0, len(cams) - 1, args.max_views).astype(int)
        cams = [cams[i] for i in idx]

    xyz = gaussians.get_xyz.detach()
    N = xyz.shape[0]
    g_err = torch.zeros(N, device="cuda")     # per-Gaussian 누적 오차
    g_cnt = torch.zeros(N, device="cuda")

    view_means = []
    for k, cam in enumerate(cams):
        stem = os.path.splitext(getattr(cam, "image_name", f"v{k:04d}"))[0]
        with torch.no_grad():
            img = render(cam, gaussians, pipe, bg)["render"].clamp(0, 1)   # (3,H,W)
        gt = cam.original_image.cuda().clamp(0, 1)
        if gt.shape != img.shape:
            gt = torch.nn.functional.interpolate(gt[None], img.shape[-2:], mode="bilinear")[0]
        err = (img - gt).abs().mean(0)                                     # (H,W) 0..~1
        sel = None
        if args.use_mask and getattr(cam, "original_mask", None) is not None:
            m = cam.original_mask.cuda(); m = m[0] if m.dim() == 3 else m
            cand = m > 0.5
            if int(cand.sum()) > 0:
                sel = cand
            elif k == 0:
                print("[warn] original_mask 가 비어있음(전부 0) → 전체 이미지로 평가(--use_mask 무시)")
        vmean = float((err * sel).sum() / sel.sum()) if sel is not None else float(err.mean())
        view_means.append(vmean)

        # 저장: 히트맵 + 3분할(GT|render|err)
        torchvision.utils.save_image(colorize(err.clamp(0, 1)),
                                     os.path.join(args.out_dir, f"{stem}_err.png"))
        trip = torch.cat([gt.cpu(), img.cpu(), colorize(err.clamp(0, 1))], dim=2)
        torchvision.utils.save_image(trip, os.path.join(args.out_dir, f"{stem}_overlay.png"))

        # per-Gaussian 역투영: 중심을 이 뷰로 투영 → 그 픽셀 오차 누적
        with torch.no_grad():
            P = torch.cat([xyz, torch.ones(N, 1, device="cuda")], 1)
            clip = P @ cam.full_proj_transform                            # (N,4)
            w = clip[:, 3:4].clamp_min(1e-6)
            ndc = clip[:, :3] / w
            H, Wd = err.shape
            u = ((ndc[:, 0] * 0.5 + 0.5) * Wd).long()
            v = ((ndc[:, 1] * 0.5 + 0.5) * H).long()
            inb = (clip[:, 3] > 0) & (u >= 0) & (u < Wd) & (v >= 0) & (v < H)
            ui, vi = u[inb], v[inb]
            g_err[inb] += err[vi, ui]
            g_cnt[inb] += 1.0
        if k < 3 or (k + 1) % 10 == 0:
            print(f"[{k+1}/{len(cams)}] {stem}: mean err {vmean:.4f}")

    # per-Gaussian 품질(0좋음~1깨짐)
    q = (g_err / g_cnt.clamp_min(1)).cpu().numpy()
    seen = g_cnt.cpu().numpy() > 0
    qn = np.zeros(N, np.float32)
    if seen.any():
        hi = np.percentile(q[seen], 95) + 1e-6
        qn[seen] = np.clip(q[seen] / hi, 0, 1)
    np.save(os.path.join(args.out_dir, "quality.npy"), qn)
    np.save(os.path.join(args.out_dir, "seen.npy"), seen.astype(np.float32))  # 1=학습뷰에서 관측됨
    # QA ply
    from plyfile import PlyData, PlyElement
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    xyz_np = xyz.detach().cpu().numpy()
    arr = np.empty(N, dt)
    arr["x"], arr["y"], arr["z"] = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
    arr["red"] = (qn * 255).astype(np.uint8); arr["green"] = ((1 - qn) * 255).astype(np.uint8); arr["blue"] = 0
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(os.path.join(args.out_dir, "quality_qa.ply"))

    vm = np.array(view_means)
    print(f"\n=== 품질 요약 ===")
    print(f"  view mean err: {vm.mean():.4f}  (max {vm.max():.4f})")
    print(f"  per-Gaussian broken(quality>0.5): {(qn>0.5).mean():.3f}  (관측된 것 중)")
    print(f"→ {args.out_dir}  (*_overlay=GT|render|err, quality.npy, quality_qa.ply)")
    print("해석: render≠GT 인 곳 = 관측됐지만 깨진 geometry. observed/unobserved 가 못 잡은 refinement 타깃.")
