#!/usr/bin/env python3
"""patch_train_novelview.py — train.py 에 novel-view soft-weighted supervision 주입 (task 6, 멱등).

generate_novel_views.py 의 gen_out/ (gen_*.jpg + weight_*.png + poses.npz) 를 추가 학습 카메라로 넣고
  loss += nv_lambda * Σ(weight · |render - gen|) / Σweight
를 매 nv_every 스텝마다 더한다. weight(soft)가 ~0인 관측 픽셀은 기여 0 → 실측 보존,
weight~1 인 미관측 픽셀만 gen 으로 당김 = GENERATE supervision.

카메라는 scene/cameras.py 의 MiniCam 으로 구성(poses.npz 의 world_view/full_proj 그대로 사용).

densify: 미관측 geometry 를 새로 키우려면 densify ON 필요 →
  실제 refinement 는 `--densify_until_iter 0` 를 *주지 말 것*(기본 densify 켜짐).
  copy 백엔드 배관검증 때는 무관(loss≈0).

실행:
  cd /home/elicer/RefineGS && python patch_train_novelview.py
  # 그다음 학습(copy 배관검증 → 안정 확인 → see3d 로 교체):
  python train.py -s data/replica_room0_v2 -m output/.../scene_b1_obj24_refine_nv \
    --init_ply output/.../refine/point_cloud_pruned.ply \
    --novelview_dir ~/See3D/dataset/refinegs_obj24/gen_out \
    --nv_lambda 0.5 --nv_every 2 --iterations 3000 --save_iterations 3000
"""
import os

P = "train.py"
s = open(P).read()
orig = s

# ---- 1) CLI 인자 ----
anchor_arg = 'parser.add_argument("--start_checkpoint", type=str, default=None)'
if "--novelview_dir" not in s and anchor_arg in s:
    s = s.replace(anchor_arg, anchor_arg +
                  '\n    parser.add_argument("--novelview_dir", type=str, default=None)  # [NV]'
                  '\n    parser.add_argument("--nv_lambda", type=float, default=0.5)      # [NV]'
                  '\n    parser.add_argument("--nv_every", type=int, default=2)           # [NV]')

# ---- 2) NV 카메라 로드 (training_setup 직후) ----
anchor_setup = "    gaussians.training_setup(opt)\n"
nv_load = (anchor_setup +
'''    # [NV] novel-view soft-weighted supervision 로드
    _NV_CAMS, _NV_LAMBDA, _NV_EVERY = [], float(getattr(args, "nv_lambda", 0.5)), int(getattr(args, "nv_every", 2))
    if getattr(args, "novelview_dir", None):
        import os as _os, numpy as _np, torch as _t
        from PIL import Image as _Img
        from scene.cameras import MiniCam as _MiniCam
        _nvd = args.novelview_dir
        _recs = _np.load(_os.path.join(_nvd, "poses.npz"), allow_pickle=True)["records"]
        for _r in _recs:
            _r = _r.item() if hasattr(_r, "item") and not isinstance(_r, dict) else _r
            _i = int(_r["idx"])
            _gp = _os.path.join(_nvd, "gen_%04d.jpg" % _i)
            _wp = _os.path.join(_nvd, "weight_%04d.png" % _i)
            if not _os.path.exists(_gp) or not _os.path.exists(_wp):
                continue
            _wvt = _t.tensor(_np.asarray(_r["world_view_transform"]), dtype=_t.float32).cuda()
            _fpt = _t.tensor(_np.asarray(_r["full_proj_transform"]), dtype=_t.float32).cuda()
            _cam = _MiniCam(int(_r["width"]), int(_r["height"]), float(_r["FoVy"]), float(_r["FoVx"]),
                            0.01, 100.0, _wvt, _fpt)
            _g = _t.from_numpy(_np.asarray(_Img.open(_gp).convert("RGB"))).float().permute(2, 0, 1).cuda() / 255.0
            _w = _t.from_numpy(_np.asarray(_Img.open(_wp).convert("L"))).float().cuda() / 255.0
            _cam.gt_image = _g
            _cam.weight = _w[None]
            _NV_CAMS.append(_cam)
        print("[NV] %d novel-view cams (lambda=%.3f every=%d) from %s" % (len(_NV_CAMS), _NV_LAMBDA, _NV_EVERY, _nvd))
''')
if "[NV] novel-view soft-weighted supervision 로드" not in s and anchor_setup in s:
    s = s.replace(anchor_setup, nv_load, 1)

# ---- 3) loss 항 (total_loss.backward() 직전) ----
anchor_bw = "        total_loss.backward()\n"
nv_loss = (
'''        # [NV] novel-view weighted supervision (weight~0 관측 보존 / weight~1 미관측 refine)
        if _NV_CAMS and (iteration % _NV_EVERY == 0):
            from random import randint as _ri
            _nv = _NV_CAMS[_ri(0, len(_NV_CAMS) - 1)]
            _nvr = render(_nv, gaussians, pipe, bg)["render"]
            _w = _nv.weight
            if _w.shape[-2:] != _nvr.shape[-2:]:
                _w = torch.nn.functional.interpolate(_w[None], _nvr.shape[-2:], mode="bilinear")[0]
            _gt = _nv.gt_image
            if _gt.shape[-2:] != _nvr.shape[-2:]:
                _gt = torch.nn.functional.interpolate(_gt[None], _nvr.shape[-2:], mode="bilinear")[0]
            _nv_l1 = (torch.abs(_nvr - _gt) * _w).sum() / _w.sum().clamp_min(1.0)
            total_loss = total_loss + _NV_LAMBDA * _nv_l1
''' + anchor_bw)
if "[NV] novel-view weighted supervision" not in s and anchor_bw in s:
    s = s.replace(anchor_bw, nv_loss, 1)

if s != orig:
    open(P + ".bak_nv", "w").write(orig)
    open(P, "w").write(s)
    print("patched train.py (novel-view supervision). 백업: train.py.bak_nv")
else:
    print("이미 패치됨 또는 anchor 불일치 — 수동 확인 필요")
    print("  '--novelview_dir' present:", "--novelview_dir" in s)
    print("  NV load present:", "[NV] novel-view soft-weighted supervision 로드" in s)
    print("  NV loss present:", "[NV] novel-view weighted supervision" in s)
