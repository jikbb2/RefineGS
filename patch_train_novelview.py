#!/usr/bin/env python3
"""patch_train_novelview.py — train.py 에 novel-view soft-weighted supervision 주입 (task 6, 멱등).
v2: 메모리 안전 — NV 이미지 lazy 로드(경로만 보관, 샘플될 때 GPU 로드) + NV 렌더를 -r 해상도로 축소.

gen_out/ (gen_*.jpg + weight_*.png + poses.npz) 를 추가 학습 카메라로.
  loss += nv_lambda * Σ(weight · |render - gen|) / Σweight   (매 nv_every 스텝)
weight=validity(검은색 제외) 또는 3단계 학습weight 등, 값 그대로 사용.

주의: 이전 v1 패치가 적용돼 있으면 먼저 되돌리고 재적용:
  cp train.py.bak_nv train.py     # v1 NV 패치 제거(init_ply 는 유지)
  python patch_train_novelview.py # v2 적용

densify: 미관측 성장 원하면 --densify_until_iter 0 를 *주지 말 것*. 메모리 빠듯하면 -r 4 + densify off.
실행:
  python train.py -s data/... -m output/... --init_ply <assembled.ply> \
    --novelview_dir <gen_out> --nv_lambda 0.5 --nv_every 2 -r 4 --densify_until_iter 0 --iterations 7000
"""
import os

P = "train.py"
s = open(P).read()
orig = s

# 1) CLI
anchor_arg = 'parser.add_argument("--start_checkpoint", type=str, default=None)'
if "--novelview_dir" not in s and anchor_arg in s:
    s = s.replace(anchor_arg, anchor_arg +
                  '\n    parser.add_argument("--novelview_dir", type=str, default=None)  # [NV]'
                  '\n    parser.add_argument("--nv_lambda", type=float, default=0.5)      # [NV]'
                  '\n    parser.add_argument("--nv_every", type=int, default=2)           # [NV]')

# 2) NV 카메라 로드 (경로만, lazy) — training_setup 직후
anchor_setup = "    gaussians.training_setup(opt)\n"
nv_load = (anchor_setup +
'''    # [NV] novel-view supervision (lazy: 경로만 보관, 렌더는 -r 해상도)
    _NV_CAMS, _NV_LAMBDA, _NV_EVERY = [], float(getattr(args, "nv_lambda", 0.5)), int(getattr(args, "nv_every", 2))
    if getattr(args, "novelview_dir", None):
        import os as _os, numpy as _np, torch as _t
        from scene.cameras import MiniCam as _MiniCam
        _res = int(getattr(args, "resolution", 1) or 1); _res = _res if _res > 0 else 1
        _nvd = args.novelview_dir
        _recs = _np.load(_os.path.join(_nvd, "poses.npz"), allow_pickle=True)["records"]
        for _r in _recs:
            _i = int(_r["idx"]); _gp = _os.path.join(_nvd, "gen_%04d.jpg" % _i); _wp = _os.path.join(_nvd, "weight_%04d.png" % _i)
            if not (_os.path.exists(_gp) and _os.path.exists(_wp)): continue
            _wvt = _t.tensor(_np.asarray(_r["world_view_transform"]), dtype=_t.float32).cuda()
            _fpt = _t.tensor(_np.asarray(_r["full_proj_transform"]), dtype=_t.float32).cuda()
            _W = max(int(_r["width"]) // _res, 1); _H = max(int(_r["height"]) // _res, 1)
            _cam = _MiniCam(_W, _H, float(_r["FoVy"]), float(_r["FoVx"]), 0.01, 100.0, _wvt, _fpt)
            _cam.gt_path = _gp; _cam.weight_path = _wp    # lazy
            _NV_CAMS.append(_cam)
        print("[NV] %d novel-view cams (lazy, render@1/%d) lambda=%.2f every=%d" % (len(_NV_CAMS), _res, _NV_LAMBDA, _NV_EVERY))
''')
if "[NV] novel-view supervision (lazy" not in s and anchor_setup in s:
    s = s.replace(anchor_setup, nv_load, 1)

# 3) loss 항 (total_loss.backward() 직전) — 샘플될 때만 이미지 GPU 로드
anchor_bw = "        total_loss.backward()\n"
nv_loss = (
'''        # [NV] novel-view weighted supervision (lazy load, 검은색/frustum-밖은 weight=0로 자동 제외)
        if _NV_CAMS and (iteration % _NV_EVERY == 0):
            from random import randint as _ri
            from PIL import Image as _Img
            import numpy as _np2
            _nv = _NV_CAMS[_ri(0, len(_NV_CAMS) - 1)]
            _nvr = render(_nv, gaussians, pipe, bg)["render"]
            _g = torch.from_numpy(_np2.asarray(_Img.open(_nv.gt_path).convert("RGB")).copy()).float().permute(2, 0, 1).cuda() / 255.0
            _w = torch.from_numpy(_np2.asarray(_Img.open(_nv.weight_path).convert("L")).copy()).float().cuda()[None] / 255.0
            if _g.shape[-2:] != _nvr.shape[-2:]:
                _g = torch.nn.functional.interpolate(_g[None], _nvr.shape[-2:], mode="bilinear")[0]
            if _w.shape[-2:] != _nvr.shape[-2:]:
                _w = torch.nn.functional.interpolate(_w[None], _nvr.shape[-2:], mode="bilinear")[0]
            _nv_l1 = (torch.abs(_nvr - _g) * _w).sum() / _w.sum().clamp_min(1.0)
            total_loss = total_loss + _NV_LAMBDA * _nv_l1
''' + anchor_bw)
if "[NV] novel-view weighted supervision" not in s and anchor_bw in s:
    s = s.replace(anchor_bw, nv_loss, 1)

if s != orig:
    open(P + ".bak_nv2", "w").write(orig)
    open(P, "w").write(s)
    print("patched train.py (novel-view v2 lazy). 백업: train.py.bak_nv2")
else:
    print("이미 패치됨 또는 anchor 불일치 — 수동 확인")
    print("  '--novelview_dir':", "--novelview_dir" in s, " NV load(lazy):", "[NV] novel-view supervision (lazy" in s)
