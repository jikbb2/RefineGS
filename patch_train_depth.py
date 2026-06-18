#!/usr/bin/env python3
"""train.py 에 GT metric depth supervision 재추가 (관측 표면 품질).

- 헬퍼 _load_gt_depth: data/<scene>/masks/<gid>/depths/<frame-stem>.png 을
  meters(=raw/6553.5)로 읽어 render 해상도로 리사이즈, 객체 마스크∩유효깊이만.
- 학습 루프에서 render_pkg["depth"] 와 L1 (마스크 내부). iter>500 부터, lambda=0.5.

멱등: 이미 패치돼 있으면 건너뜀. 실행: python patch_train_depth.py
DEPTH_SCALE(6553.5)은 nice-slam Replica 기준. 다르면 아래 상수만 바꾸세요.
"""
import io, sys

F = "train.py"
s = open(F).read()

if "RefineGS depth supervision" in s:
    print("이미 패치됨 — 건너뜀"); sys.exit(0)

# 1) 헬퍼 삽입 (training 함수 정의 직전)
helper = '''
# === [RefineGS depth supervision] ====================================
_DEPTH_CACHE = {}
def _load_gt_depth(cam, source_path, scale=6553.5):
    """GT metric depth(meters) + (객체∩유효) 마스크. 캐시."""
    import os, cv2, numpy as np, torch
    key = getattr(cam, "image_name", None)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    stem = os.path.splitext(key)[0] if key else None
    p = os.path.join(source_path, "depths", stem + ".png") if stem else None
    if not p or not os.path.exists(p):
        _DEPTH_CACHE[key] = (None, None); return None, None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / scale  # meters
    d = cv2.resize(d, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
    gd = torch.from_numpy(d[None]).float().cuda()
    am = getattr(cam, "alpha_mask", None)
    am = am if am is not None else torch.ones_like(gd)
    valid = ((gd > 1e-3) & (am > 0.5)).float()
    _DEPTH_CACHE[key] = (gd, valid)
    return gd, valid
# =====================================================================

'''
anchor_def = "def training(dataset, opt, pipe,"
assert anchor_def in s, "training 함수 정의를 못 찾음"
s = s.replace(anchor_def, helper + anchor_def, 1)

# 2) 손실 삽입 (total_loss 계산과 backward 사이)
anchor_loss = "        total_loss = loss + dist_loss + normal_loss\n        total_loss.backward()"
assert anchor_loss in s, "total_loss/backward anchor 를 못 찾음 (train.py 버전 확인)"
new_loss = (
    "        total_loss = loss + dist_loss + normal_loss\n"
    "        # [RefineGS depth supervision] GT metric depth L1 (마스크 내부)\n"
    "        _ld = 0.5 if iteration > 500 else 0.0\n"
    "        if _ld > 0 and ('depth' in render_pkg):\n"
    "            _gd, _vm = _load_gt_depth(viewpoint_cam, dataset.source_path)\n"
    "            if _gd is not None and _vm.sum() > 0:\n"
    "                _rd = render_pkg['depth']\n"
    "                if _rd.dim() == 2: _rd = _rd[None]\n"
    "                _dl = (torch.abs(_rd - _gd) * _vm).sum() / _vm.sum().clamp_min(1.0)\n"
    "                total_loss = total_loss + _ld * _dl\n"
    "        total_loss.backward()"
)
s = s.replace(anchor_loss, new_loss, 1)

open(F, "w").write(s)
print("train.py 패치 완료: depth supervision 추가")
