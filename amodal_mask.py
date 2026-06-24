#!/usr/bin/env python3
"""amodal_mask.py — fill INTERIOR occlusion holes in per-object binary masks.

입력: --in_root/<gid>/*.png  (relabel 출력; object=white(255) / bg=black(0))
출력: --out_root/<gid>/*.png (single-channel 0/255, 내부 구멍만 메움)

수정 이유: 이전 floodfill 버전이 프레임 전체를 255로 채워(over-fill) 객체 단위
학습이 장면 전체를 복원했음. 여기서는 scipy.ndimage.binary_fill_holes로
*객체에 의해 완전히 둘러싸인(enclosed) 배경 구멍*만 메운다. 프레임 전체나
경계에 연결된 배경은 절대 채워지지 않는다. max_hole_frac으로 비정상적으로
큰 구멍은 메우지 않아 폭주를 막고, 처리 후 객체 비율을 로그로 출력해 검증.

deps: numpy, Pillow, scipy (split_and_splat env).
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def to_bool(arr, invert=False):
    """L/RGB/RGBA 어떤 포맷이든 object=True 인 bool 마스크로."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)        # RGB 휘도 (alpha 무시)
    m = a > 127
    return ~m if invert else m


def fill_interior_holes(m, max_hole_frac=0.5):
    """m: bool(object=True). enclosed 내부 구멍만 메워 bool 반환."""
    filled = ndimage.binary_fill_holes(m)
    holes = filled & ~m
    if not holes.any():
        return m
    if max_hole_frac is None:
        return filled
    obj_area = max(int(m.sum()), 1)
    lbl, n = ndimage.label(holes)
    keep = np.zeros_like(holes)
    for i in range(1, n + 1):
        comp = lbl == i
        if comp.sum() <= max_hole_frac * obj_area:   # 거대한 '구멍'은 무시
            keep |= comp
    return m | keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_hole_frac", type=float, default=0.5)
    ap.add_argument("--invert", action="store_true",
                    help="relabel 마스크가 object=black 일 때만 사용")
    a = ap.parse_args()

    gids = sorted(d for d in os.listdir(a.in_root)
                  if os.path.isdir(os.path.join(a.in_root, d)))
    n_files = 0
    for g in gids:
        outd = os.path.join(a.out_root, g)
        os.makedirs(outd, exist_ok=True)
        fin = fout = 0.0
        files = sorted(glob.glob(os.path.join(a.in_root, g, "*.png")))
        for f in files:
            m = to_bool(Image.open(f), a.invert)
            out = fill_interior_holes(m, a.max_hole_frac) if m.any() else m
            # RGBA 저장: alpha 채널에 객체 마스크 (filterPLY/loadCam 둘 다 alpha를 사용)
            h, w = out.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = (out * 255).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(os.path.join(outd, os.path.basename(f)))
            fin += float(m.mean()); fout += float(out.mean()); n_files += 1
        k = max(len(files), 1)
        flag = "  <-- 의심(거의 전체)" if fout / k > 0.9 else ""
        print(f"  obj {g:>3}: in_frac={fin/k:.3f} out_frac={fout/k:.3f}{flag}")
    print(f"amodal done: {len(gids)} objects, {n_files} masks -> {a.out_root}")


if __name__ == "__main__":
    main()
