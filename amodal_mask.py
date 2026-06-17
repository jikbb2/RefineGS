#!/usr/bin/env python3
"""
축2 ❸ — amodal 마스크 생성: 객체 마스크의 *내부 구멍*을 메움.

modal 마스크가 객체 위 물체(램프 밑받침 등)를 제외해 생긴 내부 구멍을 채워,
객체 본래(amodal) 영역을 복원 → per-object recon 입력으로 사용.
경계(외곽)는 유지, 내부 구멍만 채움(=관측에 둘러싸인 occlusion 영역). scipy 불필요.

입력: re-labeling 출력 구조 <in_root>/<gid>/<stem>.png (또는 단일 폴더).
출력: <out_root>/<gid>/<stem>.png (amodal).

실행 (어느 env든; numpy+PIL):
    python amodal_mask.py --in_root ~/relabel_out --out_root ~/relabel_amodal
    python amodal_mask.py --in_dir ~/obj_masks_98 --out_dir ~/obj_masks_98_amodal  # 단일 폴더
"""
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw


def fill_interior_holes(mask_bool):
    """객체=True. 외곽 경계는 유지, 내부 구멍(객체에 둘러싸인 배경)만 채움."""
    if mask_bool.sum() == 0:
        return mask_bool
    H, W = mask_bool.shape
    # 배경(=~mask)을 테두리에서 flood → 바깥배경. 도달 못한 배경 = 내부 구멍.
    bg = (~mask_bool).astype(np.uint8) * 255          # 배경 255, 객체 0
    img = Image.fromarray(bg)
    for seed in [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1)]:
        if bg[seed[1], seed[0]] == 255:
            ImageDraw.floodfill(img, seed, 128)        # 바깥배경 → 128
    arr = np.array(img)
    holes = arr == 255                                 # 채워지지 않은 배경 = 내부 구멍
    return mask_bool | holes


def process_one(in_path, out_path):
    m = np.array(Image.open(in_path).convert("L")) > 0
    am = fill_interior_holes(m)
    Image.fromarray((am * 255).astype(np.uint8)).save(out_path)
    return int(m.sum()), int(am.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default=None, help="<gid>/<stem>.png 구조")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--in_dir", default=None, help="단일 폴더의 *.png")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    pairs = []
    if args.in_root:
        for gid in sorted(os.listdir(args.in_root)):
            sub = os.path.join(args.in_root, gid)
            if not os.path.isdir(sub): continue
            od = os.path.join(args.out_root, gid); os.makedirs(od, exist_ok=True)
            for f in glob.glob(os.path.join(sub, "*.png")):
                pairs.append((f, os.path.join(od, os.path.basename(f))))
    elif args.in_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for f in glob.glob(os.path.join(args.in_dir, "*.png")):
            pairs.append((f, os.path.join(args.out_dir, os.path.basename(f))))
    else:
        raise SystemExit("--in_root/--out_root 또는 --in_dir/--out_dir 필요")

    tot_m = tot_a = 0; n_filled = 0
    for ip, op in pairs:
        a, b = process_one(ip, op)
        tot_m += a; tot_a += b
        if b > a * 1.001: n_filled += 1
    print(f"마스크 {len(pairs)}개 처리. 구멍 메운 마스크 {n_filled}개. "
          f"평균 면적 {tot_m/max(len(pairs),1):.0f}→{tot_a/max(len(pairs),1):.0f}px")
    print("판정: 구멍 메운 마스크 수가 0보다 크고(객체 위 물체가 있던 마스크), "
          "amodal 면적>modal이면 occlusion 구멍 복원 성공.")


if __name__ == "__main__":
    main()
