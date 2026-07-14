#!/usr/bin/env python3
"""객체 마스크 오염 감사 — gid 내 다중 객체 혼입(over-merge) + 빈 마스크 검출.

프레임별 마스크를 GT depth로 back-project → 프레임 중심의 3D 산포로 판정:
  단일 객체: 중심들이 한 클러스터 / 혼입: 0.5m+ 떨어진 다봉.
출력: gid별 [빈마스크 수, 중심 산포(median→max), 이상 프레임 비율] + 혐의 gid 목록.

  python audit_masks.py --masks_root data/replica_room0_v2/masks \
    --gt_depth /home/elicer/nice-slam/Datasets/Replica/room0/results \
    --colmap data/replica_room0_v2/sparse/0 \
    --gids 0,1,10,11,12,14,15,16,17,18,19,2,20,22,23,24,27,28,3,31,32,34,35,36,37,38,4,5,6,7,8 \
    --n_frames 40

RefineGS repo 루트에서 실행. Deps: numpy, PIL, (repo) warp_gt_to_pose.
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image
from warp_gt_to_pose import read_colmap, load_depth


def load_mask(p):
    img = Image.open(p)
    a = np.array(img)
    if a.ndim == 3 and a.shape[2] == 4:
        return a[..., 3] > 0
    if a.ndim == 3:
        a = np.array(img.convert("L"))
    return (a > 0) if a.max() <= 1 else (a > 127)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_root", required=True)
    ap.add_argument("--gt_depth", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--gids", required=True)
    ap.add_argument("--depth_scale", type=float, default=6553.5)
    ap.add_argument("--n_frames", type=int, default=40, help="gid당 균등 샘플 프레임 수")
    ap.add_argument("--outlier_dist", type=float, default=0.5,
                    help="중앙 중심에서 이 거리(m) 초과 프레임 = 혼입 혐의")
    args = ap.parse_args()

    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    suspects = []
    print(f"{'gid':>4} | {'frames':>6} {'empty':>5} | {'spread_med':>10} {'spread_max':>10} {'outlier%':>8} | verdict")
    print("-" * 78)

    for gid in args.gids.split(","):
        gid = gid.strip()
        mps = sorted(glob.glob(os.path.join(args.masks_root, gid, "masks", "*.png")))
        if not mps:
            print(f"{gid:>4} | 마스크 없음"); continue
        sel = [mps[round(i * (len(mps) - 1) / max(args.n_frames - 1, 1))] for i in range(min(args.n_frames, len(mps)))]

        cents, n_empty = [], 0
        for mp in sel:
            stem = os.path.splitext(os.path.basename(mp))[0]
            c = cams.get(stem)
            if c is None:
                continue
            m = load_mask(mp)
            if m.sum() < 30:
                n_empty += 1
                continue
            dp = os.path.join(args.gt_depth, stem.replace("frame", "depth") + ".png")
            if not os.path.exists(dp):
                continue
            H, W = m.shape
            dep = load_depth(dp, args.depth_scale, W, H)
            vs, us = np.nonzero(m)
            st = max(1, len(vs) // 2000)
            vs, us = vs[::st], us[::st]
            d = dep[vs, us]
            ok = d > 1e-3
            us, vs, d = us[ok], vs[ok], d[ok]
            if len(d) < 20:
                continue
            x = (us - c["cx"]) / c["fx"] * d
            y = (vs - c["cy"]) / c["fy"] * d
            Xw = (np.stack([x, y, d], 1) - c["t"]) @ c["R"]
            cents.append(np.median(Xw, axis=0))   # 프레임 중심 (median — 프레임 내 혼입에도 강건)

        if len(cents) < 5:
            print(f"{gid:>4} | 유효 프레임 부족 ({len(cents)})"); continue
        C = np.stack(cents)
        med = np.median(C, axis=0)
        dist = np.linalg.norm(C - med, axis=1)
        out_frac = (dist > args.outlier_dist).mean()
        verdict = "OK"
        if out_frac > 0.15 or dist.max() > 1.5:
            verdict = "★MIXED?"
            suspects.append(gid)
        elif n_empty > len(sel) * 0.2:
            verdict = "empty↑"
            suspects.append(gid)
        print(f"{gid:>4} | {len(mps):>6} {n_empty:>5} | {np.median(dist):>9.2f}m {dist.max():>9.2f}m {out_frac*100:>7.1f}% | {verdict}")

    print(f"\n혐의 gid: {suspects if suspects else '없음'}")
    print("판정 가이드: spread_max>1.5m 또는 outlier>15% = 혼입 유력 → 프레임 3D 클러스터 분리 필요. "
          "empty↑ = 빈 마스크 다수(학습 뷰 낭비, 검정 프레임 원인).")


if __name__ == "__main__":
    main()
