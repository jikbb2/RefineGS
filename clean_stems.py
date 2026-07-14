#!/usr/bin/env python3
"""gid별 3D-일관 프레임 목록 생성 (재학습 없이 포즈/ref 정화용).

audit_masks 와 동일 원리(마스크 back-project 중심의 지배 클러스터)를 전 프레임에 적용,
반복 median 으로 수렴한 중심에서 --keep_dist 이내 프레임의 stem 만 <out>/<gid>.txt 로 저장.

  python clean_stems.py --masks_root data/replica_room0_v2/masks \
    --gt_depth /home/elicer/nice-slam/Datasets/Replica/room0/results \
    --colmap data/replica_room0_v2/sparse/0 \
    --gids 0,1,10,11,12,14,15,16,17,18,19,2,20,22,23,24,27,28,3,31,32,34,35,36,37,38,4,5,6,7,8 \
    --out ~/See3D/dataset/stage6/clean_stems
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
    ap.add_argument("--keep_dist", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    for gid in args.gids.split(","):
        gid = gid.strip()
        mps = sorted(glob.glob(os.path.join(args.masks_root, gid, "masks", "*.png")))
        stems, cents = [], []
        for mp in mps:
            stem = os.path.splitext(os.path.basename(mp))[0]
            c = cams.get(stem)
            if c is None:
                continue
            m = load_mask(mp)
            if m.sum() < 30:
                continue
            dp = os.path.join(args.gt_depth, stem.replace("frame", "depth") + ".png")
            if not os.path.exists(dp):
                continue
            H, W = m.shape
            dep = load_depth(dp, args.depth_scale, W, H)
            vs, us = np.nonzero(m)
            st = max(1, len(vs) // 1500)
            vs, us = vs[::st], us[::st]
            d = dep[vs, us]
            ok = d > 1e-3
            if ok.sum() < 20:
                continue
            us, vs, d = us[ok], vs[ok], d[ok]
            x = (us - c["cx"]) / c["fx"] * d
            y = (vs - c["cy"]) / c["fy"] * d
            Xw = (np.stack([x, y, d], 1) - c["t"]) @ c["R"]
            stems.append(stem)
            cents.append(np.median(Xw, axis=0))
        if len(cents) < 5:
            print(f"gid {gid}: 유효 프레임 부족 — skip"); continue
        C = np.stack(cents)
        keep = np.ones(len(C), bool)
        for _ in range(5):                      # 반복 median → 지배 클러스터 수렴
            med = np.median(C[keep], axis=0)
            new = np.linalg.norm(C - med, axis=1) <= args.keep_dist
            if (new == keep).all():
                break
            keep = new
        kept = [s for s, k in zip(stems, keep) if k]
        with open(os.path.join(out, f"{gid}.txt"), "w") as f:
            f.write("\n".join(kept))
        print(f"gid {gid:>3}: {len(kept)}/{len(stems)} 유지 (제거 {len(stems)-len(kept)})")
    print(f"→ {out}/<gid>.txt")


if __name__ == "__main__":
    main()
