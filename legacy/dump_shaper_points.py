#!/usr/bin/env python3
"""ShapeR 입력 pkl 의 조건 포인트(points_model)를 world 좌표 PLY 로 덤프 + 분포 진단.

make_shaper_input.py 는 샘플 점을 파일로 저장하지 않고 pkl 의 `points_model` 에만 넣는다.
ShapeR 가 학습한 SLAM 반정밀 포인트는 '물체 전체에 성기게' 퍼진 분포인데, 우리 입력은
'관측된 일부만 조밀하게' 덮는다 — 이 차이를 눈과 수치로 확인하기 위한 도구.

  python dump_shaper_points.py ~/ShapeR/data/obj6.pkl --out ~/prior/obj6_pts.ply

출력 통계
  - 점 수, world bbox, 최근접 이웃 간격(중앙값) ← 밀도 지표
  - 축별 점유 히스토그램 ← '어디가 비어 있는지'(예: 다리 높이대 공백) 확인
"""
import argparse
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--out", default="", help="world 좌표 PLY 경로(비우면 통계만)")
    ap.add_argument("--bins", type=int, default=12, help="축별 점유 히스토그램 구간 수")
    args = ap.parse_args()

    smp = pickle.load(open(os.path.expanduser(args.pkl), "rb"))
    P = smp["points_model"].numpy()[:, :3].astype(np.float64)      # 오브젝트 프레임
    Tmw = smp["T_model_world"].numpy()                             # world → model
    R = Tmw[:3, :3]
    c = -R.T @ Tmw[:3, 3]
    W = P @ R + c                                                  # → world
    b = smp["bounds"].numpy()
    scale = float(0.9 / np.max(b))

    print(f"[pkl] {os.path.basename(args.pkl)}")
    print(f"  points {len(W)}  caption='{smp.get('caption', smp.get('category', '-'))}'")
    print(f"  bounds(half-extent) {np.round(b, 3)}  scale {scale:.3f}")
    print(f"  world bbox  min {np.round(W.min(0), 3)}  max {np.round(W.max(0), 3)}  "
          f"extent {np.round(W.max(0) - W.min(0), 3)}")
    if "image_data" in smp:
        print(f"  views {len(smp['image_data'])}  "
              f"가시점 중앙값 {int(np.median([len(v) for v in smp['visible_points_model']]))}")

    # 밀도: 최근접 이웃 간격 (SLAM 반정밀 포인트는 보통 수 cm, 균등 메쉬 샘플은 수 mm)
    k = min(4000, len(W))
    sub = W[np.random.default_rng(0).choice(len(W), k, replace=False)]
    d = np.empty(k)
    for i in range(0, k, 256):
        dd = np.linalg.norm(sub[i:i + 256, None, :] - sub[None], axis=-1)
        np.fill_diagonal(dd[:, i:i + 256], np.inf)
        d[i:i + 256] = dd.min(1)
    print(f"  최근접 간격(서브샘플 {k}점): 중앙값 {np.median(d)*1000:.1f}mm  "
          f"90% {np.percentile(d, 90)*1000:.1f}mm")

    # 축별 점유: 비어 있는 구간(= 미관측 영역)을 드러낸다
    lo, hi = W.min(0), W.max(0)
    for ax, nm in enumerate("xyz"):
        h, _ = np.histogram(W[:, ax], bins=args.bins, range=(lo[ax], hi[ax]))
        # 0 이 '없음'인지 '아주 적음'인지 헷갈리지 않도록 기호를 분리한다
        lv = " ▁▂▃▄▅▆▇█"
        bar = "".join("·" if v == 0 else lv[max(1, min(8, int(8 * v / h.max())))] for v in h)
        print(f"  {nm}축 점유 [{lo[ax]:+.2f}..{hi[ax]:+.2f}] |{bar}|  "
              f"최대 {h.max()}점/구간  (· = 0점, ▁ = 소수)")

    if args.out:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(W)
        p = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        o3d.io.write_point_cloud(p, pc)
        print(f"→ {p}")


if __name__ == "__main__":
    main()