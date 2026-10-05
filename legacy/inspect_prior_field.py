#!/usr/bin/env python3
"""prior field npz 진단 — GPU/모델 없이 몇 초. 융합 로그가 묻혔을 때 σ·w 를 다시 확인.

  python inspect_prior_field.py ~/prior/obj1_field_ens6.npz --sigma_w 0 1 4
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--sigma_w", type=float, nargs="*", default=[0.0, 1.0, 4.0])
    ap.add_argument("--sigma_ref", type=float, default=0.05,
                    help="σ0(m). sdf_distill 의 --prior_sigma_ref 기본 = prior_trunc = 0.05")
    ap.add_argument("--trunc", type=float, default=0.05)
    args = ap.parse_args()

    z = np.load(os.path.expanduser(args.npz))
    F = z["field"].astype(np.float32)                     # metric SDF (m)
    vox = float(z["vox_world"])
    G = F.shape[0]
    inside = F < 0
    print(f"[field] {os.path.basename(args.npz)}  G={G}  voxel={vox*1000:.2f}mm")
    print(f"        내부 {inside.mean()*100:.2f}%  범위 [{F.min():.3f}, {F.max():.3f}]m  "
          f"|∇f| 정규화 배율 raw_g={float(z['raw_g']):.3f}")

    if "field_std" not in z.files:
        print("  σ 없음 (--ensemble 1 로 생성된 필드). σ 가중 불가")
        return
    S = z["field_std"].astype(np.float32)                 # 동일 metric 스케일
    near = np.abs(F) < 3 * vox                            # 표면 밴드
    band = np.abs(F) < args.trunc                         # truncation 밴드
    for nm, m in (("표면밴드(|f|<3vox)", near), ("trunc밴드(|f|<5cm)", band)):
        if not m.any():
            continue
        s = S[m] * 1000
        print(f"  σ[{nm}] 복셀 {int(m.sum())}  "
              f"중앙 {np.median(s):.1f}mm  90% {np.percentile(s,90):.1f}mm  "
              f"최대 {s.max():.1f}mm")

    print(f"\n  σ0={args.sigma_ref*1000:.0f}mm 기준 가중 w = 1/(1+w_sig·(σ/σ0)²)")
    print(f"  {'w_sig':>6} {'w중앙(표면)':>12} {'w<0.5비율':>10} {'유효표면감소':>12}")
    for wsig in args.sigma_w:
        W = 1.0 / (1.0 + wsig * (S / max(args.sigma_ref, 1e-6)) ** 2)
        # σ 가중 후 필드: v' = w·v + (1-w)·trunc  → 내부(음수)로 남는 복셀 비율 변화
        Fp = W * F + (1 - W) * args.trunc
        drop = (1 - (Fp < 0).sum() / max(inside.sum(), 1)) * 100
        wm = float(np.median(W[near])) if near.any() else float("nan")
        print(f"  {wsig:>6.1f} {wm:>12.3f} {(W < 0.5).mean()*100:>9.1f}% {drop:>11.1f}%")
    print("\n  ※ '유효표면감소' = σ 가중으로 내부 복셀이 줄어드는 비율. "
          "precision↑/recall↓ 의 크기 가늠용")


if __name__ == "__main__":
    main()
