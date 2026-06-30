#!/usr/bin/env python3
"""RefineGS — Gaussian 레벨 carve 합성 (단일 객체 표현).

문제: base ⊕ recon ⊕ gen 을 그냥 concat 하면 base 의 객체(테이블) + recon + gen 이 *공존* → 2~3개 겹침.
해결: compose_scene.py(mesh) 가 하던 carve 를 Gaussian 으로 — **객체 bbox 안의 base 점을 제거**한 뒤
      recon(+gen)을 넣어 *하나의 객체*만 남긴다.

bbox: recon(관측 객체)의 axis-aligned bbox + pad (gen 은 오벌이라 bbox 기준으로 부적합).

용도 분리:
  warp scene (See3D 입력 렌더): --gen 없이 → base_carved ⊕ recon (미관측은 비어 hole 생김)
  학습 init:                    --gen 포함 → base_carved ⊕ recon ⊕ gen (gen=미관측 init)

태그(id_0): base_carved=0, recon=1, gen=2  (assemble_gaussians 와 동일 규약)

실행:
  # warp scene
  python carve_compose.py --base <base.ply> --recon <recon.ply> --pad 0.05 --tag \
    --out output/.../scene_b1_obj24_carved/point_cloud/iteration_1/point_cloud.ply
  # 학습 init (gen 포함)
  python carve_compose.py --base <base.ply> --recon <recon.ply> --gen <gen_surfels.ply> --pad 0.05 --tag \
    --out output/.../scene_b1_obj24_carved_gen/point_cloud/iteration_1/point_cloud.ply

Deps: numpy, plyfile.
"""
import argparse, os
import numpy as np
from plyfile import PlyData, PlyElement


def load(path):
    v = PlyData.read(path)["vertex"]
    return np.asarray(v.data), [p.name for p in v.properties]


def xyz_of(data):
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--recon", required=True, help="bbox 기준 + 포함(관측 객체)")
    ap.add_argument("--gen", default=None, help="포함 시 학습 init용(미관측 prior)")
    ap.add_argument("--pad", type=float, default=0.05, help="bbox 여유(m)")
    ap.add_argument("--tag", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base, bprops = load(a.base)
    recon, rprops = load(a.recon)
    if set(rprops) != set(bprops):
        raise SystemExit(f"recon 스키마 불일치: {sorted(set(bprops) ^ set(rprops))[:6]}")

    # 객체 bbox (recon 기준) + pad
    rxyz = xyz_of(recon)
    lo, hi = rxyz.min(0) - a.pad, rxyz.max(0) + a.pad
    print(f"object bbox lo={np.round(lo,3).tolist()} hi={np.round(hi,3).tolist()} (recon {len(recon)} pts)")

    # base carve: bbox 안 점 제거
    bxyz = xyz_of(base)
    inside = np.all((bxyz >= lo) & (bxyz <= hi), axis=1)
    base_keep = base[~inside]
    print(f"base carve: {inside.sum()} 제거 / {len(base)} → {len(base_keep)} 유지")

    chunks = [base_keep.copy(), recon.astype(base_keep.dtype, copy=True)]
    if a.tag and "id_0" in bprops:
        chunks[0]["id_0"] = 0
        chunks[1]["id_0"] = 1

    if a.gen:
        gen, gprops = load(a.gen)
        if set(gprops) != set(bprops):
            raise SystemExit(f"gen 스키마 불일치: {sorted(set(bprops) ^ set(gprops))[:6]}")
        gen = gen.astype(base_keep.dtype, copy=True)
        if a.tag and "id_0" in bprops:
            gen["id_0"] = 2
        chunks.append(gen)
        print(f"gen 포함: {len(gen)} (학습 init용)")

    merged = np.concatenate(chunks)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(a.out)
    print(f"merged {len(merged)} → {a.out}")
    print("검수: 렌더에서 테이블이 *하나*만 보이면 carve 성공. 두 개면 pad/bbox 조정.")


if __name__ == "__main__":
    main()
