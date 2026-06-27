#!/usr/bin/env python3
"""RefineGS B3 — 베이스 2DGS 가우시안 ⊕ gen surfel(들)을 하나의 point_cloud.ply 로 병합.

모든 입력 ply 가 동일 스키마(Split&Splat 448필드)여야 함(mesh_to_surfels.py 출력과 베이스가 일치).
순서대로 vertex 데이터 concat. 출처 추적용 --tag_id 옵션: id_0 에 정수 태그를 써 둠
(0=base, 1,2,...=gen 객체) → 이후 joint 학습에서 unseen 보호 마스크로 활용 가능.

실행:
  python assemble_gaussians.py \
    --base output/replica_room0/scene_base/point_cloud/iteration_30000/point_cloud.ply \
    --gen  /tmp/surfels_obj29_gen.ply \
    --out  output/replica_room0_v2/scene_b1_obj29/point_cloud/iteration_1/point_cloud.ply \
    --tag

Deps: numpy, plyfile.
"""
import argparse, os
import numpy as np
from plyfile import PlyData, PlyElement


def load_vertex(path):
    v = PlyData.read(path)["vertex"]
    return v.data, [p.name for p in v.properties]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gen", nargs="+", required=True, help="하나 이상의 gen surfel ply")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", action="store_true",
                    help="id_0 에 출처 태그(base=0, gen i=1..) 기록")
    args = ap.parse_args()

    base_data, base_props = load_vertex(args.base)
    chunks = [np.asarray(base_data).copy()]
    if args.tag and "id_0" in base_props:
        chunks[0]["id_0"] = 0
    print(f"base: {len(base_data)} ({len(base_props)} props)")

    for i, g in enumerate(args.gen, start=1):
        gd, gp = load_vertex(g)
        if gp != base_props:
            # 필드 집합이 같은지(순서 무시) 확인
            if set(gp) != set(base_props):
                miss = set(base_props) ^ set(gp)
                raise SystemExit(f"스키마 불일치 {g}: 차이 {sorted(miss)[:8]}...")
            gd = np.asarray(gd)[base_props]      # 순서 맞춤
        gd = np.asarray(gd).astype(chunks[0].dtype, copy=True)
        if args.tag and "id_0" in base_props:
            gd["id_0"] = i
        chunks.append(gd)
        print(f"gen[{i}] {os.path.basename(g)}: {len(gd)}")

    merged = np.concatenate(chunks)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(args.out)
    print(f"merged {len(merged)} gaussians ({len(base_props)} props) → {args.out}")


if __name__ == "__main__":
    main()
