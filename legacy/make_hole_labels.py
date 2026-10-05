#!/usr/bin/env python3
"""RefineGS — per-Gaussian hole 라벨 생성 (S3-1, 능동적 hole authoring).

assembled scene point_cloud.ply 의 정점 순서에 정합되는 hole 라벨(.npy, 1=hole/0=keep)을 만든다.
이후 render_hole_masks.py 가 이 라벨을 '색 교체 2차 렌더 패스'로 화면에 그려 hole mask 를 만든다.

hole 라벨 정의 (설계 결정 B):
    hole = gen-origin(id_0 태그) ∧ ¬observed(confidence_map _conf.npy)

- gen-origin: id_0 ∈ --gen_tags (assemble_gaussians 의 태그; base=0, gen=1..). 미지정 시 id_0 != 0 전부.
- observed:   --conf_npy (assembled 순서와 동일 길이, 1=good/0=bad). 미지정 시 observed 무시
              → hole = gen-origin 전체 (gen 이 어디에 geometry 를 추가했는지 시각화용 1차 테스트).

⚠️ --conf_npy 는 *assembled 전체 순서*와 길이가 같아야 함. assemble 단계(다음 스크립트)에서
   per-source _conf.npy 를 같은 순서로 concat 해 emit 할 예정. 지금은 없이도(gen-origin만) 테스트 가능.

실행:
  # 1차: gen-origin 만 (conf 없이)
  python make_hole_labels.py \
    --gaussians output/replica_room0_v2/scene_b1_obj29/point_cloud/iteration_1/point_cloud.ply \
    --gen_tags 1 --out /tmp/hole_label.npy
  # 2차: ¬observed 와 AND
  python make_hole_labels.py --gaussians <assembled.ply> --gen_tags 1,2,3 \
    --conf_npy /tmp/scene_conf.npy --out /tmp/hole_label.npy

Deps: numpy, plyfile.
"""
import argparse
import numpy as np
from plyfile import PlyData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", required=True, help="assembled point_cloud.ply")
    ap.add_argument("--gen_tags", default="",
                    help="gen-origin 으로 볼 id_0 값들 (콤마구분). 비우면 id_0 != 0 전부")
    ap.add_argument("--conf_npy", default=None,
                    help="confidence _conf.npy (assembled 순서, 1=good/0=bad). AND ¬observed")
    ap.add_argument("--dilate", type=int, default=0,
                    help="(미사용 placeholder) 2D 단계에서 dilate 권장 — 여기선 per-Gaussian 라벨만")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v = PlyData.read(a.gaussians)["vertex"]
    n = len(v.data)
    names = v.data.dtype.names
    if "id_0" not in names:
        raise SystemExit(f"id_0 필드 없음. assemble_gaussians 의 --tag 로 만든 ply 인지 확인. "
                         f"(필드: {sorted(names)[:12]}...)")
    tag = np.asarray(v["id_0"]).astype(np.int64)

    if a.gen_tags.strip():
        gen_set = set(int(t) for t in a.gen_tags.split(","))
        gen_origin = np.isin(tag, list(gen_set))
    else:
        gen_origin = tag != 0  # base=0 외 전부 gen

    hole = gen_origin.copy()
    if a.conf_npy:
        conf = np.load(a.conf_npy).astype(np.float32)
        if len(conf) != n:
            raise SystemExit(f"conf 길이 {len(conf)} != gaussians {n}. "
                             f"assembled 전체 순서로 concat 됐는지 확인.")
        observed = conf > 0.5
        hole = gen_origin & (~observed)

    np.save(a.out, hole.astype(np.float32))
    uniq, cnt = np.unique(tag, return_counts=True)
    print(f"gaussians {n}")
    print(f"  id_0 분포: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"  gen-origin: {gen_origin.mean():.3f} ({gen_origin.sum()})")
    if a.conf_npy:
        print(f"  observed(conf>0.5): {(conf>0.5).mean():.3f}")
    print(f"  HOLE(gen ∧ ¬obs): {hole.mean():.3f} ({hole.sum()})")
    print(f"→ {a.out}  (1=hole/0=keep, 정점순서 정합)")


if __name__ == "__main__":
    main()
