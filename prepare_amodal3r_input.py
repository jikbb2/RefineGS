#!/usr/bin/env python3
"""Amodal3R 입력 준비 — 객체별 베스트 뷰 + 3-값 마스크 (README 규격).

mask: 배경=255(흰), 가시=188(회), 가림=0(검정).
  가시(188)  = relabel(modal) 마스크
  가림(0)    = amodal(채운) − 가시  (객체 위/뒤로 가려진 영역)
  배경(255)  = 나머지

베스트 뷰 = 가시 마스크 면적이 큰 상위 K 프레임.
출력: <out>/<gid>/rgb_<rank>.png, mask_<rank>.png  (run_amodal3r_poc.py 가 읽는 형식)

실행 (split_and_splat 등 PIL/numpy 환경):
  python prepare_amodal3r_input.py --scene replica_room0_v2 --gid 0 --topk 3 \
      --relabel ~/relabel_replica_room0_v2 --amodal ~/amodal_replica_room0_v2 \
      --scene_img data/replica_room0_v2/images --out /home/elicer/Amodal3R/input
"""
import argparse, glob, os
import numpy as np
from PIL import Image


def to_bool(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 3] if a.shape[2] == 4 else a[..., :3].mean(-1)  # alpha 우선
    return a > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="replica_room0_v2")
    ap.add_argument("--gid", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--relabel", default=os.path.expanduser("~/relabel_replica_room0_v2"))
    ap.add_argument("--amodal", default=os.path.expanduser("~/amodal_replica_room0_v2"))
    ap.add_argument("--scene_img", default=None, help="default: data/<scene>/images")
    ap.add_argument("--out", default="/home/elicer/Amodal3R/input")
    a = ap.parse_args()
    scene_img = a.scene_img or f"data/{a.scene}/images"
    scene_img = os.path.realpath(scene_img)

    vis_files = sorted(glob.glob(os.path.join(a.relabel, a.gid, "*.png")))
    if not vis_files:
        raise SystemExit(f"relabel 마스크 없음: {a.relabel}/{a.gid}")

    # 프레임별 가시 면적 → 상위 topk
    ranked = []
    for vf in vis_files:
        stem = os.path.splitext(os.path.basename(vf))[0]
        vis = to_bool(vf)
        ranked.append((vis.sum(), stem, vf))
    ranked.sort(reverse=True)
    ranked = ranked[:a.topk]

    outd = os.path.join(a.out, str(a.gid))
    os.makedirs(outd, exist_ok=True)
    for rank, (area, stem, vf) in enumerate(ranked):
        vis = to_bool(vf)
        af = os.path.join(a.amodal, a.gid, stem + ".png")
        filled = to_bool(af) if os.path.exists(af) else vis
        occ = filled & (~vis)
        # 3-값 마스크
        m = np.full(vis.shape, 255, np.uint8)   # bg
        m[vis] = 188                             # visible
        m[occ] = 0                               # occluded
        Image.fromarray(m, "L").save(os.path.join(outd, f"mask_{rank}.png"))
        # rgb
        rgb = os.path.join(scene_img, stem + ".jpg")
        if not os.path.exists(rgb):
            print(f"  [warn] rgb 없음: {rgb}"); continue
        Image.open(rgb).convert("RGB").save(os.path.join(outd, f"rgb_{rank}.png"))
        print(f"  rank{rank} {stem}: vis={int(vis.sum())} occ={int(occ.sum())} -> {outd}")
    print(f"prepared {len(ranked)} views for obj {a.gid} -> {outd}")


if __name__ == "__main__":
    main()
