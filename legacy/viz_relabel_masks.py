#!/usr/bin/env python3
"""relabel 인스턴스 마스크를 프레임 RGB 위에 색으로 오버레이 (정성 검수).

객체별 <relabel>/<gid>/<stem>.png 를 distinct 색으로 합성 → 한 장의 instance-seg 뷰.
여러 프레임(early/mid/late)을 주면 각각 출력 + contact sheet.

실행:
  python viz_relabel_masks.py --relabel ~/relabel_replica_room0_v2 \
      --scene_img data/replica_room0_v2/images \
      --stems frame000000,frame000500,frame001000,frame001500 \
      --out output/replica_room0_v2/relabel_viz
"""
import argparse, colorsys, glob, os
import numpy as np
from PIL import Image


def color(i, n):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.85, 1.0)
    return np.array([r, g, b]) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel", required=True)
    ap.add_argument("--scene_img", required=True, help="frameXXXX.jpg 폴더(심볼릭 OK)")
    ap.add_argument("--stems", required=True, help="콤마구분 frame stem들")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gids = sorted([d for d in os.listdir(a.relabel)
                   if os.path.isdir(os.path.join(a.relabel, d))], key=lambda x: int(x))
    stems = [s.strip() for s in a.stems.split(",") if s.strip()]
    pngs = []
    for stem in stems:
        rgb_p = os.path.join(a.scene_img, stem + ".jpg")
        if not os.path.exists(rgb_p):
            print("  [skip] rgb 없음:", rgb_p); continue
        rgb = np.array(Image.open(rgb_p).convert("RGB")).astype(np.float32)
        over = rgb.copy()
        present = 0
        for i, g in enumerate(gids):
            mp = os.path.join(a.relabel, g, stem + ".png")
            if not os.path.exists(mp):
                continue
            m = np.array(Image.open(mp).convert("L")) > 127
            if m.shape != rgb.shape[:2] or m.sum() == 0:
                continue
            c = color(i, len(gids))
            over[m] = (1 - a.alpha) * over[m] + a.alpha * c
            present += 1
            # 라벨 텍스트(중심)는 생략(헤드리스 단순화)
        outp = os.path.join(a.out, f"{stem}.png")
        Image.fromarray(over.astype(np.uint8)).save(outp)
        pngs.append(outp)
        print(f"  {stem}: objects present={present} -> {outp}")

    if len(pngs) > 1:
        ims = [Image.open(p) for p in pngs]
        w, h = ims[0].size
        cols = 2; rows = (len(ims) + 1) // 2
        sheet = Image.new("RGB", (w * cols, h * rows), "white")
        for i, im in enumerate(ims):
            sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
        sp = os.path.join(a.out, "contact_sheet.png"); sheet.save(sp)
        print("contact sheet ->", sp)


if __name__ == "__main__":
    main()
