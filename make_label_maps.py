#!/usr/bin/env python3
"""Merge per-gid binary masks into per-view instance label maps.

Per-object training uses one mask at a time. Scene training needs a single
image with a per-pixel instance id so a label CE loss can be applied.

Outputs under --out:
  labels/<stem>.png   uint16, 0=background, 1..K=instance
  union/<stem>.png    uint8 0/255, foreground union (for alpha supervision)
  id_map.json         {gids, label_of_gid, K, overlap}

Overlap policy (two masks claiming one pixel = SAM3 boundary jitter):
  smallest : smaller object wins (default) -- keeps big objects from swallowing small ones
  first    : lower gid wins
  drop     : set to background, excluded from the loss (most conservative)
The overlap fraction is always printed; a high value means the labels
themselves are unreliable.

  python make_label_maps.py --masks_root DATA/masks --out DATA/labels_scene
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from PIL import Image


def load_mask(path, min_px):
    a = np.array(Image.open(path))
    m = (a[..., 3] > 0) if a.ndim == 3 and a.shape[2] == 4 else (a > 0)
    if m.ndim == 3:
        m = m.any(-1)
    return m if m.sum() >= min_px else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_root", required=True, help="parent of <gid>/masks/<stem>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="", help="gids to skip (comma)")
    ap.add_argument("--overlap", default="smallest", choices=["smallest", "first", "drop"])
    ap.add_argument("--min_px", type=int, default=50, help="ignore masks smaller than this")
    ap.add_argument("--limit", type=int, default=0, help="first N views only (sanity check)")
    args = ap.parse_args()

    root = os.path.expanduser(args.masks_root)
    ex = {int(x) for x in args.exclude.split(",") if x.strip()}
    gids = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(root, "*"))
                  if os.path.basename(d).isdigit() and int(os.path.basename(d)) not in ex)
    assert gids, f"no gid folders under {root}/<gid>/masks"
    lab_of = {g: i + 1 for i, g in enumerate(gids)}      # 0 reserved for background
    print(f"[gid] {len(gids)} -> labels 1..{len(gids)}  {gids}")

    stems = sorted({os.path.splitext(os.path.basename(p))[0]
                    for g in gids
                    for p in glob.glob(os.path.join(root, str(g), "masks", "*.png"))})
    if args.limit:
        stems = stems[:args.limit]
    print(f"[views] {len(stems)}")

    od = os.path.expanduser(args.out)
    os.makedirs(os.path.join(od, "labels"), exist_ok=True)
    os.makedirs(os.path.join(od, "union"), exist_ok=True)

    n_over = n_fg = n_tot = 0
    px_of, views_of = collections.Counter(), collections.Counter()
    for si, s in enumerate(stems):
        planes = []
        for g in gids:
            p = os.path.join(root, str(g), "masks", s + ".png")
            if os.path.isfile(p):
                m = load_mask(p, args.min_px)
                if m is not None:
                    planes.append((int(m.sum()), g, m))
        if not planes:
            continue

        shape = planes[0][2].shape
        lab = np.zeros(shape, np.uint16)
        cnt = np.zeros(shape, np.uint8)
        order = sorted(planes, key=lambda x: -x[0]) if args.overlap == "smallest" \
            else sorted(planes, key=lambda x: x[1])
        for _, g, m in order:                            # last write wins
            lab[m] = lab_of[g]
            cnt[m] += 1
        if args.overlap == "drop":
            lab[cnt > 1] = 0

        union = lab > 0
        n_over += int((cnt > 1).sum()); n_fg += int(union.sum()); n_tot += union.size
        for a_, g, _ in planes:
            px_of[g] += a_; views_of[g] += 1

        Image.fromarray(lab).save(os.path.join(od, "labels", s + ".png"))
        Image.fromarray((union * 255).astype(np.uint8)).save(
            os.path.join(od, "union", s + ".png"))
        if (si + 1) % 100 == 0:
            print(f"  {si + 1}/{len(stems)}")

    json.dump({"gids": gids, "label_of_gid": {str(k): v for k, v in lab_of.items()},
               "K": len(gids), "overlap": args.overlap},
              open(os.path.join(od, "id_map.json"), "w"), indent=1)

    ov = n_over / max(n_fg, 1)
    print(f"\n[stats] foreground {n_fg / max(n_tot, 1) * 100:.1f}% of pixels"
          f"   overlap {ov * 100:.2f}% of foreground (policy={args.overlap})")
    if ov > 0.05:
        print("  WARN overlap > 5% -- SAM3 masks disagree badly at boundaries")
    print(f"\n{'gid':>5}{'label':>7}{'views':>8}{'mean px':>10}")
    for g in gids:
        nv = views_of[g]
        print(f"{g:>5}{lab_of[g]:>7}{nv:>8}{px_of[g] // max(nv, 1):>10}")
    print(f"\n-> {od}/{{labels,union}}, id_map.json")


if __name__ == "__main__":
    main()
