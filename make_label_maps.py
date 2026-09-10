#!/usr/bin/env python3
"""Merge per-gid binary masks into per-view instance label maps.

Per-object training uses one mask at a time. Scene training needs a single
image with a per-pixel instance id so a label CE loss can be applied.

Outputs under --out:
  labels/<stem>.png   uint16, 0=background, 1..K=instance, 65535=ignore
  union/<stem>.png    uint8 0/255, foreground union (for alpha supervision)
  id_map.json         {gids, label_of_gid, K, overlap, min_views, ignore}

Background is a real class (walls/floor), not ignored: the scene model contains
those gaussians and we want a label for them. IGNORE is for pixels we cannot
label honestly -- they are dropped from the CE loss but stay in the union mask,
so geometry is still supervised there.

Two sources of IGNORE:
  overlap    two masks claim one pixel (SAM3 boundary jitter). Measured 5.06%
             of foreground on replica_room0. Guessing an owner teaches noise.
  min_views  instances seen in too few views cannot be supervised; measured
             12 of 48 gids appear in <=16 views. Keeping them only adds CE
             classes with no data.

Overlap policy: ignore (default) | smallest | first

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
    ap.add_argument("--overlap", default="ignore", choices=["ignore", "smallest", "first"])
    ap.add_argument("--min_px", type=int, default=50, help="ignore masks smaller than this")
    ap.add_argument("--min_views", type=int, default=30,
                    help="instances seen in fewer views become IGNORE (0 = keep all)")
    ap.add_argument("--limit", type=int, default=0, help="first N views only (sanity check)")
    args = ap.parse_args()
    IGNORE = 65535

    root = os.path.expanduser(args.masks_root)
    ex = {int(x) for x in args.exclude.split(",") if x.strip()}
    gids = sorted(int(os.path.basename(d)) for d in glob.glob(os.path.join(root, "*"))
                  if os.path.basename(d).isdigit() and int(os.path.basename(d)) not in ex)
    assert gids, f"no gid folders under {root}/<gid>/masks"
    files_of = {g: glob.glob(os.path.join(root, str(g), "masks", "*.png")) for g in gids}
    rare = {g for g in gids if len(files_of[g]) < args.min_views}
    keep = [g for g in gids if g not in rare]
    lab_of = {g: i + 1 for i, g in enumerate(keep)}      # 0 reserved for background
    print(f"[gid] {len(gids)} found -> {len(keep)} kept as labels 1..{len(keep)}")
    if rare:
        print(f"      {len(rare)} rare (<{args.min_views} views) -> IGNORE: {sorted(rare)}")

    stems = sorted({os.path.splitext(os.path.basename(p))[0]
                    for g in gids for p in files_of[g]})
    if args.limit:
        stems = stems[:args.limit]
    print(f"[views] {len(stems)}")

    od = os.path.expanduser(args.out)
    os.makedirs(os.path.join(od, "labels"), exist_ok=True)
    os.makedirs(os.path.join(od, "union"), exist_ok=True)

    n_over = n_rare = n_fg = n_tot = 0
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
        rare_m = np.zeros(shape, bool)
        order = sorted(planes, key=lambda x: -x[0]) if args.overlap != "first" \
            else sorted(planes, key=lambda x: x[1])
        for _, g, m in order:                            # last write wins
            cnt[m] += 1
            if g in rare:
                rare_m |= m
            else:
                lab[m] = lab_of[g]
        union = cnt > 0                                  # union includes rare objects
        lab[rare_m & (lab == 0)] = IGNORE                # rare, not claimed by a kept one
        if args.overlap == "ignore":
            lab[cnt > 1] = IGNORE

        n_over += int((cnt > 1).sum()); n_rare += int(rare_m.sum())
        n_fg += int(union.sum()); n_tot += union.size
        for a_, g, _ in planes:
            px_of[g] += a_; views_of[g] += 1

        Image.fromarray(lab).save(os.path.join(od, "labels", s + ".png"))
        Image.fromarray((union * 255).astype(np.uint8)).save(
            os.path.join(od, "union", s + ".png"))
        if (si + 1) % 100 == 0:
            print(f"  {si + 1}/{len(stems)}")

    json.dump({"gids": keep, "label_of_gid": {str(k): v for k, v in lab_of.items()},
               "K": len(keep), "overlap": args.overlap,
               "min_views": args.min_views, "ignore": IGNORE,
               "rare_gids": sorted(rare)},
              open(os.path.join(od, "id_map.json"), "w"), indent=1)

    ov, rr = n_over / max(n_fg, 1), n_rare / max(n_fg, 1)
    print(f"\n[stats] foreground {n_fg / max(n_tot, 1) * 100:.1f}% of pixels")
    print(f"        overlap    {ov * 100:.2f}% of fg (policy={args.overlap})")
    print(f"        rare inst  {rr * 100:.2f}% of fg")
    if args.overlap == "ignore":
        print(f"        -> up to {(ov + rr) * 100:.1f}% of fg is IGNORE (no CE, "
              f"but still in union mask)")
    if ov > 0.05:
        print("  WARN overlap > 5% -- SAM3 masks disagree badly at boundaries")
    print(f"\n{'gid':>5}{'label':>7}{'views':>8}{'mean px':>10}")
    for g in gids:
        nv = views_of[g]
        tag = "IGNORE" if g in rare else str(lab_of[g])
        print(f"{g:>5}{tag:>7}{nv:>8}{px_of[g] // max(nv, 1):>10}")
    print(f"\nK={len(keep)}  -> {od}/{{labels,union}}, id_map.json")


if __name__ == "__main__":
    main()