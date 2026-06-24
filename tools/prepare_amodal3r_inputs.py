#!/usr/bin/env python3
"""
Prepare Amodal3R inputs from RefineGS per-object data.

For each instance label, finds the N best views (largest visible mask area),
then saves:
  <out_dir>/<label>/rgb_<rank>.png       — full RGB frame (PNG)
  <out_dir>/<label>/mask_<rank>.png      — visible mask (L, 0/255)
  <out_dir>/<label>/meta.json            — frame info, mask areas, paths

Usage (run from /home/elicer/RefineGS):
    python tools/prepare_amodal3r_inputs.py \
        --labels 97 98 75 \
        --data_root data/replica_room0/masks \
        --out_dir ~/Amodal3R/input \
        --top_k 3

Output layout (top_k=3):
    ~/Amodal3R/input/
      97/  rgb_0.png  mask_0.png  rgb_1.png  mask_1.png  rgb_2.png  mask_2.png  meta.json
      98/  ...
      75/  ...
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def load_alpha_mask(path: Path) -> np.ndarray:
    """Load binary mask from RGBA PNG: uses alpha channel (index 3).
    Returns uint8 array with 0/255 values."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA: real mask is in alpha channel
        return (arr[:, :, 3] > 0).astype(np.uint8) * 255
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # RGB (no alpha): use any non-zero channel
        return ((arr.max(axis=2)) > 0).astype(np.uint8) * 255
    else:
        # Grayscale
        return (arr > 0).astype(np.uint8) * 255


def find_best_views(masks_dir: Path, top_k: int = 3):
    """Return list of (mask_area, stem) sorted descending by mask area.
    Uses alpha channel for RGBA masks."""
    results = []
    for p in sorted(masks_dir.glob("*.png")):
        mask = load_alpha_mask(p)
        area = int((mask > 0).sum())
        if area > 0:
            results.append((area, p.stem))
    results.sort(reverse=True)
    return results[:top_k]


def stem_to_image(images_dir: Path, stem: str):
    """Find image file matching stem (handles .jpg / .JPG / .JPEG / .png)."""
    for ext in (".jpg", ".JPG", ".JPEG", ".jpeg", ".png", ".PNG"):
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="Instance labels to process")
    ap.add_argument("--data_root", default="data/replica_room0/masks",
                    help="Root dir with per-label subdirs")
    ap.add_argument("--out_dir", default="~/Amodal3R/input",
                    help="Output directory for Amodal3R inputs")
    ap.add_argument("--top_k", type=int, default=3,
                    help="Number of best views to extract per instance")
    ap.add_argument("--mask_dilate", type=int, default=0,
                    help="Optional dilation radius for mask (pixels)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir).expanduser()

    for label in args.labels:
        inst_dir = data_root / str(label)
        masks_dir = inst_dir / "masks"
        images_dir = inst_dir / "images"

        if not masks_dir.exists():
            print(f"[SKIP] {label}: masks_dir not found ({masks_dir})")
            continue
        if not images_dir.exists():
            print(f"[SKIP] {label}: images_dir not found ({images_dir})")
            continue

        best = find_best_views(masks_dir, top_k=args.top_k)
        if not best:
            print(f"[SKIP] {label}: no non-empty masks found")
            continue

        out_inst = out_dir / str(label)
        out_inst.mkdir(parents=True, exist_ok=True)

        meta = {"label": label, "views": []}
        print(f"\n[{label}] top-{args.top_k} views:")

        for rank, (area, stem) in enumerate(best):
            img_path = stem_to_image(images_dir, stem)
            mask_path = masks_dir / f"{stem}.png"

            if img_path is None:
                print(f"  rank{rank}: image not found for stem {stem}, skipping")
                continue

            # load & save RGB as PNG
            img = Image.open(img_path).convert("RGB")
            rgb_out = out_inst / f"rgb_{rank}.png"
            img.save(rgb_out)

            # load mask from alpha channel → 0/255 L
            mask_bin = load_alpha_mask(mask_path)
            if args.mask_dilate > 0:
                import cv2
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (args.mask_dilate*2+1,)*2)
                mask_bin = cv2.dilate(mask_bin, k, iterations=1)
            mask_img = Image.fromarray(mask_bin, mode="L")
            mask_out = out_inst / f"mask_{rank}.png"
            mask_img.save(mask_out)

            meta["views"].append({
                "rank": rank,
                "stem": stem,
                "mask_area_px": area,
                "rgb": str(rgb_out),
                "mask": str(mask_out),
                "image_size": list(img.size),   # [W, H]
            })
            print(f"  rank{rank}: {stem}  area={area}px  → {rgb_out.name}, {mask_out.name}")

        (out_inst / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  meta → {out_inst}/meta.json")

    print("\nDone. Summary:")
    for label in args.labels:
        out_inst = out_dir / str(label)
        meta_f = out_inst / "meta.json"
        if meta_f.exists():
            m = json.loads(meta_f.read_text())
            print(f"  label {label}: {len(m['views'])} views saved to {out_inst}")


if __name__ == "__main__":
    main()
