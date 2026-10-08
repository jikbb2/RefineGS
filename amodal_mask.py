#!/usr/bin/env python3
"""amodal_mask.py -- fill INTERIOR occlusion holes in per-object binary masks.

in : --in_root/<gid>/*.png   (relabel output; object = white 255, background = black 0)
out: --out_root/<gid>/*.png  (single channel 0/255, interior holes filled)

Why it was rewritten: the previous flood-fill version filled whole frames with 255, so the
per-object training reconstructed the entire scene. This fills only background holes
ENCLOSED by the object, using scipy.ndimage.binary_fill_holes -- a whole frame, or any
background connected to the border, can never be filled. max_hole_frac leaves an
implausibly large hole alone, which stops the runaway case, and the object fraction before
and after is printed per object so the result can be checked.

deps: numpy, Pillow, scipy (refinegs env).
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def to_bool(arr, invert=False):
    """Any of L / RGB / RGBA -> a bool mask with object = True."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)        # RGB luminance (alpha ignored)
    m = a > 127
    return ~m if invert else m


def fill_interior_holes(m, max_hole_frac=0.5):
    """m: bool with object = True. Fills only enclosed interior holes; returns bool."""
    filled = ndimage.binary_fill_holes(m)
    holes = filled & ~m
    if not holes.any():
        return m
    if max_hole_frac is None:
        return filled
    obj_area = max(int(m.sum()), 1)
    lbl, n = ndimage.label(holes)
    keep = np.zeros_like(holes)
    for i in range(1, n + 1):
        comp = lbl == i
        if comp.sum() <= max_hole_frac * obj_area:   # ignore an implausibly large 'hole'
            keep |= comp
    return m | keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_hole_frac", type=float, default=0.5)
    ap.add_argument("--invert", action="store_true",
                    help="only for relabel masks where the object is black")
    a = ap.parse_args()

    gids = sorted(d for d in os.listdir(a.in_root)
                  if os.path.isdir(os.path.join(a.in_root, d)))
    n_files = 0
    for g in gids:
        outd = os.path.join(a.out_root, g)
        os.makedirs(outd, exist_ok=True)
        fin = fout = 0.0
        files = sorted(glob.glob(os.path.join(a.in_root, g, "*.png")))
        for f in files:
            m = to_bool(Image.open(f), a.invert)
            out = fill_interior_holes(m, a.max_hole_frac) if m.any() else m
            # Saved as RGBA with the object mask in alpha: filterPLY and loadCam both read alpha.
            h, w = out.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = (out * 255).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(os.path.join(outd, os.path.basename(f)))
            fin += float(m.mean()); fout += float(out.mean()); n_files += 1
        k = max(len(files), 1)
        flag = "  <-- suspect: nearly the whole frame" if fout / k > 0.9 else ""
        print(f"  obj {g:>3}: in_frac={fin/k:.3f} out_frac={fout/k:.3f}{flag}")
    print(f"amodal done: {len(gids)} objects, {n_files} masks -> {a.out_root}")


if __name__ == "__main__":
    main()