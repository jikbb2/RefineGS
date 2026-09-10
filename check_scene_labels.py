#!/usr/bin/env python3
"""Decode per-gaussian labels from a scene model and check they are 3-D coherent.

A low CE only says the rendered embedding matches the 2-D masks. What we actually
need is that each label forms one compact blob in 3-D -- that is the property the
per-object fusion pipeline depends on, and the one that fixes SAM3 merge/split
errors.

Checks, in order of how much they tell you:
  1. compactness   largest connected component / total points per class.
                   A correct instance is ~1.0; a merged label (two objects with
                   one id) splits into several blobs and drops well below.
  2. size          points per class. A class with almost nothing was never learned.
  3. separation    distance from a class centroid to the nearest other centroid,
                   relative to its own radius. Low = two labels sit on one object.

  python check_scene_labels.py \
      --ply  OUT/scene/point_cloud/iteration_30000/point_cloud.ply \
      --head OUT/scene/label_head_30000.pth
"""
import argparse
import os

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

C0 = 0.28209479177387814


def load_ids(ply_path):
    p = PlyData.read(os.path.expanduser(ply_path))["vertex"]
    xyz = np.stack([p[k] for k in ("x", "y", "z")], 1).astype(np.float64)
    names = [n for n in p.data.dtype.names if n.startswith("id_")]
    assert names, "no id_* fields in the ply -- was the model trained with --label_dir?"
    ids = np.stack([p[n] for n in sorted(names)], 1).astype(np.float32)
    return xyz, ids


def assign(ids, head):
    P = head["lo"] + (head["hi"] - head["lo"]) * torch.sigmoid(head["proto_raw"])
    E = torch.from_numpy(np.clip(C0 * ids + 0.5, 0, 1))
    return torch.cdist(E, P).argmin(1).numpy(), P.numpy()


def largest_component_frac(pts, radius):
    """Fraction of points in the biggest connected blob (union-find over a radius graph)."""
    n = len(pts)
    if n < 2:
        return 1.0
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(pts).query_pairs(radius, output_type="ndarray"):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    return np.bincount(roots).max() / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--radius", type=float, default=0.05,
                    help="connectivity radius (m) for the compactness test")
    ap.add_argument("--max_pts", type=int, default=20000,
                    help="subsample per class before the O(n log n) graph")
    ap.add_argument("--min_pts", type=int, default=200)
    args = ap.parse_args()

    xyz, ids = load_ids(args.ply)
    head = torch.load(os.path.expanduser(args.head), map_location="cpu")
    lab, P = assign(ids, head)
    K = int(head["K"])
    print(f"[model] {len(xyz):,} gaussians, K={K} (+background)")

    rng = np.random.default_rng(0)
    rows = []
    cen = np.full((K + 1, 3), np.nan)
    for c in range(K + 1):
        m = lab == c
        n = int(m.sum())
        if n == 0:
            rows.append((c, 0, np.nan, np.nan)); continue
        pts = xyz[m]
        cen[c] = pts.mean(0)
        r = float(np.linalg.norm(pts - cen[c], axis=1).mean())
        if n > args.max_pts:
            pts = pts[rng.choice(n, args.max_pts, replace=False)]
        frac = largest_component_frac(pts, args.radius) if n >= args.min_pts else np.nan
        rows.append((c, n, frac, r))

    # nearest other centroid, relative to own radius
    sep = {}
    for c, n, _, r in rows:
        if n == 0 or not np.isfinite(cen[c]).all():
            continue
        d = np.linalg.norm(cen - cen[c], axis=1)
        d[c] = np.inf
        d[~np.isfinite(d)] = np.inf
        sep[c] = float(d.min()) / max(r, 1e-6)

    print(f"\n{'class':>6}{'points':>9}{'compact':>9}{'radius':>8}{'sep/r':>8}  note")
    bad = 0
    for c, n, frac, r in rows:
        if n == 0:
            print(f"{c:>6}{0:>9}{'-':>9}{'-':>8}{'-':>8}  never used")
            bad += 1
            continue
        s = sep.get(c, np.nan)
        note = []
        if frac == frac and frac < 0.8:
            note.append("split into blobs")
        if s == s and s < 1.0:
            note.append("overlaps a neighbour")
        if n < args.min_pts:
            note.append("tiny")
        bad += bool(note)
        print(f"{c:>6}{n:>9}{frac:>9.3f}{r:>8.3f}{s:>8.2f}  "
              f"{'BG' if c == 0 else ''}{', '.join(note)}")

    ok = [r_[2] for r_ in rows if r_[1] >= args.min_pts and r_[2] == r_[2]]
    print(f"\nclasses used {sum(1 for r_ in rows if r_[1] > 0)}/{K + 1}"
          f"   flagged {bad}")
    if ok:
        print(f"compactness  mean {np.mean(ok):.3f}  min {np.min(ok):.3f}"
              f"   (1.0 = one blob; < 0.8 means the label covers several objects)")
    print("\nCE only measures the 2-D fit. These numbers are what the fusion "
          "pipeline actually needs.")


if __name__ == "__main__":
    main()
