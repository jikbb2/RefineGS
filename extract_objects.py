#!/usr/bin/env python3
"""Split a scene gaussian model into per-object model dirs using voted labels.

Each output is a normal 3DGS model directory, so the existing fusion / eval
pipeline runs on it unchanged:
    <out>/<label>/point_cloud/iteration_<it>/point_cloud.ply
    <out>/<label>/cameras.json, cfg_args      (copied from the scene model)

Optional --split_below runs connected components inside a label and keeps only
the largest blob. Compactness below ~0.5 means the label covers several objects
(measured: voted labels have mean compactness 0.754, but classes 4/9/14/20/30
sit at 0.16-0.40), and a merged label poisons the prior: ShapeR is asked to
complete two objects as one.

  python extract_objects.py --ply SCENE/point_cloud/iteration_30000/point_cloud.ply \\
      --labels SCENE/vote/labels.npy --scene_dir SCENE --out OUT/objects
"""
import argparse
import json
import os
import shutil

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def largest_blob(pts, k=3.0):
    """Indices of the biggest connected component; radius scales with point spacing."""
    n = len(pts)
    if n < 3:
        return np.arange(n)
    d, _ = cKDTree(pts).query(pts, k=2)
    rad = float(k * np.median(d[:, 1]))
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(pts).query_pairs(rad, output_type="ndarray"):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    big = np.bincount(roots).argmax()
    return np.nonzero(roots == big)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scene_dir", required=True, help="for cfg_args / cameras.json")
    ap.add_argument("--id_map", default="",
                    help="id_map.json from make_label_maps.py. With it, output dirs are "
                         "named by the ORIGINAL gid, so <masks>/<gid>/masks keeps working "
                         "and results line up with the per-object baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--min_pts", type=int, default=500, help="skip labels smaller than this")
    ap.add_argument("--skip_bg", action="store_true", default=True,
                    help="label 0 is background (walls/floor), not an object")
    ap.add_argument("--split_below", type=float, default=0.0,
                    help="if >0, keep only the largest blob when compactness < this")
    args = ap.parse_args()

    ply = PlyData.read(os.path.expanduser(args.ply))
    el = ply["vertex"]
    lab = np.load(os.path.expanduser(args.labels))
    assert len(lab) == len(el), f"labels {len(lab)} != gaussians {len(el)}"
    xyz = np.stack([el[k] for k in ("x", "y", "z")], 1).astype(np.float64)

    gid_of = {}
    if args.id_map:
        m = json.load(open(os.path.expanduser(args.id_map)))
        gid_of = {int(v): int(k) for k, v in m["label_of_gid"].items()}
        print(f"[extract] naming dirs by original gid ({len(gid_of)} labels mapped)")

    sd, od = os.path.expanduser(args.scene_dir), os.path.expanduser(args.out)
    os.makedirs(od, exist_ok=True)
    kept, skipped = [], []
    for c in range(int(lab.max()) + 1):
        if c == 0 and args.skip_bg:
            continue
        idx = np.nonzero(lab == c)[0]
        if len(idx) < args.min_pts:
            skipped.append((c, len(idx), "too small"))
            continue
        note = ""
        if args.split_below > 0:
            keep = largest_blob(xyz[idx])
            frac = len(keep) / len(idx)
            if frac < args.split_below:
                note = f"kept largest blob {frac * 100:.0f}%"
                idx = idx[keep]

        name = str(gid_of.get(c, c))
        d = os.path.join(od, name, "point_cloud", f"iteration_{args.iter}")
        os.makedirs(d, exist_ok=True)
        PlyData([PlyElement.describe(el.data[idx], "vertex")]).write(
            os.path.join(d, "point_cloud.ply"))
        for f in ("cfg_args", "cameras.json"):
            s = os.path.join(sd, f)
            if os.path.isfile(s):
                shutil.copy(s, os.path.join(od, name, f))
        kept.append((c, name, len(idx), note))

    print(f"[extract] {len(kept)} objects -> {od}")
    print(f"{'label':>6}{'dir':>6}{'gaussians':>11}  note")
    for c, name, n, note in kept:
        print(f"{c:>6}{name:>6}{n:>11,}  {note}")
    if skipped:
        print("\nskipped: " + ", ".join(f"{c}({n})" for c, n, _ in skipped))
    json.dump({"dirs": {name: c for c, name, _, _ in kept}, "iter": args.iter},
              open(os.path.join(od, "objects.json"), "w"), indent=1)
    print(f"\nnext: build fuse_post.ply for each dir, then run the fusion batch "
          f"with OUT={od}")


if __name__ == "__main__":
    main()