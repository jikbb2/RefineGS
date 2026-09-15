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
import re
import shutil

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def load_vote(vote_dir, n):
    """margin.npy / votes_total.npy from vote_labels.py, or (None, None).

    margin = (top vote - runner-up) / total. Low margin means the views disagreed, which
    happens exactly at object boundaries -- the gaussians that produce the dark fringe
    once the object is sliced out of the scene.
    """
    if not vote_dir:
        return None, None
    vd = os.path.expanduser(vote_dir)
    out = []
    for f in ("margin.npy", "votes_total.npy"):
        q = os.path.join(vd, f)
        out.append(np.load(q) if os.path.isfile(q) else None)
    for a in out:
        if a is not None and len(a) != n:
            print(f"[warn] {vd}: length {len(a)} != {n} gaussians -- vote filters off")
            return None, None
    return out[0], out[1]


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
    ap.add_argument("--source_root", default="",
                    help="per-gid data root, e.g. data/<scene>/masks. cfg_args copied "
                         "from the scene has source_path=<scene root>, so --mask_dir auto "
                         "resolves to <root>/masks which holds gid folders, not PNGs -- "
                         "every view is then skipped. Rewrite it to <source_root>/<gid>.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--min_pts", type=int, default=500, help="skip labels smaller than this")
    ap.add_argument("--skip_bg", action="store_true", default=True,
                    help="label 0 is background (walls/floor), not an object")
    ap.add_argument("--split_below", type=float, default=0.0,
                    help="if >0, keep only the largest blob when compactness < this")
    ap.add_argument("--vote_dir", default="",
                    help="vote_labels.py output dir (margin.npy, votes_total.npy). "
                         "Needed for --min_margin / --min_votes")
    ap.add_argument("--min_margin", type=float, default=0.0,
                    help="drop gaussians whose vote margin is below this. Contested "
                         "gaussians sit on object boundaries and carry the wrong label")
    ap.add_argument("--min_votes", type=int, default=0,
                    help="drop gaussians seen (unoccluded) in fewer views than this. A "
                         "gaussian the scene model never had to render correctly is "
                         "unconstrained, and slicing away its occluder exposes it")
    ap.add_argument("--min_opacity", type=float, default=0.0,
                    help="drop gaussians with sigmoid(opacity) below this. The dark fringe "
                         "on a sliced object is mostly low-opacity gaussians that were "
                         "never visible in the full scene")
    ap.add_argument("--stats_only", action="store_true",
                    help="print the per-label distributions and exit, so thresholds are "
                         "chosen from data instead of guessed")
    args = ap.parse_args()

    ply = PlyData.read(os.path.expanduser(args.ply))
    el = ply["vertex"]
    lab = np.load(os.path.expanduser(args.labels))
    assert len(lab) == len(el), f"labels {len(lab)} != gaussians {len(el)}"
    xyz = np.stack([el[k] for k in ("x", "y", "z")], 1).astype(np.float64)

    marg, votes = load_vote(args.vote_dir, len(el))
    opa = sigmoid(np.asarray(el["opacity"]).astype(np.float64)) \
        if "opacity" in el.data.dtype.names else None
    if args.min_margin > 0 and marg is None:
        raise SystemExit("--min_margin needs --vote_dir with margin.npy")
    if args.min_votes > 0 and votes is None:
        raise SystemExit("--min_votes needs --vote_dir with votes_total.npy")
    if args.min_opacity > 0 and opa is None:
        raise SystemExit("the ply has no opacity field")

    if args.stats_only:
        print(f"{'label':>6}{'gauss':>9}" + "".join(
            f"{h:>9}" for h in ("marg p10", "marg p50", "vote p10", "vote p50",
                                "opa p10", "opa p50")))
        for c in range(int(lab.max()) + 1):
            idx = np.nonzero(lab == c)[0]
            if len(idx) < args.min_pts:
                continue
            def q(a, p):
                return f"{np.percentile(a[idx], p):.3f}" if a is not None else "-"
            print(f"{c:>6}{len(idx):>9,}"
                  + "".join(f"{v:>9}" for v in (q(marg, 10), q(marg, 50),
                                                q(votes, 10), q(votes, 50),
                                                q(opa, 10), q(opa, 50))))
        print("\nPick --min_margin near the p10 of a label you know is clean, and check "
              "how many gaussians each threshold costs before committing.")
        return

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
        n_raw = len(idx)
        keepm = np.ones(n_raw, bool)
        if args.min_margin > 0:
            keepm &= marg[idx] >= args.min_margin
        if args.min_votes > 0:
            keepm &= votes[idx] >= args.min_votes
        if args.min_opacity > 0:
            keepm &= opa[idx] >= args.min_opacity
        idx = idx[keepm]
        if len(idx) < args.min_pts:
            skipped.append((c, len(idx), f"too small after filters ({n_raw} raw)"))
            continue
        note = f"filtered {n_raw - len(idx)} ({(n_raw - len(idx)) / n_raw * 100:.0f}%)" \
            if len(idx) < n_raw else ""
        if args.split_below > 0:
            keep = largest_blob(xyz[idx])
            frac = len(keep) / len(idx)
            if frac < args.split_below:
                note = (note + "  " if note else "") + f"largest blob {frac * 100:.0f}%"
                idx = idx[keep]

        name = str(gid_of.get(c, c))
        d = os.path.join(od, name, "point_cloud", f"iteration_{args.iter}")
        os.makedirs(d, exist_ok=True)
        PlyData([PlyElement.describe(el.data[idx], "vertex")]).write(
            os.path.join(d, "point_cloud.ply"))
        for f in ("cfg_args", "cameras.json"):
            s = os.path.join(sd, f)
            if not os.path.isfile(s):
                continue
            dst = os.path.join(od, name, f)
            if f == "cfg_args" and args.source_root:
                txt = open(s).read()
                sp = os.path.join(os.path.expanduser(args.source_root), name)
                txt = re.sub(r"source_path='[^']*'", f"source_path='{sp}'", txt)
                txt = re.sub(r"model_path='[^']*'",
                             f"model_path='{os.path.join(od, name)}'", txt)
                open(dst, "w").write(txt)
            else:
                shutil.copy(s, dst)
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