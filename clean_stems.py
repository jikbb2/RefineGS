#!/usr/bin/env python3
"""Build a per-gid list of 3-D consistent frames (cleans the pose / reference set
without retraining).

Applies the same principle as audit_masks -- the dominant cluster of back-projected
mask centres -- to every frame, converges that centre by iterated median, and writes
the stems of the frames within --keep_dist of it to <out>/<gid>.txt.

Why this exists: a mask that drifts onto a neighbouring object, or a grazing view whose
back-projection lands metres away, pulls the per-object reference set off the object.
Those frames are removed here rather than at training time, so the conditioning points,
the fusion and the evaluation all see the same view list.

The output directory is what run_field_fusion_batch.sh reads as STEMS_DIR. Without it,
--stems is never passed and the evaluation falls back to every COLMAP view, which moves
the reported numbers.

  python clean_stems.py --masks_root data/replica_room0_v2/masks \
    --gt_depth "$HOME"/nice-slam/Datasets/Replica/room0/results \
    --colmap data/replica_room0_v2/sparse/0 \
    --gids all \
    --out data/replica_room0_v2/clean_stems
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image
from warp_gt_to_pose import read_colmap, load_depth


def load_mask(p):
    img = Image.open(p)
    a = np.array(img)
    if a.ndim == 3 and a.shape[2] == 4:
        return a[..., 3] > 0
    if a.ndim == 3:
        a = np.array(img.convert("L"))
    return (a > 0) if a.max() <= 1 else (a > 127)


def resolve_gids(spec, masks_root):
    """'all' discovers every numeric object directory under masks_root.

    Typing the list by hand is how a run silently covers a subset: the gid set differs
    per scene, and a missing id produces no error, only a shorter output.
    """
    if spec.strip().lower() != "all":
        return [g.strip() for g in spec.split(",") if g.strip()]
    gids = sorted((d for d in os.listdir(masks_root)
                   if d.isdigit() and os.path.isdir(os.path.join(masks_root, d, "masks"))),
                  key=int)
    print(f"[gids] discovered {len(gids)} object directories under {masks_root}")
    return gids


def main():
    ap = argparse.ArgumentParser(
        description="per-object 3-D consistent frame lists (stems) for evaluation")
    ap.add_argument("--masks_root", required=True)
    ap.add_argument("--gt_depth", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--gids", required=True,
                    help="comma-separated object ids, or 'all' to discover them from "
                         "--masks_root")
    ap.add_argument("--depth_scale", type=float, default=6553.5)
    ap.add_argument("--keep_dist", type=float, default=0.5,
                    help="keep a frame when its back-projected mask centre lies within "
                         "this distance (m) of the converged cluster centre")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    centers = {}

    for gid in resolve_gids(args.gids, args.masks_root):
        mps = sorted(glob.glob(os.path.join(args.masks_root, gid, "masks", "*.png")))
        stems, cents = [], []
        for mp in mps:
            stem = os.path.splitext(os.path.basename(mp))[0]
            c = cams.get(stem)
            if c is None:
                continue
            m = load_mask(mp)
            if m.sum() < 30:
                continue
            dp = os.path.join(args.gt_depth, stem.replace("frame", "depth") + ".png")
            if not os.path.exists(dp):
                continue
            H, W = m.shape
            dep = load_depth(dp, args.depth_scale, W, H)
            vs, us = np.nonzero(m)
            st = max(1, len(vs) // 1500)
            vs, us = vs[::st], us[::st]
            d = dep[vs, us]
            ok = d > 1e-3
            if ok.sum() < 20:
                continue
            us, vs, d = us[ok], vs[ok], d[ok]
            x = (us - c["cx"]) / c["fx"] * d
            y = (vs - c["cy"]) / c["fy"] * d
            Xw = (np.stack([x, y, d], 1) - c["t"]) @ c["R"]
            stems.append(stem)
            cents.append(np.median(Xw, axis=0))
        if len(cents) < 5:
            print(f"gid {gid}: too few valid frames -- skipped"); continue
        C = np.stack(cents)
        keep = np.ones(len(C), bool)
        for _ in range(5):              # iterated median -> converge on the dominant cluster
            med = np.median(C[keep], axis=0)
            new = np.linalg.norm(C - med, axis=1) <= args.keep_dist
            if (new == keep).all():
                break
            keep = new
        kept = [s for s, k in zip(stems, keep) if k]
        with open(os.path.join(out, f"{gid}.txt"), "w") as f:
            f.write("\n".join(kept))
        med = np.median(C[keep], axis=0)      # 3-D centre of the dominant cluster (measured)
        spread = float(np.percentile(np.linalg.norm(C[keep] - med, axis=1), 90))
        centers[gid] = dict(center=med.tolist(), radius=max(spread, 0.15))
        print(f"gid {gid:>3}: kept {len(kept)}/{len(stems)} (dropped {len(stems)-len(kept)})  "
              f"center {np.round(med,2).tolist()}  r90 {spread:.2f}")
    import json
    with open(os.path.join(out, "centers.json"), "w") as f:
        json.dump(centers, f, indent=1)
    print(f"-> {out}/<gid>.txt + centers.json")


if __name__ == "__main__":
    main()