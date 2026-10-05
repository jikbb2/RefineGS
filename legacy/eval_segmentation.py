#!/usr/bin/env python3
"""3D instance segmentation: per-object reconstructed meshes vs GT object_id.

Why: this project claims 3D segmentation as a contribution and had no metric for it.
GT mesh_semantic.ply carries object_id, so it is computable from the data we already have.

Method (point based):
  1) area-weighted uniform sampling of the GT scene mesh -> a GT object_id per point
  2) for each GT point find the nearest RECONSTRUCTED object, but only within --tau
     -> a predicted instance id per point; farther than tau counts as unassigned
  3) cross-tabulate (GT id x predicted id) and take IoU per instance
  4) precision / recall / F1 at IoU 0.25 and 0.50

  No confidence scores exist here, so a formal AP is not computed. With every prediction
  scored 1, AP at a threshold equals precision at that threshold, so P/R/F1 is reported
  instead to avoid implying more than the numbers support.

  Splits (one GT object covered by several predictions) and merges (one prediction
  spanning several GT objects) are counted separately: they are the dominant failure mode
  here and a mean IoU hides them.

  --exclude_classes drops GT instances of classes the pipeline never tries to
  reconstruct (walls, floor, ceiling, doors, windows). Leaving them in puts them in the
  recall denominator, which understates recall by a large factor: run_full_pipeline.sh
  already excludes them when selecting objects, so they can never be matched.

Usage:
  python eval_segmentation.py --gt_mesh ~/room_0/habitat/mesh_semantic.ply \
      --gt_info ~/room_0/habitat/info_semantic.json \
      --root ~/RefineGS/output/replica_room0_v2/refinegs_full --mesh fused_field_post.ply
  python eval_segmentation.py ... --mesh fuse_post.ply        # baseline
"""
import argparse
import collections
import glob
import json
import os

import re

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

DEFAULT_EXCLUDE = "0,9,20,21,25,26,28,29,31,33"
# same list run_full_pipeline.sh uses to pick objects, plus the room shell
DEFAULT_EXCLUDE_CLASSES = ("wall,floor,ceiling,door,window,blind,vent,"
                           "light switch,thermostat,rug,stair,beam,panel,pillar")


def load_class_names(path):
    """object_id -> class name, from Replica's info_semantic.json.

    Two layouts appear in the wild; support both and fall back to an empty map so the
    script still runs without the file.
    """
    if not path:
        return {}
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print(f"[warn] no info_semantic.json at {path} -- class exclusion is off")
        return {}
    m = json.load(open(path))
    by_cls = {}
    for c in m.get("classes", []):
        if isinstance(c, dict) and "id" in c:
            by_cls[int(c["id"])] = str(c.get("name", "")).strip().lower()
    out = {}
    for o in m.get("objects", []):
        if not isinstance(o, dict) or "id" not in o:
            continue
        nm = o.get("class_name")
        if nm is None:
            nm = by_cls.get(int(o.get("class_id", -1)), "")
        out[int(o["id"])] = str(nm).strip().lower()
    if not out and "id_to_label" in m:
        for i, c in enumerate(m["id_to_label"]):
            out[i] = by_cls.get(int(c), "")
    print(f"[info] class names for {len(out)} GT objects from {os.path.basename(path)}")
    return out


def _tok(name):
    """Class name -> token set. 'indoor-plant' -> {indoor, plant, indoor-plant}."""
    parts = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    return set(parts) | {name.lower().strip()}


def _hits(name, terms):
    """True when a term matches the whole name or one of its tokens.

    Substring matching is wrong here: 'door' is a substring of 'indoor-plant', which
    silently removed a reconstructable object from the evaluation. Multi-word terms
    ('light switch') are compared against the full name, and a trailing 's' is ignored so
    'blind' still matches 'blinds'.
    """
    tk = _tok(name)
    for t in terms:
        t = t.strip().lower()
        if not t:
            continue
        if " " in t or "-" in t:
            if t.replace("-", " ") in name.lower().replace("-", " "):
                return True
        elif t in tk or t + "s" in tk or (t.endswith("s") and t[:-1] in tk):
            return True
    return False


def load_gt(path):
    """GT mesh plus a per-face object_id; quads are fan-triangulated."""
    p = PlyData.read(os.path.expanduser(path))
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)
    fe = p["face"]
    idx = fe["vertex_indices"]
    oid = np.asarray(fe["object_id"]) if "object_id" in fe.data.dtype.names else None
    assert oid is not None, "the GT mesh has no object_id"
    T, L = [], []
    for f, o in zip(idx, oid):
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1])); L.append(o)
    return V, np.asarray(T, np.int64), np.asarray(L)


def sample_tris(V, T, L, n, seed=0):
    rng = np.random.default_rng(seed)
    e1 = V[T[:, 1]] - V[T[:, 0]]; e2 = V[T[:, 2]] - V[T[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    i = rng.choice(len(T), n, p=area / area.sum())
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    P = ((1 - r1)[:, None] * V[T[i, 0]] + (r1 * (1 - r2))[:, None] * V[T[i, 1]]
         + (r1 * r2)[:, None] * V[T[i, 2]])
    return P, L[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_info", default="",
                    help="Replica info_semantic.json, for --exclude_classes. Without it no "
                         "class-based exclusion happens and recall is measured against "
                         "every GT instance, including walls and floors we never build")
    ap.add_argument("--exclude_classes", default=DEFAULT_EXCLUDE_CLASSES,
                    help="comma-separated GT class names to drop entirely. 'none' keeps "
                         "them. Matching is on whole tokens, not substrings: a plain "
                         "substring test made 'door' swallow 'indoor-plant'")
    ap.add_argument("--root", required=True, help="parent of the per-object model dirs")
    ap.add_argument("--mesh", default="fused_field_post.ply")
    ap.add_argument("--fallback", default="fuse_post.ply",
                    help="file to use when --mesh is absent, e.g. a gate-blocked object. "
                         "Empty skips the object")
    ap.add_argument("--iter", default=7000, type=int)
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                    help="reconstructed gids to skip; 'none' keeps all")
    ap.add_argument("--tau", type=float, default=0.02,
                    help="max distance (m) for assigning a GT point to an instance; "
                         "farther than this is unassigned")
    ap.add_argument("--n_sample", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_gt_pts", type=int, default=200,
                    help="drop GT instances with fewer sampled points than this (noise)")
    args = ap.parse_args()

    V, T, L = load_gt(args.gt_mesh)
    G, GL = sample_tris(V, T, L, args.n_sample, args.seed)
    print(f"[GT] {len(T)} tris, {len(np.unique(L))} object_ids -> {len(G)} samples")

    cls_of = load_class_names(args.gt_info)
    drop_cls = set() if args.exclude_classes.strip().lower() in ("none", "") else {
        c.strip().lower() for c in args.exclude_classes.split(",") if c.strip()}
    dropped = collections.Counter()
    if cls_of and drop_cls:
        bad = {o for o, nm in cls_of.items() if nm and _hits(nm, drop_cls)}
        for o in bad:
            n = int((GL == o).sum())
            if n:
                dropped[cls_of[o]] += 1
        keep = ~np.isin(GL, list(bad))
        print(f"[exclude] dropped {int((~keep).sum())}/{len(G)} GT points "
              f"({sum(dropped.values())} instances): "
              + ", ".join(f"{k} x{v}" for k, v in dropped.most_common(8)))
        G, GL = G[keep], GL[keep]
    elif drop_cls:
        print("[exclude] class exclusion requested but no class names available "
              "-- pass --gt_info")

    ex = set() if args.exclude.strip().lower() in ("none", "") else \
        {int(x) for x in args.exclude.split(",") if x.strip()}

    # gather every reconstructed object into one cloud with an instance id per point
    P, PI, used, fell = [], [], [], []
    for d in sorted(glob.glob(os.path.join(os.path.expanduser(args.root), "*"))):
        g = os.path.basename(d)
        if not g.isdigit() or int(g) in ex:
            continue
        od = os.path.join(d, "train", f"ours_{args.iter}")
        p = os.path.join(od, args.mesh)
        if not os.path.isfile(p) and args.fallback:
            p2 = os.path.join(od, args.fallback)
            if os.path.isfile(p2):
                p = p2; fell.append(g)
        if not os.path.isfile(p):
            continue
        m = o3d.io.read_triangle_mesh(p)
        if len(m.vertices) < 100:
            continue
        try:
            pc = m.sample_points_uniformly(number_of_points=20000, seed=args.seed)
        except TypeError:
            pc = m.sample_points_uniformly(number_of_points=20000)
        q = np.asarray(pc.points)
        P.append(q); PI.append(np.full(len(q), int(g))); used.append(int(g))
    assert P, f"no reconstructed mesh found -- check --root / --mesh ({args.mesh})"
    P = np.concatenate(P); PI = np.concatenate(PI)
    print(f"[recon] {len(used)} instances, {len(P):,} points"
          + (f"  (fallback: {', '.join(fell)})" if fell else ""))

    # GT point -> nearest reconstructed instance, within tau only
    d, j = cKDTree(P).query(G, workers=-1)
    pred = np.where(d < args.tau, PI[j], -1)
    print(f"[assign] {100*(pred>=0).mean():.1f}% of GT points have a reconstructed "
          f"instance within {args.tau*1000:.0f}mm")

    # cross-tabulate -> IoU
    gt_ids = [g for g in np.unique(GL) if (GL == g).sum() >= args.min_gt_pts]
    inter = collections.Counter(zip(GL[pred >= 0], pred[pred >= 0]))
    gt_cnt = collections.Counter(GL.tolist())
    pr_cnt = collections.Counter(pred[pred >= 0].tolist())

    best = {}                                  # gt_id -> (iou, pred_id)
    for g in gt_ids:
        cand = [(k[1], c) for k, c in inter.items() if k[0] == g]
        if not cand:
            best[g] = (0.0, None); continue
        pid, ci = max(cand, key=lambda x: x[1])
        iou = ci / (gt_cnt[g] + pr_cnt[pid] - ci)
        best[g] = (iou, pid)

    # splits and merges: the dominant failure mode here
    split = collections.defaultdict(set)       # gt -> several preds
    merge = collections.defaultdict(set)       # pred -> several gts
    for (g, p_), c in inter.items():
        if c >= args.min_gt_pts:
            split[g].add(p_); merge[p_].add(g)
    n_split = sum(1 for v in split.values() if len(v) > 1)
    n_merge = sum(1 for v in merge.values() if len(v) > 1)

    # a very low IoU is not a match, just a graze against a neighbour -- hide the id
    MIN_SHOW = 0.05
    print(f"\n{'GT id':>7}{'class':>16}{'pts':>9}{'IoU':>8}  matched instance "
          f"(IoU<{MIN_SHOW} counts as unmatched)")
    n_hidden = 0
    for g in sorted(gt_ids, key=lambda x: -best[x][0]):
        iou, pid = best[g]
        if iou < MIN_SHOW:
            n_hidden += 1
            if n_hidden > 5:                    # fold a long tail
                continue
            pid = None
        extra = f"  (split into {len(split[g])})" if len(split.get(g, ())) > 1 else ""
        nm = (cls_of.get(int(g), "") or "")[:15]
        print(f"{g:>7}{nm:>16}{gt_cnt[g]:>9}{iou:>8.3f}  "
              f"{pid if pid is not None else 'unmatched'}{extra}")
    if n_hidden > 5:
        print(f"{'...':>7}{'':>16}{'':>9}{'':>8}  {n_hidden} unmatched (list folded)")

    ious = np.array([best[g][0] for g in gt_ids])
    print(f"\n=== summary ({len(gt_ids)} GT instances, tau={args.tau*1000:.0f}mm) ===")
    print(f"  coverage    {len(used)} reconstructed / {len(gt_ids)} GT "
          f"({len(used)/max(len(gt_ids),1)*100:.0f}%)  <- upper bound on recall")
    print(f"  mIoU                   {ious.mean():.4f}")
    print(f"  mIoU (matched only)    {ious[ious > 0].mean() if (ious > 0).any() else 0:.4f}"
          f"  ({int((ious > 0).sum())} instances)")
    for t in (0.25, 0.50):
        tp = int((ious >= t).sum())
        prec = tp / max(len(used), 1)          # against predicted instances
        rec = tp / max(len(gt_ids), 1)         # against GT instances
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  IoU>={t:.2f}  TP {tp:>3}   P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}")
    print(f"  split (one GT -> several predictions)  {n_split}")
    print(f"  merge (one prediction -> several GT)   {n_merge}")
    print(f"\nNo confidence scores, so P/R/F1 is reported instead of AP.")
    print(f"Recall is against {len(gt_ids)} GT instances"
          + (f" after dropping {sum(dropped.values())} of excluded classes."
             if dropped else
             " -- pass --gt_info to drop walls/floors the pipeline never builds."))


if __name__ == "__main__":
    main()