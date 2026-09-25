#!/usr/bin/env python3
"""Instance-segmentation metrics for the voted object set, against Replica GT instances.

Why this exists:
  The project measures geometry (seen accuracy) and unseen completion, but the third axis
  -- segmentation -- had no metric at all, so its failures only showed up indirectly. Room1
  made that untenable: 7 of 14 objects reached seen F@1cm below 0.35 while their recon->GT
  match distance was a healthy 9-15 mm. The geometry was right and the instance boundaries
  were wrong, and nothing in the pipeline reported that.

Method (ScanNet-style, on a common point set):
  The GT mesh is sampled into points carrying their GT instance id. Each GT point is then
  assigned to the nearest predicted instance within --thr, or to nothing. Both labellings
  now live on the same points, so IoU, PQ and the split counts are all well defined.

  Limitation, stated rather than hidden: predicted surface that sits where the GT mesh has
  no geometry cannot appear in any of these numbers, because the point set is GT's. That
  kind of error is what eval_seen_unseen.py's free-space violation measures; the two are
  complementary and neither replaces the other.

  No AP. Average precision needs a confidence score per predicted instance and the vote
  produces none. Reporting AP over an invented score would be a number that cannot be
  reproduced by anyone else, so this prints precision / recall / F1 at fixed IoU thresholds
  instead -- the same information without the fabricated ranking.

  python eval_instance_seg.py --root output/<scene>/objects_voted \\
      --gt_mesh ~/room_0/habitat/mesh_semantic.ply \\
      --gt_info ~/room_0/habitat/info_semantic.json
"""
import argparse
import collections
import json
import os
import re

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

STRUCTURE_WORDS = ("floor", "wall", "ceiling", "rug", "carpet", "window", "door",
                   "blind", "curtain", "beam", "pillar", "column", "stair")


def is_structure_class(name):
    """True when a GT class name IS a structure, by whole word (not substring: 'door'
    is inside 'indoor-plant')."""
    return any(t in STRUCTURE_WORDS for t in re.split(r"[^a-z]+", (name or "").lower()))


def load_xyz(path):
    v = PlyData.read(path)["vertex"]
    return np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64)


def gt_samples(mesh, info, n, seed=0):
    """Area-weighted GT points with their instance id, plus id -> class name."""
    p = PlyData.read(os.path.expanduser(mesh))
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)
    fe = p["face"]
    key = "vertex_indices" if "vertex_indices" in fe.data.dtype.names else "vertex_index"
    T, L = [], []
    for f, o in zip(fe[key], fe["object_id"]):
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1])); L.append(o)
    T = np.asarray(T, np.int64); L = np.asarray(L)
    e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
    a = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(T), n, p=a / a.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    over = (u + v) > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    P = V[T[i, 0]] + u * e1[i] + v * e2[i]

    names = {}
    if info and os.path.isfile(os.path.expanduser(info)):
        m = json.load(open(os.path.expanduser(info)))
        by = {int(c["id"]): str(c.get("name", "")).lower() for c in m.get("classes", [])}
        for o in m.get("objects", []):
            names[int(o["id"])] = str(o.get("class_name")
                                      or by.get(int(o.get("class_id", -1)), "")).lower()
    return P, L[i], names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="parent of the per-object dirs")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_info", default="")
    ap.add_argument("--only", default="", help="restrict to these gids, e.g. \"1 2 5\"")
    ap.add_argument("--n_gt", type=int, default=400000)
    ap.add_argument("--thr", type=float, default=0.05,
                    help="a GT point joins a predicted instance within this distance (m). "
                         "Matches audit_objects.py's GT-class test, so the two agree")
    ap.add_argument("--iou_thresholds", default="0.25,0.5")
    # A fragment is counted by its share of the thing it fragments, not by IoU: three
    # drawers each covering a third of one cabinet have IoU ~0.33 at best, so an IoU rule
    # would call them all misses and never say "one object, three labels".
    ap.add_argument("--frag_min", type=float, default=0.10,
                    help="share of an instance another one must cover to count as a piece "
                         "of it, for the over/under-segmentation counts")
    ap.add_argument("--min_gt_points", type=int, default=200,
                    help="drop GT instances with fewer samples than this: they are too "
                         "small to be detected at this sampling density, and counting them "
                         "as misses only measures the sampling rate")
    ap.add_argument("--all_gt", action="store_true",
                    help="also score structural GT instances (wall/floor/rug/...). Off by "
                         "default: the pipeline never tries to segment them, so including "
                         "them would report a recall the method never aimed at")
    ap.add_argument("--out", default="", help="default <root>/instance_seg.tsv")
    args = ap.parse_args()

    rd = os.path.expanduser(args.root)
    gids = sorted((g for g in os.listdir(rd) if g.isdigit()), key=int)
    if args.only:
        keep = set(args.only.split())
        gids = [g for g in gids if g in keep]
    assert gids, f"no object dir under {rd}"

    # --- predictions: one label per point, all instances in one array ---
    PX, PID = [], []
    for gid in gids:
        p = os.path.join(rd, gid, "point_cloud", f"iteration_{args.iter}", "point_cloud.ply")
        if not os.path.isfile(p):
            print(f"  [skip {gid}] no ply at iteration_{args.iter}")
            continue
        X = load_xyz(p)
        PX.append(X); PID.append(np.full(len(X), int(gid), np.int64))
    assert PX, "no predicted instances loaded"
    PX = np.concatenate(PX); PID = np.concatenate(PID)
    preds = sorted(set(PID.tolist()))

    G, GL, names = gt_samples(args.gt_mesh, args.gt_info, args.n_gt)

    # --- assign every GT point to the nearest predicted instance within thr ---
    d, j = cKDTree(PX).query(G, workers=-1)
    assigned = d < args.thr
    A = np.where(assigned, PID[j], -1)

    # --- GT instances worth scoring ---
    cnt_gt = collections.Counter(GL.tolist())
    gt_ids = [g for g, c in cnt_gt.items() if c >= args.min_gt_points]
    if not args.all_gt:
        gt_ids = [g for g in gt_ids if not is_structure_class(names.get(int(g), ""))]
    gt_ids = sorted(gt_ids)
    assert gt_ids, "no GT instance survived the filters -- check --gt_info and --min_gt_points"

    keep_gt = np.isin(GL, gt_ids)
    # |pred| counts only GT points inside the scored GT set, so a prediction is not punished
    # for also covering a wall we never asked it to find.
    n_pred = {p: int(((A == p) & keep_gt).sum()) for p in preds}
    inter = collections.defaultdict(int)
    for g, p in zip(GL[keep_gt], A[keep_gt]):
        if p >= 0:
            inter[(int(g), int(p))] += 1

    iou = {}
    for (g, p), n in inter.items():
        u = cnt_gt[g] + n_pred[p] - n
        if u > 0:
            iou[(g, p)] = n / u

    # --- per-GT coverage, per-pred purity, and the split counts ---
    cover = {g: (0.0, None) for g in gt_ids}
    pieces_of_gt = collections.defaultdict(list)
    for (g, p), v in iou.items():
        if v > cover[g][0]:
            cover[g] = (v, p)
    for (g, p), n in inter.items():
        if n / cnt_gt[g] >= args.frag_min:
            pieces_of_gt[g].append(p)

    purity = {p: (0.0, None) for p in preds}
    spans_of_pred = collections.defaultdict(list)
    for (g, p), v in iou.items():
        if v > purity[p][0]:
            purity[p] = (v, g)
    for (g, p), n in inter.items():
        if n_pred[p] and n / n_pred[p] >= args.frag_min:
            spans_of_pred[p].append(g)

    over = sorted(g for g in gt_ids if len(pieces_of_gt.get(g, [])) >= 2)
    under = sorted(p for p in preds if len(spans_of_pred.get(p, [])) >= 2)

    print(f"[seg] {len(gt_ids)} GT instances scored "
          f"({'all classes' if args.all_gt else 'structure classes excluded'}, "
          f">= {args.min_gt_points} samples), {len(preds)} predicted, thr {args.thr*100:.0f}cm")

    hdr = f"{'GT id':>7}{'class':>16}{'pts':>9}{'coverage':>10}{'best':>7}{'pieces':>8}  note"
    print("\n" + hdr); print("-" * len(hdr))
    for g in sorted(gt_ids, key=lambda x: -cnt_gt[x]):
        v, p = cover[g]
        pc = pieces_of_gt.get(g, [])
        note = ""
        if len(pc) >= 2:
            note = f"OVER-SEG: split across gids {sorted(pc)}"
        elif v == 0.0:
            note = "MISSED"
        print(f"{g:>7}{names.get(int(g), '?')[:15]:>16}{cnt_gt[g]:>9,}{v:>10.3f}"
              f"{('-' if p is None else p):>7}{len(pc):>8}  {note}")

    if under:
        print("\nunder-segmentation (one predicted instance spanning several GT instances):")
        for p in under:
            gs = ", ".join(f"{g} ({names.get(int(g), chr(63))})"
                           for g in sorted(spans_of_pred[p]))
            print(f"  gid {p}: covers {gs}")

    ths = [float(t) for t in args.iou_thresholds.split(",")]
    print(f"\n{'IoU thr':>9}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>9}"
          f"{'F1':>8}{'SQ':>8}{'PQ':>8}")
    rows_sum = []
    for t in ths:
        # Greedy one-to-one, highest IoU first: the standard matching, and it cannot count
        # one prediction as a hit for two GT instances.
        used_g, used_p, tp_iou = set(), set(), []
        for (g, p), v in sorted(iou.items(), key=lambda kv: -kv[1]):
            if v < t or g in used_g or p in used_p:
                continue
            used_g.add(g); used_p.add(p); tp_iou.append(v)
        tp = len(tp_iou); fp = len(preds) - tp; fn = len(gt_ids) - tp
        prec = tp / max(len(preds), 1); rec = tp / max(len(gt_ids), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        sq = float(np.mean(tp_iou)) if tp_iou else 0.0
        pq = sum(tp_iou) / max(tp + 0.5 * fp + 0.5 * fn, 1e-12)
        print(f"{t:>9.2f}{tp:>5}{fp:>5}{fn:>5}{prec:>11.3f}{rec:>9.3f}{f1:>8.3f}"
              f"{sq:>8.3f}{pq:>8.3f}")
        rows_sum.append((t, tp, fp, fn, prec, rec, f1, sq, pq))

    mcov = float(np.mean([cover[g][0] for g in gt_ids]))
    mpur = float(np.mean([purity[p][0] for p in preds]))
    print(f"\nmean coverage (GT side) {mcov:.3f}   mean purity (prediction side) {mpur:.3f}")
    print(f"over-segmented GT instances  {len(over)}/{len(gt_ids)}  {over}")
    print(f"under-segmented predictions  {len(under)}/{len(preds)}  {under}")
    print(f"missed GT instances          "
          f"{sum(1 for g in gt_ids if cover[g][0] == 0.0)}/{len(gt_ids)}")

    out = os.path.expanduser(args.out) if args.out else os.path.join(rd, "instance_seg.tsv")
    with open(out, "w") as f:
        f.write("gt_id\tclass\tgt_points\tcoverage_iou\tbest_gid\tn_pieces\tpieces\n")
        for g in sorted(gt_ids):
            v, p = cover[g]
            pc = sorted(pieces_of_gt.get(g, []))
            f.write(f"{g}\t{names.get(int(g), '')}\t{cnt_gt[g]}\t{v:.4f}\t"
                    f"{'' if p is None else p}\t{len(pc)}\t{' '.join(map(str, pc))}\n")
        f.write("\n# iou_thr\ttp\tfp\tfn\tprecision\trecall\tf1\tsq\tpq\n")
        for r in rows_sum:
            f.write("# " + "\t".join(f"{x:.4f}" if isinstance(x, float) else str(x)
                                     for x in r) + "\n")
    print(f"[seg] -> {out}")


if __name__ == "__main__":
    main()
