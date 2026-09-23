#!/usr/bin/env python3
"""Audit every extracted object before it reaches an evaluation table.

Two failures were silently contaminating results on room0, and neither shows up in a
geometry metric:

  structure   a label whose gaussians are the floor, a rug or a wall. Reported against a
              chair's GT it produces 1.4 m "unseen completion", and fed to ShapeR as
              "a chair" it produces a 600 mm surface error. Measured: gid2 -> floor 78%
              + rug 18%, gid5 -> floor 62% + wall 35%.

  mismatch    the per-gid MASK and the voted GAUSSIANS describe different things. gid2's
              mask back-projects to 0.64 m above the floor (a chair) while its extracted
              gaussians are a 2.4x3.4x0.13 m slab lying ON the floor. Everything
              downstream -- mesh, prior, metrics -- then describes the slab.

The mismatch test is the important one: it needs no ground truth, so it can run on a new
dataset, and it catches a wrong label assignment rather than a wrong reconstruction.
The GT class column is a control and is never used for the verdict.

  python audit_objects.py --root output/<scene>/objects_voted --iter 30000 \
      --masks_root data/<scene>/masks --colmap data/<scene>/sparse/0 \
      --carve_depth_dir output/<scene>/carve_depth \
      --gt_mesh ~/room_0/habitat/mesh_semantic.ply --gt_info ~/room_0/habitat/info_semantic.json
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from warp_gt_to_pose import read_colmap
except Exception:
    read_colmap = None

STRUCTURE_WORDS = ("floor", "wall", "ceiling", "rug", "carpet", "window", "door",
                   "blind", "curtain", "beam", "pillar", "column", "stair")


def is_structure_class(name):
    """True when a GT class name IS a structure, by whole word.

    Substring matching fires on 'door' inside 'indoor-plant', which is how room1's
    indoor-plant raised the structure warning on a perfectly ordinary object. Split the
    name into words instead: Replica class names are lowercase words joined by '-' or ' '.
    """
    return any(t in STRUCTURE_WORDS for t in re.split(r"[^a-z]+", (name or "").lower()))


def load_xyz(path):
    v = PlyData.read(path)["vertex"]
    return np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64)


def gt_samples(mesh, info, n, seed=0):
    """Area-weighted GT points with a per-sample class name. Control only."""
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


def mask_points(gid, stems, cams, masks_root, depth_dir, gt_depth_dir, scale, stride):
    """Back-project the object's own mask through the reference depth.

    This is what the instance ACTUALLY covers in the images, independent of which
    gaussians the vote ended up assigning to it.
    """
    out = []
    for st in stems[::stride]:
        p = os.path.join(masks_root, str(gid), "masks", st + ".png")
        if st not in cams or not os.path.isfile(p):
            continue
        a = np.array(Image.open(p))
        m = a[..., 3] if (a.ndim == 3 and a.shape[2] == 4) else (
            np.array(Image.open(p).convert("L")) if a.ndim == 3 else a)
        m = (m == 188) if (m == 188).any() else (m > 0 if m.max() <= 1 else m > 127)
        D = None
        if depth_dir:
            q = os.path.join(os.path.expanduser(depth_dir), st + ".npz")
            if os.path.isfile(q):
                D = np.load(q)["depth"].astype(np.float32)
        elif gt_depth_dir:
            for c in (st.replace("frame", "depth"), st):
                q = os.path.join(os.path.expanduser(gt_depth_dir), c + ".png")
                if os.path.isfile(q):
                    D = np.array(Image.open(q)).astype(np.float32) / scale
                    break
        if D is None:
            continue
        H, W = D.shape
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m.astype(np.uint8))
                         .resize((W, H), Image.NEAREST)) > 0
        sel = m & (D > 0.01)
        if not sel.any():
            continue
        c = cams[st]
        sx, sy = W / c["W"], H / c["H"]
        vv, uu = np.nonzero(sel)
        z = D[vv, uu]
        x = (uu / sx - c["cx"]) / c["fx"] * z
        y = (vv / sy - c["cy"]) / c["fy"] * z
        out.append((np.stack([x, y, z], 1) - c["t"]) @ c["R"])
    return np.concatenate(out) if out else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="parent of the per-object dirs")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--masks_root", default="")
    ap.add_argument("--colmap", default="")
    ap.add_argument("--carve_depth_dir", default="", help="dump_scene_depth.py npz (preferred)")
    ap.add_argument("--gt_depth_dir", default="")
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--gt_mesh", default="", help="control column only")
    ap.add_argument("--gt_info", default="")
    ap.add_argument("--up", default="z", choices=["x", "y", "z"])
    ap.add_argument("--view_stride", type=int, default=40)
    ap.add_argument("--n_gt", type=int, default=400000)
    # Structure thresholds. A slab lying on the floor over more than a square metre is not
    # an object one completes; neither is a thin panel two metres wide and a metre tall.
    ap.add_argument("--slab_thick", type=float, default=0.20, help="m")
    ap.add_argument("--slab_area", type=float, default=1.5, help="m^2")
    ap.add_argument("--floor_gap", type=float, default=0.15, help="m above the floor")
    ap.add_argument("--wall_span", type=float, default=2.0, help="m")
    ap.add_argument("--wall_thick", type=float, default=0.30, help="m")
    # The mask and the gaussians must describe the same thing. Measured on room0's break:
    # 0.55 m apart along the up axis, which no correct assignment produces.
    ap.add_argument("--max_centre_dist", type=float, default=0.40,
                    help="floor for the mismatch limit (m), for objects too small for the "
                         "relative test to mean anything")
    # An absolute limit is unfair to large objects: a 2.3 m sofa whose mask covers its
    # front face puts the two centres 44 cm apart, which is normal, while a 0.3 m book
    # 1.9 m from its mask is six times its own size. Scale with the object.
    ap.add_argument("--max_centre_frac", type=float, default=0.5,
                    help="mismatch limit as a fraction of the object's own diagonal")
    # A raw min/max box is set by its worst point. A handful of gaussians that the vote
    # dropped on the wrong object stretches a cushion to 2.8 m, and every size-based test
    # then reads the strays instead of the object. Measure the body, report the tail.
    ap.add_argument("--ext_pct", type=float, default=1.0,
                    help="percentile for the robust extent (0 = raw min/max)")
    ap.add_argument("--max_stray_ratio", type=float, default=1.6,
                    help="raw extent / robust extent above which the label is carrying "
                         "strays. extract_objects.py --min_margin / --min_votes / "
                         "--min_opacity exist for this and are off by default")
    # The ratio alone punishes thin objects: a 4 cm picture whose raw box is 7.6 cm reads
    # 1.9x, the same as a 0.4 m object with a stray a metre away. The ratio says how far
    # the tail reaches relative to the body, the length says whether it reaches anywhere
    # at all, and only both together mean a foreign object got voted in.
    ap.add_argument("--max_stray_m", type=float, default=0.15,
                    help="longest tail outside the robust box (m); flagged only when the "
                         "ratio is also above --max_stray_ratio")
    ap.add_argument("--max_pair_iou", type=float, default=0.30,
                    help="two labels whose robust boxes reach this IoU are one object "
                         "reported twice; both are flagged DUPLICATE. 0 = off")
    ap.add_argument("--out", default="", help="default <root>/audit.tsv")
    args = ap.parse_args()

    up = {"x": 0, "y": 1, "z": 2}[args.up]
    rd = os.path.expanduser(args.root)
    gids = sorted((g for g in os.listdir(rd) if g.isdigit()), key=int)
    assert gids, f"no object dir under {rd}"

    G = GL = names = None
    if args.gt_mesh:
        G, GL, names = gt_samples(args.gt_mesh, args.gt_info, args.n_gt)
        tree = cKDTree(G)

    cams, stems = {}, []
    if args.masks_root and args.colmap:
        assert read_colmap is not None, "warp_gt_to_pose import failed -- run from the repo root"
        cams = {c["stem"]: c for c in read_colmap(args.colmap)}
        stems = sorted(cams)

    # The floor is the lowest thing in the room; take it from all objects at once so a
    # single bad label cannot move it.
    allz = np.concatenate([load_xyz(os.path.join(rd, g, "point_cloud",
                                                 f"iteration_{args.iter}", "point_cloud.ply"))[:, up]
                           for g in gids
                           if os.path.isfile(os.path.join(rd, g, "point_cloud",
                                                          f"iteration_{args.iter}",
                                                          "point_cloud.ply"))])
    floor = float(np.percentile(allz, 1))
    print(f"[audit] {len(gids)} objects, up={args.up}, floor ~ {floor:+.2f} m")

    rows, boxes = [], {}
    for gid in gids:
        p = os.path.join(rd, gid, "point_cloud", f"iteration_{args.iter}", "point_cloud.ply")
        if not os.path.isfile(p):
            continue
        P = load_xyz(p)
        raw = P.max(0) - P.min(0)
        q = args.ext_pct
        lo, hi = (np.percentile(P, q, axis=0), np.percentile(P, 100 - q, axis=0)) \
            if q > 0 else (P.min(0), P.max(0))
        ext = np.maximum(hi - lo, 1e-6)
        stray = float(np.max(raw / ext))
        stray_m = float(np.max(raw - ext))
        n_out = int((~np.all((P >= lo) & (P <= hi), axis=1)).sum())
        horiz = [i for i in range(3) if i != up]
        area = float(ext[horiz[0]] * ext[horiz[1]])
        gap = float(lo[up] - floor)

        verdict, why = "object", []
        if stray > args.max_stray_ratio and stray_m > args.max_stray_m:
            verdict = "OUTLIERS"
            why.append(f"raw box {stray:.1f}x robust, tail {stray_m*100:.0f}cm, "
                       f"{n_out} pts outside")
        if ext[up] < args.slab_thick and area > args.slab_area and gap < args.floor_gap:
            verdict = "STRUCTURE"; why.append("slab on the floor")
        if (min(ext[horiz]) < args.wall_thick and max(ext[horiz]) > args.wall_span
                and ext[up] > 1.0):
            verdict = "STRUCTURE"; why.append("vertical panel")

        # mask vs gaussians -- no ground truth involved
        dctr = float("nan")
        if cams and args.masks_root:
            M = mask_points(gid, stems, cams, args.masks_root, args.carve_depth_dir,
                            args.gt_depth_dir, args.gt_depth_scale, args.view_stride)
            if len(M) > 50:
                dctr = float(np.linalg.norm(np.median(M, 0) - np.median(P, 0)))
                lim = max(args.max_centre_dist,
                          args.max_centre_frac * float(np.linalg.norm(ext)))
                if dctr > lim and verdict in ("object", "OUTLIERS"):
                    verdict = "MISMATCH"
                    why.append(f"mask {dctr*100:.0f}cm off, limit {lim*100:.0f}cm")

        cls, share = "", 0.0
        if G is not None:
            d, j = tree.query(P if len(P) <= 20000 else
                              P[np.random.default_rng(0).choice(len(P), 20000, False)],
                              workers=-1)
            ok = d < 0.05
            if ok.sum() > 20:
                cnt = collections.Counter(GL[j[ok]].tolist())
                tid, n = cnt.most_common(1)[0]
                cls = names.get(int(tid), str(tid)) if names else str(tid)
                share = n / sum(cnt.values())
        boxes[gid] = (lo, hi)
        rows.append([gid, len(P), ext, area, gap, dctr, cls, share, verdict,
                     ";".join(why), stray, stray_m])

    # One physical object split into two labels is invisible to every per-object test above:
    # each half is a plausible object on its own. It only shows up between objects. Measured
    # on room1, gid 0 and gid 3 are both "pillow" at 1.6 x 0.9 m, 29 cm apart, IoU 0.44 --
    # one bed reported twice. IoU (not containment) is the right test: a vase standing inside
    # a cabinet's box scores low because the volumes differ, while two halves of one object
    # score high because they overlap AND are the same size.
    if args.max_pair_iou > 0:
        idx = {r[0]: r for r in rows}
        for i, ga in enumerate(sorted(boxes, key=int)):
            for gb in sorted(boxes, key=int)[i + 1:]:
                (la, ha), (lb, hb) = boxes[ga], boxes[gb]
                inter = float(np.prod(np.maximum(np.minimum(ha, hb) - np.maximum(la, lb), 0)))
                if inter <= 0:
                    continue
                va, vb = float(np.prod(ha - la)), float(np.prod(hb - lb))
                iou = inter / max(va + vb - inter, 1e-12)
                if iou < args.max_pair_iou:
                    continue
                for g, other in ((ga, gb), (gb, ga)):
                    r = idx[g]
                    if r[8] == "object":
                        r[8] = "DUPLICATE"
                    note = f"overlaps gid {other} at IoU {iou:.2f}"
                    r[9] = f"{r[9]};{note}" if r[9] else note

    hdr = (f"{'gid':>5}{'gauss':>9}  {'extent p{:g} (m)'.format(args.ext_pct):<20}"
           f"{'raw/rob':>8}{'tail':>7}{'area':>7}{'floor':>7}"
           f"{'mask off':>10}  {'GT class (control)':<24}verdict")
    print("\n" + hdr); print("-" * len(hdr))
    for g, n, e, a, gap, dc, cls, sh, v, why, st, sm in rows:
        off = "--" if dc != dc else f"{dc * 100:.0f}cm"
        gt = f"{cls} {sh * 100:.0f}%" if cls else "-"
        print(f"{g:>5}{n:>9,}  {e[0]:.2f}x{e[1]:.2f}x{e[2]:<10.2f}{st:>7.1f}x"
              f"{sm * 100:>6.0f}c{a:>6.2f}m2"
              f"{gap:>+7.2f}{off:>10}  {gt:<24}{v}" + (f"  ({why})" if why else ""))

    bad = [r for r in rows if r[8] != "object"]
    print(f"\n[audit] {len(rows) - len(bad)} objects, {len(bad)} flagged "
          f"({sum(1 for r in bad if r[8] == 'STRUCTURE')} structure, "
          f"{sum(1 for r in bad if r[8] == 'MISMATCH')} mask/gaussian mismatch, "
          f"{sum(1 for r in bad if r[8] == 'OUTLIERS')} stray votes, "
          f"{sum(1 for r in bad if r[8] == 'DUPLICATE')} split across two labels)")
    if any(is_structure_class(r[6]) and r[8] == "object" for r in rows):
        print("  WARN a GT structure class appears on an object the geometry rule passed "
              "-- widen the rule rather than trusting the GT column, which is a control")

    out = os.path.expanduser(args.out) if args.out else os.path.join(rd, "audit.tsv")
    with open(out, "w") as f:
        f.write("gid\tgaussians\text_x\text_y\text_z\tarea\tfloor_gap\tmask_off\t"
                "gt_class\tgt_share\tverdict\twhy\tstray_ratio\tstray_m\n")
        for g, n, e, a, gap, dc, cls, sh, v, why, st, sm in rows:
            f.write(f"{g}\t{n}\t{e[0]:.3f}\t{e[1]:.3f}\t{e[2]:.3f}\t{a:.3f}\t{gap:.3f}\t"
                    f"{'' if dc != dc else f'{dc:.3f}'}\t{cls}\t{sh:.2f}\t{v}\t{why}\t"
                    f"{st:.2f}\t{sm:.3f}\n")
    keep = " ".join(r[0] for r in rows if r[8] == "object")
    print(f"[audit] -> {out}")
    print(f'[audit] evaluate only what passed:  ONLY="{keep}"')


if __name__ == "__main__":
    main()