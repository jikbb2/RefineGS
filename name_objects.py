#!/usr/bin/env python3
"""Name each reconstructed object by majority vote against the GT semantic mesh.

Two uses:
  - logs and tables become readable: "obj6 table" instead of "obj6"
  - the result doubles as ShapeR captions. Without it every object is generated from
    "a 3D object in a room"; ShapeR takes a text condition, so a real class name is
    strictly more information.

Output is TSV with the caption in column 2, which is the format run_field_fusion_batch.sh
already reads via CAPTIONS:
    gid <TAB> a table <TAB> 0.95 <TAB> 11

Runs on the sliced gaussians by default, so it can be called right after extraction,
before any mesh exists.

  python name_objects.py --gt_mesh ~/room_0/habitat/mesh_semantic.ply \
      --gt_info ~/room_0/habitat/info_semantic.json --root OUT/objects_reg --iter 30000
"""
import argparse
import collections
import json
import os

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def load_names(path):
    if not path or not os.path.isfile(os.path.expanduser(path)):
        return {}
    m = json.load(open(os.path.expanduser(path)))
    by = {int(c["id"]): str(c.get("name", "")).lower()
          for c in m.get("classes", []) if isinstance(c, dict) and "id" in c}
    out = {}
    for o in m.get("objects", []):
        if isinstance(o, dict) and "id" in o:
            out[int(o["id"])] = str(o.get("class_name")
                                    or by.get(int(o.get("class_id", -1)), "")).lower()
    if not out and "id_to_label" in m:
        out = {i: by.get(int(c), "") for i, c in enumerate(m["id_to_label"])}
    return out


def gt_points(path, n, seed=0):
    """Area-weighted samples of the GT mesh, with a per-sample object_id."""
    p = PlyData.read(os.path.expanduser(path))
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)
    fe = p["face"]
    T, L = [], []
    for f, o in zip(fe["vertex_indices"], fe["object_id"]):
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1])); L.append(o)
    T = np.asarray(T, np.int64); L = np.asarray(L)
    e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
    a = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T), n, p=a / a.sum())
    u = rng.random((n, 1)); v = rng.random((n, 1))
    over = (u + v) > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    return V[T[idx, 0]] + u * e1[idx] + v * e2[idx], L[idx]


def obj_points(d, it, source):
    for p in ([os.path.join(d, "point_cloud", f"iteration_{it}", "point_cloud.ply")]
              if source == "ply" else
              [os.path.join(d, "train", f"ours_{it}", "fuse_post.ply")]):
        if os.path.isfile(p):
            v = PlyData.read(p)["vertex"]
            return np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_info", default="")
    ap.add_argument("--root", required=True, help="parent of the per-object dirs")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--source", default="ply", choices=["ply", "mesh"],
                    help="ply = sliced gaussians (available right after extraction)")
    ap.add_argument("--tau", type=float, default=0.05, help="match radius (m)")
    ap.add_argument("--n_gt", type=int, default=400000)
    ap.add_argument("--max_pts", type=int, default=20000)
    ap.add_argument("--out", default="", help="default <root>/names.tsv")
    args = ap.parse_args()

    G, GL = gt_points(args.gt_mesh, args.n_gt)
    names = load_names(args.gt_info)
    tree = cKDTree(G)
    rng = np.random.default_rng(0)

    rd = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out) if args.out else os.path.join(rd, "names.tsv")
    rows = []
    for gid in sorted((g for g in os.listdir(rd) if g.isdigit()), key=int):
        P = obj_points(os.path.join(rd, gid), args.iter, args.source)
        if P is None or not len(P):
            continue
        if len(P) > args.max_pts:
            P = P[rng.choice(len(P), args.max_pts, replace=False)]
        d, j = tree.query(P, workers=-1)
        ok = d < args.tau
        if ok.sum() < 20:
            rows.append((gid, "a 3D object in a room", 0.0, "")); continue
        cnt = collections.Counter(GL[j[ok]].tolist())
        tot = sum(cnt.values())
        top = cnt.most_common(4)
        nm = names.get(int(top[0][0]), "") or "3D object"
        share = top[0][1] / tot
        ids = ",".join(f"{i}:{c/tot*100:.0f}%" for i, c in top if c / tot >= 0.05)
        rows.append((gid, f"a {nm}".replace("_", " "), share, ids))

    with open(out, "w") as f:
        for gid, cap, share, ids in rows:
            f.write(f"{gid}\t{cap}\t{share:.2f}\t{ids}\n")

    print(f"{'gid':>5}  {'name':<20}{'share':>7}  gt ids")
    for gid, cap, share, ids in rows:
        flag = "  <- spans several GT objects" if 0 < share < 0.6 else ""
        print(f"{gid:>5}  {cap:<20}{share:>7.2f}  {ids}{flag}")
    print(f"\n[names] {len(rows)} objects -> {out}")
    print(f"use as ShapeR captions:  CAPTIONS={out} bash run_field_fusion_batch.sh")


if __name__ == "__main__":
    main()
