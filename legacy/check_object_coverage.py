#!/usr/bin/env python3
"""For each GT object: does the scene model represent it, and did voting assign it?

Small objects come out of extraction with 0 gaussians, and there are two very different
reasons for that:
  represented but not voted  -> vote_labels.py is the problem, and it is cheap to fix
                                (the visibility test needs |z - d_gt| < margin in >= 1 view)
  not represented at all     -> the scene model never reconstructed it; only training or
                                resolution can help, which is expensive

'near' counts scene gaussians within --tau of the object's GT surface. That over-counts on
a cluttered surface (a book on a table picks up table gaussians), so read it together with
'voted'.

  python check_object_coverage.py --ply SCENE/point_cloud/iteration_30000/point_cloud.ply \
      --labels SCENE/vote/labels.npy --id_map DATA/labels_scene/id_map.json \
      --gt_mesh ~/room_0/habitat/mesh_semantic.ply --gt_info ~/room_0/habitat/info_semantic.json
"""
import argparse
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


def gt_samples(path, n, seed=0):
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
    u, v = rng.random((n, 1)), rng.random((n, 1))
    over = (u + v) > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    return V[T[idx, 0]] + u * e1[idx] + v * e2[idx], L[idx], a, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="scene point_cloud.ply")
    ap.add_argument("--labels", default="", help="vote/labels.npy")
    ap.add_argument("--id_map", default="", help="labels_scene/id_map.json")
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_info", default="")
    ap.add_argument("--tau", type=float, default=0.02, help="gaussian-to-GT distance (m)")
    ap.add_argument("--n_sample", type=int, default=400000)
    ap.add_argument("--min_pts", type=int, default=500,
                    help="extract_objects.py's threshold, for the verdict column")
    args = ap.parse_args()

    v = PlyData.read(os.path.expanduser(args.ply))["vertex"]
    G = np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float64)
    P, PL, area, L_all = gt_samples(args.gt_mesh, args.n_sample)
    names = load_names(args.gt_info)
    print(f"[scene] {len(G):,} gaussians   [gt] {len(np.unique(L_all))} objects")

    voted = {}
    if args.labels and args.id_map:
        lab = np.load(os.path.expanduser(args.labels))
        m = json.load(open(os.path.expanduser(args.id_map)))
        gid_of = {int(x): int(g) for g, x in m["label_of_gid"].items()}
        for c in range(int(lab.max()) + 1):
            if c in gid_of:
                voted[gid_of[c]] = int((lab == c).sum())

    tree = cKDTree(G)
    rows = []
    for o in sorted(np.unique(PL)):
        q = P[PL == o]
        if not len(q):
            continue
        near = len(np.unique(np.concatenate(tree.query_ball_point(q, args.tau)))) \
            if len(q) else 0
        rows.append((int(o), names.get(int(o), ""), len(q), near, voted.get(int(o))))

    print(f"\n{'oid':>5}  {'class':<16}{'gt pts':>8}{'near':>9}{'voted':>8}  verdict")
    for o, nm, npts, near, vt in sorted(rows, key=lambda r: -r[3]):
        if near < args.min_pts:
            vd = "NOT REPRESENTED by the scene model"
        elif vt is None:
            vd = ""
        elif vt < args.min_pts:
            vd = "represented but NOT VOTED  <- fix voting"
        else:
            vd = "ok"
        print(f"{o:>5}  {nm or 'undefined':<16}{npts:>8}{near:>9}"
              f"{(vt if vt is not None else -1):>8}  {vd}")
    nr = sum(1 for r in rows if r[3] < args.min_pts)
    nv = sum(1 for r in rows if r[3] >= args.min_pts and r[4] is not None
             and r[4] < args.min_pts)
    print(f"\nnot represented {nr}   represented but not voted {nv}   "
          f"(threshold {args.min_pts} gaussians)")
    print("The second group is a voting problem and cheap to fix; the first needs a better"
          " reconstruction.")


if __name__ == "__main__":
    main()
