#!/usr/bin/env python3
"""Where does a GT instance actually sit in the room -- and how far up.

Written to settle one question: both room0 and room1 lose exactly four `lamp` instances,
of near-identical size within each scene (room0 1515-1702 pts, room1 2604-2732), with no
predicted instance within 5 cm of any of them. Four identical fixtures vanishing together
is a configuration, not a model failure, and the two candidate configurations are told
apart by height:

  - high, near the ceiling -> either SAM3 called them something containing `ceiling`, which
    the relabel stage's --exclude_concepts then drops on purpose, or the trajectory barely
    sees them and MIN_AREA / MIN_TRACK drops them.
  - at table height, next to the lamps we DID find -> neither explanation holds and the
    loss is somewhere else entirely.

Heights are reported as a fraction of the room's own vertical extent, because Replica's
world frame is not gravity-aligned in any way we should assume: 0.0 is the lowest GT
surface in the scene, 1.0 the highest. The floor and ceiling rows are printed alongside so
the fraction can be read against something real rather than trusted on its own.

  python probe_gt_where.py \\
      --gt_mesh ~/replica_dl/room_1/habitat/mesh_semantic.ply \\
      --gt_info ~/replica_dl/room_1/habitat/info_semantic.json \\
      --klass lamp --ids 53,40,22,43
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from plyfile import PlyData


def load(mesh):
    """Per-object_id vertex positions, via the faces that carry the label."""
    p = PlyData.read(os.path.expanduser(mesh))
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)
    fe = p["face"]
    key = "vertex_indices" if "vertex_indices" in fe.data.dtype.names else "vertex_index"
    idx = defaultdict(set)
    for f, o in zip(fe[key], fe["object_id"]):
        idx[int(o)].update(int(i) for i in f)
    return V, {o: np.asarray(sorted(s), np.int64) for o, s in idx.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_info", required=True)
    ap.add_argument("--klass", default="", help="report every instance of this GT class")
    ap.add_argument("--ids", default="", help="comma-separated object_ids to report as well")
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"],
                    help="which axis is up. 'auto' picks the one the floor is flattest in")
    args = ap.parse_args()

    info = json.load(open(os.path.expanduser(args.gt_info)))
    by_cid = {int(c["id"]): str(c.get("name", "")).lower() for c in info.get("classes", [])}
    name_of = {int(o["id"]): str(o.get("class_name")
                                 or by_cid.get(int(o.get("class_id", -1)), "")).lower()
               for o in info.get("objects", [])}

    V, idx = load(args.gt_mesh)

    # Pick the up axis from the floor rather than assuming one: the floor is by definition
    # the surface with the smallest extent along up. Falling back to a guess here is how a
    # height argument ends up measuring depth.
    if args.up == "auto":
        floors = [o for o, n in name_of.items() if n == "floor" and o in idx]
        if floors:
            P = np.concatenate([V[idx[o]] for o in floors])
            up = int(np.argmin(P.max(0) - P.min(0)))
        else:
            up = 1
            print("[warn] no GT floor found; assuming axis 1 is up")
    else:
        up = {"x": 0, "y": 1, "z": 2}[args.up]
    axis_name = "xyz"[up]

    allP = np.concatenate([V[i] for i in idx.values()])
    lo, hi = float(allP[:, up].min()), float(allP[:, up].max())
    span = hi - lo if hi > lo else 1.0
    print(f"up axis = {axis_name}   scene extent {lo:.2f} .. {hi:.2f} m  (span {span:.2f} m)")

    want = set()
    if args.klass:
        want |= {o for o, n in name_of.items() if n == args.klass.lower()}
    want |= {int(t) for t in args.ids.replace(",", " ").split() if t.strip()}
    # The reference rows: whatever is at the very bottom and very top of the room.
    want |= {o for o, n in name_of.items() if n in ("floor", "ceiling")}

    rows = []
    for o in sorted(want):
        if o not in idx:
            rows.append((None, o, name_of.get(o, "?"), "not in mesh"))
            continue
        P = V[idx[o]]
        c = P.mean(0)
        f_lo = (P[:, up].min() - lo) / span
        f_hi = (P[:, up].max() - lo) / span
        f_c = (c[up] - lo) / span
        ext = P.max(0) - P.min(0)
        rows.append((f_c, o, name_of.get(o, "?"),
                     f"height {f_lo:.2f}-{f_hi:.2f} (mid {f_c:.2f})   "
                     f"size {ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f} m   "
                     f"centre ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})"))

    print(f"\n{'id':>6}  {'class':<14} detail")
    print("-" * 92)
    for f_c, o, nm, detail in sorted(rows, key=lambda r: (-1e9 if r[0] is None else -r[0])):
        print(f"{o:>6}  {nm:<14} {detail}")

    print("\nRead it as: mid near 1.0 = at the ceiling, near 0.0 = on the floor. If the "
          "missed instances sit high and the found ones do not, the loss is a ceiling "
          "fixture problem -- check --exclude_concepts first, MIN_AREA/MIN_TRACK second.")


if __name__ == "__main__":
    main()
