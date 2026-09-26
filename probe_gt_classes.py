#!/usr/bin/env python3
"""How much of a scene's GT surface carries a usable class name -- and how much does not.

Why this matters right now: the vocabulary audit reported 42 of office2's 91 GT instances
and 23 of office0's 64 with class_id -1, against 2 in room1. An object with class_id -1 has
no name in info_semantic.json, so `eval_instance_seg.py` resolves it to the empty string.
The empty string matches no entry in --exclude_classes, so the instance stays in the
denominator: a GT object that SAM3 can never be prompted for and that our vote can never
name still counts as a miss. If those instances carry real surface area, the office scenes'
recall and PQ are capped by the annotation, not by the method, and the cap is invisible in
the output.

Instance COUNT does not settle it. eval_instance_seg.py samples the mesh by area, so an
instance with no faces in mesh_semantic.ply, or with a sliver of them, barely moves the
denominator however many rows it occupies in the JSON. Area share is the number that decides
whether this is a real measurement problem or a bookkeeping curiosity.

  python probe_gt_classes.py \\
      --gt_mesh ~/replica_dl/office_2/habitat/mesh_semantic.ply \\
      --gt_info ~/replica_dl/office_2/habitat/info_semantic.json
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from plyfile import PlyData

# Kept byte-identical to eval_instance_seg.py's default so the two agree on the denominator.
EXCLUDE_DEFAULT = ("door,blind,vent,window,wall,floor,ceiling,light switch,thermostat,"
                   "rug,carpet,curtain,beam,pillar,column,stair")


def _singular(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def is_excluded_class(name, words):
    low = (name or "").lower()
    toks = [_singular(t) for t in re.split(r"[^a-z]+", low) if t]
    for w in words:
        if " " in w:
            if w in low:
                return True
        elif _singular(w) in toks:
            return True
    return False


def face_area_by_instance(mesh):
    """Total triangle area per object_id, exactly as eval_instance_seg.py weights its sample."""
    p = PlyData.read(os.path.expanduser(mesh))
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)
    fe = p["face"]
    key = "vertex_indices" if "vertex_indices" in fe.data.dtype.names else "vertex_index"
    area = defaultdict(float)
    for f, o in zip(fe[key], fe["object_id"]):
        for k in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[k]], V[f[k + 1]]
            area[int(o)] += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    return area


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_info", required=True)
    ap.add_argument("--exclude_classes", default=EXCLUDE_DEFAULT)
    ap.add_argument("--top", type=int, default=25, help="rows to print per section")
    args = ap.parse_args()

    info = json.load(open(os.path.expanduser(args.gt_info)))
    by_cid = {int(c["id"]): str(c.get("name", "")).lower()
              for c in info.get("classes", [])}
    # Resolved exactly the way eval_instance_seg.py resolves it, including the fallback to
    # "" -- reproducing the bug-prone line is the point, not working around it.
    name_of = {}
    for o in info.get("objects", []):
        name_of[int(o["id"])] = str(o.get("class_name")
                                    or by_cid.get(int(o.get("class_id", -1)), "")).lower()

    area = face_area_by_instance(args.gt_mesh)
    total = sum(area.values())
    excl = tuple(w.strip().lower() for w in args.exclude_classes.split(",") if w.strip())

    named_a = unnamed_a = excluded_a = orphan_a = 0.0
    named_n = unnamed_n = excluded_n = orphan_n = 0
    unnamed_rows, scored_rows = [], []

    for oid, a in area.items():
        if oid not in name_of:
            # In the mesh but not in the JSON at all: usually background triangles.
            orphan_a += a; orphan_n += 1
            continue
        nm = name_of[oid]
        if not nm:
            unnamed_a += a; unnamed_n += 1
            unnamed_rows.append((a, oid))
        elif is_excluded_class(nm, excl):
            excluded_a += a; excluded_n += 1
        else:
            named_a += a; named_n += 1
            scored_rows.append((a, oid, nm))

    def pct(x):
        return 100.0 * x / total if total else 0.0

    print(f"mesh  {args.gt_mesh}")
    print(f"total GT surface {total:.2f} m^2 across {len(area)} object_ids\n")
    print(f"  named & scored    {named_n:4d} inst  {named_a:8.2f} m^2  {pct(named_a):5.1f}%")
    print(f"  UNNAMED (cid -1)  {unnamed_n:4d} inst  {unnamed_a:8.2f} m^2  {pct(unnamed_a):5.1f}%")
    print(f"  excluded by class {excluded_n:4d} inst  {excluded_a:8.2f} m^2  {pct(excluded_a):5.1f}%")
    print(f"  not in info json  {orphan_n:4d} inst  {orphan_a:8.2f} m^2  {pct(orphan_a):5.1f}%")

    # The number that actually decides the ceiling: of the surface eval_instance_seg.py
    # scores, what fraction belongs to instances that cannot be named at all.
    denom = named_a + unnamed_a
    print(f"\n  scored denominator = named + UNNAMED = {denom:.2f} m^2")
    print(f"  UNNAMED share of the scored denominator: {100.0 * unnamed_a / denom if denom else 0:.1f}%")
    print(f"  => recall and PQ are capped at about "
          f"{100.0 * named_a / denom if denom else 0:.1f}% before the method does anything")

    print(f"\n  largest UNNAMED instances (object_id, m^2):")
    for a, oid in sorted(unnamed_rows, reverse=True)[:args.top]:
        print(f"    id {oid:5d}  {a:7.3f} m^2  {pct(a):5.2f}% of scene")

    print(f"\n  largest named & scored instances:")
    for a, oid, nm in sorted(scored_rows, reverse=True)[:args.top]:
        print(f"    id {oid:5d}  {a:7.3f} m^2  {pct(a):5.2f}%  {nm}")


if __name__ == "__main__":
    main()
