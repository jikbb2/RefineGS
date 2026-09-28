#!/usr/bin/env python3
"""One A->B summary, computed the same way for every scene, from a run's results.csv.

Why this exists: room0's reported numbers came from a three-row table built by hand, while
room1 and room2 came from the per-object mean block the driver prints. Nobody checked that
the two agree, and a table comparing three scenes is worthless if one row was pooled and the
others averaged. This reads the CSV each run already writes and applies one rule to all of
them, so the only thing that differs between rows is the scene.

  python agg_ab.py output/replica_room2_v2/runs/0927_1517/results.csv
  python agg_ab.py output/replica_room0_v2/runs/0921_1022/results.csv --only "1 2 3 4"

A and B are told apart by the mesh path, not by row order: side A is the observation-only
`fuse_post.ply`, side B the `fused_<RUN>_post.ply` the fusion wrote. A row whose mesh matches
neither is reported and skipped rather than silently averaged in -- the ShapeR-alone runs
wrote their own meshes into CSVs of the same shape.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# (csv column, printed name, unit scale, higher_is_better)
METRICS = [
    ("seen_acc", "seen accuracy(mm)", 1.0, False),
    ("seen_F1.0", "seen F@1cm", 1.0, True),
    ("unseen_comp", "unseen completion(mm)", 1.0, False),
    ("unseen_F2.0", "unseen F@2cm", 1.0, True),
    ("free_pct", "free violation(%)", 1.0, False),
]


def side_of(mesh):
    """A = observation only, B = fused. Decided by filename, never by row order."""
    b = os.path.basename(mesh or "")
    if b.startswith("fused_"):
        return "B"
    if b.startswith("fuse_post") or b.startswith("fuse."):
        return "A"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--only", default="",
                    help="restrict to these gids (space or comma separated), matching the "
                         "audit's ONLY list. Default: every object present in the CSV")
    ap.add_argument("--label", default="", help="name to print in the header")
    args = ap.parse_args()

    keep = {t.strip() for t in args.only.replace(",", " ").split() if t.strip()}

    rows = defaultdict(dict)
    unknown, seen_cols = [], None
    with open(os.path.expanduser(args.csv_path)) as fh:
        for r in csv.DictReader(fh):
            seen_cols = r.keys()
            s = side_of(r.get("mesh", ""))
            if s is None:
                unknown.append(os.path.basename(r.get("mesh", "?")))
                continue
            tag = (r.get("tag") or "").strip()
            gid = tag[3:] if tag.startswith("obj") else tag
            if keep and gid not in keep:
                continue
            rows[tag][s] = r

    missing = [c for c, _, _, _ in METRICS if seen_cols and c not in seen_cols]
    if missing:
        sys.exit(f"[abort] columns not in this CSV: {', '.join(missing)}\n"
                 f"        present: {', '.join(seen_cols)}")

    paired = {t: v for t, v in rows.items() if "A" in v and "B" in v}
    lonely = sorted(t for t, v in rows.items() if len(v) < 2)
    if not paired:
        sys.exit("[abort] no object has both an A and a B row -- check the mesh column")

    # A passthrough is an object the gate declined: B is byte-for-byte A, so every metric is
    # unchanged. Counting them matters because they pull every mean toward zero change, and
    # a reader comparing scenes with different passthrough counts is comparing two things.
    def vals(r):
        return [float(r[c]) for c, _, _, _ in METRICS]

    through = [t for t, v in paired.items()
               if all(abs(a - b) < 1e-9 for a, b in zip(vals(v["A"]), vals(v["B"])))]

    name = args.label or os.path.basename(os.path.dirname(args.csv_path))
    print(f"=== {name}   {len(paired)} objects paired"
          f"   ({len(through)} passthrough, {len(paired) - len(through)} fused) ===")
    if lonely:
        print(f"    unpaired (one side only, skipped): {', '.join(lonely)}")
    if unknown:
        print(f"    rows with an unrecognised mesh, skipped: {len(unknown)}"
              f"  e.g. {unknown[0]}")

    def block(tags, title):
        if not tags:
            return
        print(f"\n  --- {title}: {len(tags)} objects ---")
        for col, label, scale, higher in METRICS:
            a = sum(float(paired[t]["A"][col]) for t in tags) / len(tags) * scale
            b = sum(float(paired[t]["B"][col]) for t in tags) / len(tags) * scale
            d = b - a
            good = (d > 0) if higher else (d < 0)
            mark = "" if abs(d) < 1e-9 else ("  better" if good else "  worse")
            print(f"    {label:>24}: {a:9.3f} -> {b:9.3f}  ({d:+.3f}){mark}")
        # Win/loss on the headline metric, which a mean cannot show: one object moving a long
        # way looks the same as every object moving a little, and they are different claims.
        wins = sum(1 for t in tags
                   if float(paired[t]["B"]["unseen_F2.0"]) > float(paired[t]["A"]["unseen_F2.0"]))
        ties = sum(1 for t in tags
                   if float(paired[t]["B"]["unseen_F2.0"]) == float(paired[t]["A"]["unseen_F2.0"]))
        print(f"    {'unseen F@2 win/tie/loss':>24}: {wins}/{ties}/{len(tags) - wins - ties}")

    all_tags = sorted(paired, key=lambda t: (len(t), t))
    block(all_tags, "all objects (the number for the paper)")
    block([t for t in all_tags if t not in set(through)],
          "fused only (secondary -- the denominator is chosen after the fact)")


if __name__ == "__main__":
    main()
