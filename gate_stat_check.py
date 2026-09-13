#!/usr/bin/env python3
"""Correlate candidate gate statistics with batch results, to pick a threshold.

Background:
  The original gate (min_unknown_frac) uses ufrac = the unknown fraction of the
  generated SURFACE shell. Counting alpha < 0.25 voxels inside a |SG| < 1.5 vox shell
  is noisy: three runs of the same command gave 38.4 / 38.8 / 40.0%. Objects near the
  threshold flip on that drift alone -- obj16 passed at 10.2% against a 10% threshold
  and then lost seen F@1 0.914 -> 0.704.
  The gate was once removed entirely; the batch then broke exactly the objects it used
  to block (obj16, obj8, obj10), so the DECISION was right and only the statistic was
  poor.

Candidate:
  generated-interior = (SG < 0) & (Wo == 0) & ~FREE & ~OTH, as a volume fraction. It is
  the volume the prior actually fills, so it is far more stable than a thin shell. Both
  numbers are already printed by every fusion run.

Usage:
  python gate_stat_check.py --logdir ~/prior/logs --csv <batch csv>

Output: per object (statistic, unseen F@2 change, seen F@1 change) and a confusion
matrix per threshold, for both candidate statistics.
"""
import argparse
import csv
import glob
import os
import re

# log lines are English now; the Korean patterns keep older logs readable
RE_INTERIOR = [r"generated-interior\s+([\d.]+)%", r"생성 ?내부\(미관측\)\s+([\d.]+)%"]
RE_UFRAC = [r"generated surface in unknown space\s+([\d.]+)%",
            r"생성 표면 중 unknown 비율\s+([\d.]+)%"]


def grab(text, patterns):
    """Last match of the first pattern that hits, as a float."""
    for p in patterns:
        m = re.findall(p, text)
        if m:
            return float(m[-1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.path.expanduser("~/prior/logs"))
    ap.add_argument("--csv", default=os.path.expanduser(
        "~/RefineGS/output/replica_room0_v2/refinegs_full/_field_batch.csv"))
    ap.add_argument("--noise", type=float, default=0.0066,
                    help="2-sigma band on unseen F@2; smaller changes count as no change")
    ap.add_argument("--stat", default="both", choices=["interior", "ufrac", "both"])
    ap.add_argument("--thresholds", default="0.02,0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--min_seen_a", type=float, default=0.30,
                    help="drop objects whose BASELINE is already broken")
    ap.add_argument("--max_seen_acc_b", type=float, default=20.0,
                    help="drop blow-ups (mm) so they cannot dominate the counts")
    args = ap.parse_args()

    stat, ufrac = {}, {}
    logs = glob.glob(os.path.join(args.logdir, "fuse_*.log"))
    for p in logs:
        g = os.path.basename(p)[5:-4]
        t = open(p, errors="ignore").read()
        v = grab(t, RE_INTERIOR)
        if v is not None:
            stat[g] = v
        v = grab(t, RE_UFRAC)
        if v is not None:
            ufrac[g] = v

    rows = list(csv.DictReader(open(os.path.expanduser(args.csv))))
    by = {}
    for r in rows:
        g = re.sub(r"\D", "", r.get("tag", ""))
        by.setdefault(g, {})["A" if r["mesh"].startswith("fuse_post") else "B"] = r

    rec, skipped = [], []
    for g, ab in sorted(by.items(), key=lambda x: int(x[0]) if x[0] else 0):
        if "A" not in ab or "B" not in ab:
            continue
        try:
            f2a, f2b = float(ab["A"]["unseen_F2.0"]), float(ab["B"]["unseen_F2.0"])
            s1a, s1b = float(ab["A"]["seen_F1.0"]), float(ab["B"]["seen_F1.0"])
            aca, acb = float(ab["A"]["seen_acc"]), float(ab["B"]["seen_acc"])
        except (KeyError, ValueError):
            continue
        if aca != aca or s1a < args.min_seen_a:
            continue
        if acb > args.max_seen_acc_b:
            continue
        if g not in stat and g not in ufrac:
            skipped.append(g)
            continue
        rec.append((g, stat.get(g, float("nan")), ufrac.get(g, float("nan")),
                    f2b - f2a, s1b - s1a))

    if not rec:
        raise SystemExit(
            f"no object matched.\n"
            f"  logs found      : {len(logs)} in {args.logdir}\n"
            f"  statistics read : interior {len(stat)}, ufrac {len(ufrac)}\n"
            f"  csv rows        : {len(rows)} from {args.csv}\n"
            f"Check the paths, and that the logs are from a run that printed "
            f"'[grid-fuse] observed ...' and '[gate] generated surface ...'.")
    if skipped:
        print(f"[warn] no log statistic for: {', '.join(skipped)}\n")

    print(f"{'obj':>6}{'interior%':>11}{'ufrac%':>9}{'d unsF2':>9}{'d seenF1':>10}  verdict")
    for g, s, u, df2, ds1 in sorted(rec, key=lambda x: (x[1] != x[1], x[1])):
        v = ("better" if df2 > args.noise else
             "worse" if df2 < -args.noise else "no change")
        print(f"{g:>6}{s:>11.2f}{u:>9.1f}{df2:>+9.3f}{ds1:>+10.3f}  {v}")

    cols = ([("interior", 1)] if args.stat == "interior" else
            [("ufrac", 2)] if args.stat == "ufrac" else
            [("interior", 1), ("ufrac", 2)])
    for name, idx in cols:
        print(f"\nconfusion matrix for '{name}' (prior applies when stat >= threshold)")
        print(f"{'thr%':>7}{'applied-ok':>12}{'wasted':>8}{'blocked-ok':>12}"
              f"{'missed':>8}{'net d unsF2':>13}")
        for t in [float(x) for x in args.thresholds.split(",")]:
            t = t * 100 if t <= 1 else t          # accept 0.10 or 10
            sel = [r for r in rec if r[idx] == r[idx]]
            tp = sum(1 for r in sel if r[idx] >= t and r[3] > args.noise)
            fp = sum(1 for r in sel if r[idx] >= t and r[3] < -args.noise)
            tn = sum(1 for r in sel if r[idx] < t and r[3] < -args.noise)
            fn = sum(1 for r in sel if r[idx] < t and r[3] > args.noise)
            net = sum(r[3] for r in sel if r[idx] >= t)
            print(f"{t:>7.1f}{tp:>12}{fp:>8}{tn:>12}{fn:>8}{net:>+13.3f}")

    print("\nA threshold with 'wasted' and 'missed' both 0 separates the objects perfectly.")
    print("Otherwise weigh the two errors: a regression (obj16 seen F@1 0.914 -> 0.704)")
    print("costs more than missing an improvement, so prefer the higher threshold.")
    print("Leave margin: the statistic drifts +-1.6%p between identical runs, so a")
    print("threshold within that of any object's value is a coin flip.")


if __name__ == "__main__":
    main()