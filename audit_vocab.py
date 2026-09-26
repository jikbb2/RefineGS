#!/usr/bin/env python3
"""Find classes a scene contains that the SAM3 relabel vocabulary cannot name.

Why this exists: room1's dominant segmentation failure was not a model failure. The scene's
largest object was a bed, `bed` was absent from vocab.json, and every downstream number --
coverage 0.029, five of six under-segmentation cases, the depressed PQ -- followed from that
one omission. Nothing in the pipeline reports it: a class the vocabulary cannot name simply
never becomes an instance, so it looks like a recall miss rather than a missing word.

Running a scene before checking this risks spending the whole pipeline on a run that has to
be repeated. The check costs seconds.

  python audit_vocab.py \
      --info ~/replica_dl/room_2/habitat/info_semantic.json \
      --vocab ~/sam3/vocab.json

Several scenes at once (the label is just for the report):

  python audit_vocab.py --vocab ~/sam3/vocab.json \
      --info room2=~/replica_dl/room_2/habitat/info_semantic.json \
      --info office0=~/replica_dl/office_0/habitat/info_semantic.json

Exit status is 1 when any scene has a MISSING class, so it can gate a batch script.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

# The relabel stage's own exclusion list (run_refinegs.sh EXCLUDE). A class on this list is
# deliberately not segmented, so its absence from the vocabulary is correct, not a gap.
EXCLUDE_DEFAULT = ("door,blind,vent,window,wall,floor,ceiling,"
                   "light switch,thermostat")

# Classes that are structure or clutter rather than objects we would ever fuse. Reported
# separately so the MISSING list stays short enough to act on.
STRUCTURE_HINT = {
    "wall", "floor", "ceiling", "beam", "pillar", "column", "stair", "stairs",
    "wall-plug", "switch", "panel", "rug", "carpet", "curtain", "blinds",
    "ceiling-light", "wall-cabinet", "window", "door", "doorframe", "handrail",
}


def _singular(w):
    """Fold a trailing plural s. Replica writes `blinds`, our lists write `blind`."""
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _tokens(name):
    return [_singular(t) for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]


def _norm(name):
    """A comparable form: lowercase, separators unified, each token singularised."""
    return "-".join(_tokens(name))


def collect_strings(obj, out):
    """Pull every string out of a JSON structure of unknown shape.

    vocab.json's layout is not fixed across SAM3 setups -- it may be a flat list, a dict of
    lists, or objects carrying a `name` field. Reading every string is deliberately crude:
    over-reading only makes the audit more forgiving, and a false "present" is visible on the
    next run, while a false "missing" would send you editing a file that was already correct.
    """
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            collect_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_strings(v, out)


def load_vocab(path):
    with open(os.path.expanduser(path)) as fh:
        raw = json.load(fh)
    strings = set()
    collect_strings(raw, strings)
    return {_norm(s) for s in strings if s}, len(strings)


def load_scene(path):
    """Return {class_name: instance_count} for one Replica info_semantic.json."""
    with open(os.path.expanduser(path)) as fh:
        info = json.load(fh)

    id2name = {}
    for c in info.get("classes", []):
        if isinstance(c, dict) and "id" in c:
            id2name[int(c["id"])] = c.get("name", f"class{c['id']}")

    counts = Counter()
    objs = info.get("objects")
    if isinstance(objs, list) and objs:
        for o in objs:
            if not isinstance(o, dict):
                continue
            cid = o.get("class_id", o.get("classId"))
            if cid is None:
                continue
            counts[id2name.get(int(cid), f"class{cid}")] += 1
    else:
        # Fallback layout: id_to_label[object_id] = class_id, with 0/-1 meaning unlabelled.
        for cid in info.get("id_to_label", []):
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            if cid <= 0:
                continue
            counts[id2name.get(cid, f"class{cid}")] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info", action="append", required=True,
                    help="info_semantic.json path, or label=path. Repeatable.")
    ap.add_argument("--vocab", required=True, help="SAM3 vocab.json")
    ap.add_argument("--exclude", default=EXCLUDE_DEFAULT,
                    help="comma-separated concepts the relabel stage drops on purpose")
    ap.add_argument("--min_count", type=int, default=1,
                    help="ignore classes with fewer than this many instances in the scene")
    args = ap.parse_args()

    vocab, n_raw = load_vocab(args.vocab)
    excl = {_norm(w) for w in args.exclude.split(",") if w.strip()}
    print(f"vocab   {args.vocab}: {n_raw} strings -> {len(vocab)} normalised concepts")
    print(f"exclude {len(excl)} concepts: {', '.join(sorted(excl))}")

    any_missing = False
    for spec in args.info:
        label, _, path = spec.partition("=")
        if not path:
            path, label = label, os.path.basename(os.path.dirname(os.path.dirname(
                os.path.expanduser(label))))
        counts = load_scene(path)

        missing, structure, covered, excluded = [], [], 0, 0
        for name, n in counts.items():
            if n < args.min_count:
                continue
            key = _norm(name)
            # A class matches the vocabulary if its normalised form is there, or if any of
            # its tokens is -- `tv-screen` is named by `tv`, `office-chair` by `chair`.
            hit = key in vocab or any(t in vocab for t in _tokens(name))
            if key in excl or any(t in excl for t in _tokens(name)):
                excluded += 1
            elif hit:
                covered += 1
            elif key in STRUCTURE_HINT or any(t in STRUCTURE_HINT for t in _tokens(name)):
                structure.append((n, name))
            else:
                missing.append((n, name))

        print(f"\n=== {label or path}")
        print(f"    {sum(counts.values())} instances across {len(counts)} classes  "
              f"| covered {covered}  excluded {excluded}  "
              f"structure-like {len(structure)}  MISSING {len(missing)}")
        for n, name in sorted(missing, reverse=True):
            # Instance count is the whole point of the ordering: a missing class with one
            # instance costs one object, a missing class with forty changes the scene's score.
            print(f"    MISSING    {n:4d}  {name}")
            any_missing = True
        for n, name in sorted(structure, reverse=True)[:10]:
            print(f"    structure  {n:4d}  {name}")
        if len(structure) > 10:
            print(f"    ... and {len(structure) - 10} more structure-like")

    if any_missing:
        print("\nAdd the MISSING names above to vocab.json before running these scenes, "
              "or add them to EXCLUDE if they should not be segmented.")
        sys.exit(1)
    print("\nNo missing classes. The vocabulary names everything these scenes contain.")


if __name__ == "__main__":
    main()
