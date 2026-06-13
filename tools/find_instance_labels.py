#!/usr/bin/env python3
"""
Find RefineGS label → GT furniture id correspondence.

Usage (run from /home/elicer/RefineGS):
    python tools/find_instance_labels.py
    python tools/find_instance_labels.py --recon_glob "output/replica_room0/raw_graph_reg/*/train/ours_7000/fuse_post.ply"
    python tools/find_instance_labels.py --dist_thresh 0.3   # tighter threshold

Prints:
  - Best matching recon label for each target GT furniture id
  - All recon labels sorted by nearest GT candidate (so you can spot doubles)
"""
import argparse
import glob
import os
import struct
import sys
import numpy as np


# ── target GT instances (from list-gt on Replica room0) ─────────────────────
TARGET_GT = {
    7:  ("table",  np.array([2.25, -0.71, -0.91])),
    9:  ("sofa",   np.array([3.67, -0.52, -1.05])),
    11: ("table",  np.array([3.71,  1.03, -1.18])),   # large coffee table
    27: ("table",  np.array([2.25, -0.39, -1.00])),
    39: ("stool",  np.array([1.82,  1.53, -1.21])),
    41: ("stool",  np.array([1.77,  0.65, -1.22])),
    73: ("chair",  np.array([5.77,  1.49, -0.95])),
    74: ("chair",  np.array([5.76,  0.49, -0.95])),
    77: ("sofa",   np.array([3.80,  2.72, -1.02])),
    93: ("table",  np.array([5.19,  2.76, -0.91])),
}

# ── PLY centroid reader (open3d, fallback pure-numpy) ────────────────────────

def _read_centroid_o3d(path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(path)
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return None
    return v.mean(axis=0)


def _read_centroid_numpy(path):
    """Pure-numpy binary/ascii PLY vertex reader (x,y,z only)."""
    with open(path, "rb") as f:
        raw = f.read()

    # parse header
    header_end = raw.find(b"end_header\n")
    if header_end == -1:
        header_end = raw.find(b"end_header\r\n")
        if header_end == -1:
            return None
        data_start = header_end + len(b"end_header\r\n")
    else:
        data_start = header_end + len(b"end_header\n")

    header = raw[:header_end].decode("ascii", errors="replace")
    lines = header.splitlines()

    fmt = "binary_little_endian"
    n_verts = 0
    props = []
    in_vertex = False
    for line in lines:
        tok = line.strip().split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            in_vertex = (tok[1] == "vertex")
            if in_vertex:
                n_verts = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            props.append((tok[1], tok[2]))  # (type, name)

    if n_verts == 0:
        return None

    # find x y z indices and their byte offsets
    type_sizes = {"float": 4, "double": 8, "int": 4, "uint": 4,
                  "short": 2, "ushort": 2, "char": 1, "uchar": 1,
                  "float32": 4, "float64": 8, "int32": 4, "uint8": 1,
                  "int8": 1, "uint32": 4, "int16": 2, "uint16": 2}
    stride = sum(type_sizes.get(t, 4) for t, _ in props)
    xyz_offsets = []
    offset = 0
    for t, name in props:
        if name in ("x", "y", "z"):
            xyz_offsets.append((name, offset, t))
        offset += type_sizes.get(t, 4)

    if len(xyz_offsets) < 3:
        return None

    if fmt == "ascii":
        name_to_col = {p[1]: i for i, p in enumerate(props)}
        xi, yi, zi = name_to_col["x"], name_to_col["y"], name_to_col["z"]
        text = raw[data_start:].decode("ascii", errors="replace")
        vlines = text.splitlines()[:n_verts]
        verts = []
        for vl in vlines:
            vs = vl.split()
            if len(vs) > max(xi, yi, zi):
                verts.append([float(vs[xi]), float(vs[yi]), float(vs[zi])])
        if not verts:
            return None
        return np.array(verts).mean(axis=0)
    else:
        # binary
        data = raw[data_start: data_start + stride * n_verts]
        if len(data) < stride * n_verts:
            return None
        buf = np.frombuffer(data, dtype=np.uint8).reshape(n_verts, stride)
        coords = []
        for name, off, t in xyz_offsets:
            dt = np.float32 if t in ("float", "float32") else np.float64
            sz = 4 if dt == np.float32 else 8
            col = np.frombuffer(buf[:, off:off+sz].tobytes(), dtype=dt)
            coords.append(col.astype(np.float64))
        return np.stack(coords, axis=1).mean(axis=0)


def read_centroid(path):
    try:
        return _read_centroid_o3d(path)
    except Exception:
        pass
    try:
        return _read_centroid_numpy(path)
    except Exception:
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_glob", default=
        "output/replica_room0/raw_graph_reg/*/train/ours_7000/fuse_post.ply")
    ap.add_argument("--dist_thresh", type=float, default=0.5,
        help="Max centroid distance (m) to consider a valid match")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.recon_glob))
    if not paths:
        print(f"No files found: {args.recon_glob}")
        sys.exit(1)
    print(f"Scanning {len(paths)} recon meshes …")

    # compute centroid for each recon
    recon = {}   # label -> centroid
    for path in paths:
        parts = path.replace("\\", "/").split("/")
        # label is the numeric directory just before 'train'
        try:
            train_idx = parts.index("train")
            label = parts[train_idx - 1]
        except ValueError:
            label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        c = read_centroid(path)
        if c is not None:
            recon[label] = c
        else:
            print(f"  [warn] could not read centroid: {path}")

    print(f"Read {len(recon)} centroids.\n")

    # ── Table 1: best recon label per target GT ──────────────────────────────
    print("=" * 60)
    print("TARGET GT  →  best matching recon label")
    print("=" * 60)
    print(f"{'GT id':>6}  {'class':>8}  {'label':>7}  {'dist(m)':>9}  note")
    print("-" * 60)

    gt_to_label = {}
    for gt_id in sorted(TARGET_GT):
        cls, gt_c = TARGET_GT[gt_id]
        best_label, best_dist = None, 1e9
        for label, c in recon.items():
            d = np.linalg.norm(c - gt_c)
            if d < best_dist:
                best_dist = d
                best_label = label
        flag = "" if best_dist <= args.dist_thresh else "  ← dist > thresh, suspect"
        print(f"{gt_id:>6}  {cls:>8}  {best_label:>7}  {best_dist:>9.3f}{flag}")
        gt_to_label[gt_id] = (best_label, best_dist)

    # ── Table 2: every recon label with its nearest GT candidate ─────────────
    print("\n" + "=" * 60)
    print("ALL recon labels  →  nearest GT candidate")
    print("=" * 60)
    print(f"{'label':>7}  {'centroid':>30}  {'GT id':>6}  {'class':>8}  {'dist(m)':>9}")
    print("-" * 60)

    rows = []
    for label, c in recon.items():
        best_gt, best_dist = None, 1e9
        for gt_id, (cls, gt_c) in TARGET_GT.items():
            d = np.linalg.norm(c - gt_c)
            if d < best_dist:
                best_dist = d
                best_gt = gt_id
        rows.append((label, c, best_gt, best_dist))

    rows.sort(key=lambda r: r[3])  # sort by dist
    for label, c, best_gt, best_dist in rows:
        cls = TARGET_GT[best_gt][0]
        cent_str = f"({c[0]:5.2f},{c[1]:5.2f},{c[2]:5.2f})"
        marker = " ◀" if best_dist <= args.dist_thresh else ""
        print(f"{label:>7}  {cent_str:>30}  {best_gt:>6}  {cls:>8}  {best_dist:>9.3f}{marker}")

    # ── Recommendation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMMENDED eval instances (dist < thresh, diverse):")
    print("=" * 60)
    seen_cls = set()
    picks = []
    for gt_id in [73, 74, 9, 7, 27, 77, 11, 39, 41, 93]:
        if gt_id not in gt_to_label:
            continue
        label, dist = gt_to_label[gt_id]
        cls = TARGET_GT[gt_id][0]
        if dist <= args.dist_thresh and cls not in seen_cls:
            picks.append((gt_id, cls, label, dist))
            seen_cls.add(cls)
        if len(picks) >= 3:
            break

    for gt_id, cls, label, dist in picks:
        print(f"  GT {gt_id:>2} ({cls:<6}) ← recon label {label}  dist={dist:.3f}m")

    if picks:
        labels_str = " ".join(p[2] for p in picks)
        gt_map_lines = "\n".join(f"{p[2]},{p[0]}" for p in picks)
        print(f"\n# Run sweep with:")
        print(f'  INSTANCES="{labels_str}" bash run_axis3_reg_sweep.sh')
        print(f"\n# Or pass GT map directly (label,gt_id CSV):")
        print(f'  GT_MAP=gt_map.csv INSTANCES="{labels_str}" bash run_axis3_reg_sweep.sh')
        print(f"\n# gt_map.csv content:")
        print(gt_map_lines)


if __name__ == "__main__":
    main()
