#!/usr/bin/env python3
"""RefineGS Axis 3 — per-object geometry evaluation harness.

Compares reconstructed object meshes (2DGS fuse_post.ply) against the
Replica GT semantic mesh (habitat/mesh_semantic.ply, per-face object_id).

Metrics (2DGS/DTU-style):
  - accuracy   : mean dist recon->GT          (lower better)
  - completion : mean dist GT->recon          (lower better)
  - chamfer_l1 : (accuracy + completion) / 2  (lower better)
  - f@tau      : F-score at threshold(s) tau  (higher better)
  - normal_consistency : mean |cos| of matched normals (higher better)

Coordinate frames are assumed aligned (replica_to_refinegs.py uses GT poses
and metric scale). Optional --icp does a point-to-point ICP refinement.

NOTE (protocol): distances are point-to-point between surface samples, so
there is a sampling floor ~0.5*sqrt(area/n_samples) (e.g. ~1.5mm for a 1m^2
object at n=100k). Keep n_samples fixed across variants (reg OFF/ON/strong,
3DGS vs 2DGS) so the floor cancels in comparisons; raise it for small
absolute numbers.

Subcommands:
  list-gt     List GT instances (id, class, #faces, bbox center/extent).
  extract-gt  Save one GT instance mesh to .ply (also used as cache).
  eval        Evaluate one recon mesh vs one GT instance (--gt_id or --auto_match).
  batch       Evaluate many recon meshes (glob) -> CSV (+ optional label->gt_id map).

Examples:
  python tools/eval_object_mesh.py list-gt --gt_mesh habitat/mesh_semantic.ply \
      --info habitat/info_semantic.json

  python tools/eval_object_mesh.py eval \
      --gt_mesh habitat/mesh_semantic.ply --gt_id 12 \
      --recon output/replica_room0/raw_graph_reg/12/train/ours_7000/fuse_post.ply

  python tools/eval_object_mesh.py batch \
      --gt_mesh habitat/mesh_semantic.ply \
      --recon_glob 'output/replica_room0/raw_graph_reg/*/train/ours_7000/fuse_post.ply' \
      --label_from_path -4 --auto_match --out results_reg.csv

Deps: numpy, scipy, trimesh, plyfile  (no open3d needed; headless-safe).
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree

DEFAULT_TAUS = (0.005, 0.01, 0.02)  # meters: 5mm / 1cm / 2cm
METRIC_KEYS_BASE = ["accuracy", "completion", "chamfer_l1", "normal_consistency",
                    "n_pts_recon", "n_pts_gt",
                    # visibility-split metrics (require --colmap_dir/--masks_dir)
                    "accuracy_obs", "accuracy_unobs",
                    "completion_obs", "completion_unobs",
                    "frac_recon_obs", "frac_gt_obs"]


# ---------------------------------------------------------------------------
# GT semantic mesh loading
# ---------------------------------------------------------------------------

def load_semantic_mesh(path):
    """Parse Replica habitat mesh_semantic.ply.

    Returns (vertices [V,3] float64, tri_faces [F,3] int64, tri_object_ids [F] int64).
    Faces may be quads -> fan-triangulated; object_id replicated per triangle.
    """
    ply = PlyData.read(path)
    v = ply["vertex"]
    vertices = np.column_stack([np.asarray(v["x"]), np.asarray(v["y"]),
                                np.asarray(v["z"])]).astype(np.float64)

    face_el = ply["face"]
    names = face_el.data.dtype.names
    idx_key = "vertex_indices" if "vertex_indices" in names else "vertex_index"
    if "object_id" not in names:
        raise ValueError(f"'{path}' has no per-face object_id "
                         f"(face properties: {names}). Use habitat/mesh_semantic.ply.")
    raw_faces = face_el.data[idx_key]
    obj_ids = np.asarray(face_el.data["object_id"]).astype(np.int64)

    # Fast path: uniform vertex count per face
    lens = np.fromiter((len(f) for f in raw_faces), dtype=np.int64,
                       count=len(raw_faces))
    tris, tri_ids = [], []
    if (lens == 3).all():
        tris = np.vstack(raw_faces).astype(np.int64)
        tri_ids = obj_ids
    elif (lens == 4).all():
        quads = np.vstack(raw_faces).astype(np.int64)
        tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)
        tri_ids = np.concatenate([obj_ids, obj_ids], axis=0)
    else:  # mixed polygon sizes: generic fan triangulation
        t_list, id_list = [], []
        for f, oid in zip(raw_faces, obj_ids):
            f = np.asarray(f, dtype=np.int64)
            for k in range(1, len(f) - 1):
                t_list.append((f[0], f[k], f[k + 1]))
                id_list.append(oid)
        tris = np.asarray(t_list, dtype=np.int64)
        tri_ids = np.asarray(id_list, dtype=np.int64)

    return vertices, tris, tri_ids


def load_class_names(info_json):
    """info_semantic.json -> {object_id: class_name} (best effort)."""
    if not info_json or not os.path.isfile(info_json):
        return {}
    with open(info_json) as f:
        info = json.load(f)
    mapping = {}
    for obj in info.get("objects", []):
        oid = obj.get("id")
        cls = obj.get("class_name", obj.get("class", ""))
        if oid is not None:
            mapping[int(oid)] = str(cls)
    return mapping


def extract_instance_mesh(vertices, tris, tri_ids, object_id):
    """Sub-mesh of triangles with the given object_id, as trimesh.Trimesh."""
    sel = tris[tri_ids == object_id]
    if len(sel) == 0:
        raise ValueError(f"object_id {object_id} has no faces in GT mesh")
    uniq, inv = np.unique(sel.reshape(-1), return_inverse=True)
    return trimesh.Trimesh(vertices=vertices[uniq],
                           faces=inv.reshape(-1, 3), process=False)


# ---------------------------------------------------------------------------
# Visibility classification (observed vs unobserved surface regions)
#
# A sample point is OBSERVED if, in >= min_views training views:
#   (1) it projects inside the image with z > 0,
#   (2) the pixel lies inside that view's instance mask
#       (masks already encode all occlusion, incl. by other objects),
#   (3) it is front-facing (normal . (cam_center - X) > 0; self-occlusion proxy).
# accuracy on UNOBSERVED recon points = hallucination metric.
#
# CAVEAT: the front-facing test assumes outward-consistent face normals on
# both meshes (TSDF/marching-cubes output satisfies this). Inconsistent
# winding inflates frac_*_obs.
# ---------------------------------------------------------------------------

def _qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]])


def _read_colmap_cameras_txt(path):
    cams = {}
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            cid, model, w, h = int(t[0]), t[1], int(t[2]), int(t[3])
            p = list(map(float, t[4:]))
            if model == "PINHOLE":
                fx, fy, cx, cy = p[:4]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                raise ValueError(f"unsupported camera model {model}")
            cams[cid] = (fx, fy, cx, cy, w, h)
    return cams


def _read_colmap_images_txt(path):
    out = []
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for i in range(0, len(lines), 2):  # every other line is 2D points
        t = lines[i].split()
        if len(t) < 10:
            continue
        q = list(map(float, t[1:5])); tv = np.array(list(map(float, t[5:8])))
        out.append({"R": _qvec2rot(q), "t": tv, "camera_id": int(t[8]),
                    "name": t[9]})
    return out


def _read_colmap_bin(sparse_dir):
    import struct
    cams = {}
    with open(os.path.join(sparse_dir, "cameras.bin"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        model_params = {0: 3, 1: 4, 2: 4, 3: 5}  # SIMPLE_PINHOLE,PINHOLE,SIMPLE_RADIAL,RADIAL
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            np_ = model_params.get(model)
            if np_ is None:
                raise ValueError(f"unsupported camera model id {model}")
            p = struct.unpack(f"<{np_}d", f.read(8*np_))
            if model == 1:
                fx, fy, cx, cy = p[:4]
            else:
                fx = fy = p[0]; cx, cy = p[1], p[2]
            cams[cid] = (fx, fy, cx, cy, int(w), int(h))
    images = []
    with open(os.path.join(sparse_dir, "images.bin"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _iid = struct.unpack("<I", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32))
            tv = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n2d)
            images.append({"R": _qvec2rot(q), "t": tv, "camera_id": cid,
                           "name": name.decode()})
    return cams, images


def load_instance_cameras(sparse_dir):
    """Read COLMAP sparse model (bin or txt). Returns list of camera dicts."""
    if os.path.isfile(os.path.join(sparse_dir, "images.bin")):
        cams, images = _read_colmap_bin(sparse_dir)
    else:
        cams = _read_colmap_cameras_txt(os.path.join(sparse_dir, "cameras.txt"))
        images = _read_colmap_images_txt(os.path.join(sparse_dir, "images.txt"))
    out = []
    for im in images:
        fx, fy, cx, cy, w, h = cams[im["camera_id"]]
        out.append({"R": im["R"], "t": im["t"], "fx": fx, "fy": fy,
                    "cx": cx, "cy": cy, "W": w, "H": h,
                    "stem": os.path.splitext(os.path.basename(im["name"]))[0]})
    return out


def load_instance_masks(masks_dir, cameras):
    """Map camera stem -> bool mask array. Missing masks -> all-False."""
    from PIL import Image
    files = {os.path.splitext(f)[0]: os.path.join(masks_dir, f)
             for f in os.listdir(masks_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    out = {}
    for cam in cameras:
        path = files.get(cam["stem"])
        if path is None:
            out[cam["stem"]] = None  # treated as not visible in this view
        else:
            out[cam["stem"]] = np.asarray(Image.open(path).convert("L")) > 127
    return out


def classify_visibility(pts, normals, cameras, masks, min_views=1,
                        view_stride=1):
    """Boolean array: True = observed in >= min_views views."""
    count = np.zeros(len(pts), dtype=np.int32)
    for cam in cameras[::view_stride]:
        m = masks.get(cam["stem"])
        if m is None:
            continue
        Xc = pts @ cam["R"].T + cam["t"]
        z = Xc[:, 2]
        ok = z > 1e-6
        u = np.where(ok, cam["fx"] * Xc[:, 0] / np.where(ok, z, 1) + cam["cx"], -1)
        v = np.where(ok, cam["fy"] * Xc[:, 1] / np.where(ok, z, 1) + cam["cy"], -1)
        Hm, Wm = m.shape
        sy = Hm / cam["H"]; sx = Wm / cam["W"]  # mask may differ in resolution
        ui = (u * sx).astype(np.int64); vi = (v * sy).astype(np.int64)
        ok &= (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm)
        inmask = np.zeros(len(pts), dtype=bool)
        inmask[ok] = m[vi[ok], ui[ok]]
        C = -cam["R"].T @ cam["t"]
        front = np.einsum("ij,ij->i", normals, C[None, :] - pts) > 0
        count += (inmask & front).astype(np.int32)
    return count >= min_views


# ---------------------------------------------------------------------------
# Sampling & metrics
# ---------------------------------------------------------------------------

def sample_with_normals(mesh, n):
    """Uniform surface samples + face normals. Returns (pts [n,3], normals [n,3])."""
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    pts, fidx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[fidx]
    return np.asarray(pts, dtype=np.float64), np.asarray(normals, dtype=np.float64)


def icp_align(src_pts, dst_tree, dst_pts, iters=30, tol=1e-7):
    """Point-to-point ICP. Returns 4x4 transform mapping src->dst."""
    T = np.eye(4)
    src = src_pts.copy()
    prev_err = np.inf
    for _ in range(iters):
        d, idx = dst_tree.query(src, k=1)
        err = float(np.mean(d))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        tgt = dst_pts[idx]
        mu_s, mu_t = src.mean(0), tgt.mean(0)
        H = (src - mu_s).T @ (tgt - mu_t)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_t - R @ mu_s
        src = src @ R.T + t
        Ti = np.eye(4)
        Ti[:3, :3], Ti[:3, 3] = R, t
        T = Ti @ T
    return T


def compute_metrics(recon_mesh, gt_mesh, n_samples=100_000, taus=DEFAULT_TAUS,
                    icp=False, visibility=None):
    """All metrics for one recon/GT mesh pair. Returns dict.

    visibility: optional (cameras, masks, min_views) tuple; adds
    observed/unobserved split metrics (accuracy_unobs = hallucination).
    """
    r_pts, r_nrm = sample_with_normals(recon_mesh, n_samples)
    g_pts, g_nrm = sample_with_normals(gt_mesh, n_samples)

    g_tree = cKDTree(g_pts)

    icp_T = None
    if icp:
        icp_T = icp_align(r_pts[:: max(1, len(r_pts) // 20000)], g_tree, g_pts)
        R, t = icp_T[:3, :3], icp_T[:3, 3]
        r_pts = r_pts @ R.T + t
        r_nrm = r_nrm @ R.T

    r_tree = cKDTree(r_pts)

    d_r2g, i_r2g = g_tree.query(r_pts, k=1)   # recon -> GT
    d_g2r, i_g2r = r_tree.query(g_pts, k=1)   # GT -> recon

    acc = float(np.mean(d_r2g))
    comp = float(np.mean(d_g2r))

    nc_r2g = np.abs(np.sum(r_nrm * g_nrm[i_r2g], axis=1))
    nc_g2r = np.abs(np.sum(g_nrm * r_nrm[i_g2r], axis=1))
    nc = float((np.mean(nc_r2g) + np.mean(nc_g2r)) / 2.0)

    out = {
        "accuracy": acc,
        "completion": comp,
        "chamfer_l1": (acc + comp) / 2.0,
        "normal_consistency": nc,
        "n_pts_recon": len(r_pts),
        "n_pts_gt": len(g_pts),
    }
    for tau in taus:
        precision = float(np.mean(d_r2g < tau))
        recall = float(np.mean(d_g2r < tau))
        f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out[f"precision@{tau}"] = precision
        out[f"recall@{tau}"] = recall
        out[f"f@{tau}"] = f
    if icp_T is not None:
        out["icp_translation_norm"] = float(np.linalg.norm(icp_T[:3, 3]))

    if visibility is not None:
        cameras, vmasks, min_views = visibility
        r_vis = classify_visibility(r_pts, r_nrm, cameras, vmasks, min_views)
        g_vis = classify_visibility(g_pts, g_nrm, cameras, vmasks, min_views)
        out["frac_recon_obs"] = float(np.mean(r_vis))
        out["frac_gt_obs"] = float(np.mean(g_vis))
        out["accuracy_obs"] = float(np.mean(d_r2g[r_vis])) if r_vis.any() else float("nan")
        # accuracy on unobserved recon points = hallucination metric
        out["accuracy_unobs"] = float(np.mean(d_r2g[~r_vis])) if (~r_vis).any() else float("nan")
        out["completion_obs"] = float(np.mean(d_g2r[g_vis])) if g_vis.any() else float("nan")
        out["completion_unobs"] = float(np.mean(d_g2r[~g_vis])) if (~g_vis).any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# Auto-matching recon mesh -> GT object_id
# ---------------------------------------------------------------------------

def auto_match_gt_id(recon_mesh, vertices, tris, tri_ids, shortlist_k=8,
                     n_probe=2000, exclude_ids=(0,), min_faces=10):
    """Pick GT object_id whose instance mesh best fits recon (min one-way chamfer).

    Shortlists by centroid distance, then probes with small samples.
    Returns (best_id, best_dist, ranking list[(id, dist)]).
    """
    r_pts, _ = sample_with_normals(recon_mesh, n_probe)
    r_centroid = r_pts.mean(0)

    ids, counts = np.unique(tri_ids, return_counts=True)
    cand = []
    for oid, cnt in zip(ids, counts):
        if int(oid) in exclude_ids or cnt < min_faces:
            continue
        vsel = vertices[np.unique(tris[tri_ids == oid].reshape(-1))]
        cand.append((int(oid), float(np.linalg.norm(vsel.mean(0) - r_centroid))))
    cand.sort(key=lambda x: x[1])
    cand = cand[:shortlist_k]

    ranking = []
    for oid, _ in cand:
        gmesh = extract_instance_mesh(vertices, tris, tri_ids, oid)
        g_pts, _ = sample_with_normals(gmesh, n_probe)
        d, _ = cKDTree(g_pts).query(r_pts, k=1)
        ranking.append((oid, float(np.mean(d))))
    ranking.sort(key=lambda x: x[1])
    if not ranking:
        raise ValueError("no GT candidates found for auto-match")
    return ranking[0][0], ranking[0][1], ranking


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_list_gt(args):
    vertices, tris, tri_ids = load_semantic_mesh(args.gt_mesh)
    names = load_class_names(args.info)
    ids, counts = np.unique(tri_ids, return_counts=True)
    print(f"{'id':>5} {'class':<20} {'tris':>8}  centroid (x,y,z)            extent (x,y,z)")
    for oid, cnt in zip(ids, counts):
        vsel = vertices[np.unique(tris[tri_ids == oid].reshape(-1))]
        c, e = vsel.mean(0), vsel.max(0) - vsel.min(0)
        cls = names.get(int(oid), "?")
        print(f"{int(oid):>5} {cls:<20} {int(cnt):>8}  "
              f"({c[0]:6.2f},{c[1]:6.2f},{c[2]:6.2f})  "
              f"({e[0]:5.2f},{e[1]:5.2f},{e[2]:5.2f})")


def cmd_extract_gt(args):
    vertices, tris, tri_ids = load_semantic_mesh(args.gt_mesh)
    mesh = extract_instance_mesh(vertices, tris, tri_ids, args.gt_id)
    mesh.export(args.out)
    print(f"saved object_id={args.gt_id}: {len(mesh.vertices)} verts, "
          f"{len(mesh.faces)} tris -> {args.out}")


def _load_recon(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"empty recon mesh: {path}")
    return mesh


def _print_metrics(m, taus):
    print(f"  accuracy            : {m['accuracy']*1000:8.2f} mm")
    print(f"  completion          : {m['completion']*1000:8.2f} mm")
    print(f"  chamfer-L1          : {m['chamfer_l1']*1000:8.2f} mm")
    print(f"  normal consistency  : {m['normal_consistency']:8.4f}")
    for tau in taus:
        print(f"  F@{tau*100:g}cm (P/R)       : {m[f'f@{tau}']:.4f} "
              f"({m[f'precision@{tau}']:.4f}/{m[f'recall@{tau}']:.4f})")
    if "accuracy_unobs" in m:
        print(f"  [vis] acc obs/unobs : {m['accuracy_obs']*1000:8.2f} / "
              f"{m['accuracy_unobs']*1000:.2f} mm  (unobs = hallucination)")
        print(f"  [vis] comp obs/unobs: {m['completion_obs']*1000:8.2f} / "
              f"{m['completion_unobs']*1000:.2f} mm")
        print(f"  [vis] frac obs r/gt : {m['frac_recon_obs']:8.3f} / "
              f"{m['frac_gt_obs']:.3f}")


def _load_visibility(args, label=None):
    """Build (cameras, masks, min_views) from CLI args, or None."""
    cdir, mdir = args.colmap_dir, args.masks_dir
    if label is not None and args.data_root:
        cdir = os.path.join(args.data_root, label, "sparse", "0")
        mdir = os.path.join(args.data_root, label, "masks")
    if not cdir or not mdir:
        return None
    cams = load_instance_cameras(cdir)
    vmasks = load_instance_masks(mdir, cams)
    return (cams, vmasks, args.min_vis_views)


def _eval_one(recon_path, gt_data, args, gt_id=None, label=None):
    vertices, tris, tri_ids = gt_data
    recon = _load_recon(recon_path)
    matched_dist = None
    if gt_id is None:
        gt_id, matched_dist, ranking = auto_match_gt_id(
            recon, vertices, tris, tri_ids,
            exclude_ids=tuple(args.exclude_ids))
        top = ", ".join(f"{i}:{d*1000:.1f}mm" for i, d in ranking[:3])
        print(f"  auto-match -> object_id {gt_id} (top: {top})")
        if len(ranking) > 1 and ranking[1][1] < 1.5 * ranking[0][1]:
            print("  WARNING: ambiguous match (runner-up within 1.5x). "
                  "Verify gt_id manually.")
    gt_mesh = extract_instance_mesh(vertices, tris, tri_ids, gt_id)
    m = compute_metrics(recon, gt_mesh, n_samples=args.n_samples,
                        taus=args.taus, icp=args.icp,
                        visibility=_load_visibility(args, label))
    m["gt_id"] = gt_id
    if matched_dist is not None:
        m["match_dist"] = matched_dist
    return m


def cmd_eval(args):
    gt_data = load_semantic_mesh(args.gt_mesh)
    gt_id = args.gt_id if not args.auto_match else None
    if gt_id is None and not args.auto_match:
        sys.exit("eval: need --gt_id or --auto_match")
    print(f"recon: {args.recon}")
    m = _eval_one(args.recon, gt_data, args, gt_id=gt_id)
    print(f"GT object_id: {m['gt_id']}")
    _print_metrics(m, args.taus)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"saved -> {args.out}")


def cmd_diagnose(args):
    """Pinpoint failure stage: global frame vs instance identity vs masks."""
    vertices, tris, tri_ids = load_semantic_mesh(args.gt_mesh)
    recon = _load_recon(args.recon)
    r_pts, r_nrm = sample_with_normals(recon, 5000)

    print("== 1. global frames ==")
    print(f"  recon bbox min/max : {recon.vertices.min(0).round(2)} / "
          f"{recon.vertices.max(0).round(2)}")
    print(f"  GT scene bbox      : {vertices.min(0).round(2)} / "
          f"{vertices.max(0).round(2)}")
    # one-way distance recon -> FULL GT scene (all instances incl. walls)
    scene = trimesh.Trimesh(vertices=vertices, faces=tris, process=False)
    sub, _ = sample_with_normals(scene, 200_000)
    d, _ = cKDTree(sub).query(r_pts, k=1)
    print(f"  recon -> full GT scene: mean {np.mean(d)*1000:.1f} mm, "
          f"median {np.median(d)*1000:.1f} mm")
    print("  -> median < ~50mm: frames ALIGNED (problem is instance identity)")
    print("  -> median large  : frame MISMATCH (transform needed)")

    if args.colmap_dir and args.masks_dir:
        print("== 2. cameras & masks ==")
        cams = load_instance_cameras(args.colmap_dir)
        vmasks = load_instance_masks(args.masks_dir, cams)
        n_match = sum(1 for c in cams if vmasks.get(c["stem"]) is not None)
        print(f"  views: {len(cams)}, masks matched by stem: {n_match}")
        if n_match == 0:
            stems = [c["stem"] for c in cams[:3]]
            files = sorted(os.listdir(args.masks_dir))[:3]
            print(f"  MASK NAME MISMATCH. camera stems e.g. {stems}")
            print(f"                      mask files   e.g. {files}")
        else:
            in_img = in_mask = front = 0
            n_cam_used = 0
            for cam in cams:
                m = vmasks.get(cam["stem"])
                if m is None:
                    continue
                n_cam_used += 1
                Xc = r_pts @ cam["R"].T + cam["t"]
                z = Xc[:, 2]
                ok = z > 1e-6
                Hm, Wm = m.shape
                u = (cam["fx"] * Xc[:, 0] / np.where(ok, z, 1) + cam["cx"]) * (Wm / cam["W"])
                v = (cam["fy"] * Xc[:, 1] / np.where(ok, z, 1) + cam["cy"]) * (Hm / cam["H"])
                ui, vi = u.astype(np.int64), v.astype(np.int64)
                ok &= (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm)
                in_img += ok.mean()
                im = np.zeros(len(r_pts), bool)
                im[ok] = m[vi[ok], ui[ok]]
                in_mask += im.mean()
                C = -cam["R"].T @ cam["t"]
                front += ((np.einsum("ij,ij->i", r_nrm, C[None] - r_pts) > 0) & im).mean()
            print(f"  recon pts avg over {n_cam_used} views: "
                  f"in-image {in_img/n_cam_used:.3f}, in-mask {in_mask/n_cam_used:.3f}, "
                  f"in-mask&front {front/n_cam_used:.3f}")
            print("  -> in-image~0: camera/recon frame mismatch")
            print("  -> in-image ok, in-mask~0: wrong masks_dir or empty masks")
            print("  -> in-mask ok, front~0: normal orientation flipped")


def cmd_batch(args):
    gt_data = load_semantic_mesh(args.gt_mesh)

    label_map = {}  # label -> gt_id
    if args.map:
        with open(args.map) as f:
            for row in csv.DictReader(f):
                label_map[str(row["label"])] = int(row["gt_id"])

    paths = sorted(glob.glob(args.recon_glob))
    if not paths:
        sys.exit(f"batch: no files match {args.recon_glob}")

    rows = []
    for p in paths:
        parts = os.path.normpath(p).split(os.sep)
        label = parts[args.label_from_path]
        gt_id = label_map.get(label)
        if gt_id is None and not args.auto_match:
            print(f"[skip] {label}: no gt_id in map (use --auto_match or --map)")
            continue
        print(f"[{label}] {p}")
        try:
            m = _eval_one(p, gt_data, args, gt_id=gt_id, label=label)
        except Exception as e:  # keep batch going
            print(f"  ERROR: {e}")
            rows.append({"label": label, "recon": p, "error": str(e)})
            continue
        _print_metrics(m, args.taus)
        m.update({"label": label, "recon": p})
        rows.append(m)

    tau_keys = []
    for tau in args.taus:
        tau_keys += [f"precision@{tau}", f"recall@{tau}", f"f@{tau}"]
    fields = (["label", "gt_id"] + METRIC_KEYS_BASE + tau_keys +
              ["match_dist", "icp_translation_norm", "recon", "error"])
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {args.out}")

    ok = [r for r in rows if "chamfer_l1" in r]
    if ok:
        print(f"mean chamfer-L1 : {np.mean([r['chamfer_l1'] for r in ok])*1000:.2f} mm")
        for tau in args.taus:
            print(f"mean F@{tau*100:g}cm     : {np.mean([r[f'f@{tau}'] for r in ok]):.4f}")


# ---------------------------------------------------------------------------

def _add_common_eval_args(p):
    p.add_argument("--gt_mesh", required=True, help="habitat/mesh_semantic.ply")
    p.add_argument("--n_samples", type=int, default=100_000)
    p.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    p.add_argument("--icp", action="store_true",
                   help="point-to-point ICP refinement before metrics")
    p.add_argument("--auto_match", action="store_true",
                   help="pick GT object_id by min one-way chamfer")
    p.add_argument("--exclude_ids", type=int, nargs="+", default=[0],
                   help="GT ids excluded from auto-match (default: 0=unlabeled)")
    # visibility split (hallucination metric)
    p.add_argument("--colmap_dir", default=None,
                   help="COLMAP sparse dir of the instance (sparse/0)")
    p.add_argument("--masks_dir", default=None,
                   help="per-view instance mask dir (png, stem-matched)")
    p.add_argument("--data_root", default=None,
                   help="batch: per-label root -> <root>/<label>/{sparse/0,masks}")
    p.add_argument("--min_vis_views", type=int, default=2,
                   help="views required to count a point as observed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-gt", help="list GT instances")
    p.add_argument("--gt_mesh", required=True)
    p.add_argument("--info", default=None, help="habitat/info_semantic.json")
    p.set_defaults(func=cmd_list_gt)

    p = sub.add_parser("extract-gt", help="save one GT instance mesh")
    p.add_argument("--gt_mesh", required=True)
    p.add_argument("--gt_id", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_extract_gt)

    p = sub.add_parser("eval", help="evaluate one recon mesh")
    _add_common_eval_args(p)
    p.add_argument("--recon", required=True, help="fuse_post.ply")
    p.add_argument("--gt_id", type=int, default=None)
    p.add_argument("--out", default=None, help="save metrics JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("diagnose", help="pinpoint failure stage (frames/masks)")
    p.add_argument("--gt_mesh", required=True)
    p.add_argument("--recon", required=True)
    p.add_argument("--colmap_dir", default=None)
    p.add_argument("--masks_dir", default=None)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("batch", help="evaluate many recon meshes -> CSV")
    _add_common_eval_args(p)
    p.add_argument("--recon_glob", required=True,
                   help="e.g. 'output/scene/exp/*/train/ours_7000/fuse_post.ply'")
    p.add_argument("--label_from_path", type=int, default=-4,
                   help="path component index used as label (default -4 = <id>)")
    p.add_argument("--map", default=None,
                   help="CSV with columns label,gt_id (overrides auto-match)")
    p.add_argument("--out", default="eval_object_mesh.csv")
    p.set_defaults(func=cmd_batch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
