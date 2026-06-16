#!/usr/bin/env python3
"""
Axis 3 — Observation-consistent fusion (carve).

Closes Per-Object Refinement with the *visibility-gated* principle:
  recon is kept faithful; generated (Amodal3R) geometry is added ONLY where it
  does not contradict image evidence.

Pipeline per object:
  1. recon (TSDF) kept exactly as-is.
  2. occ filter: keep only generated faces far (> occ_thresh) from the recon
     surface  (avoid duplicating the visible front recon already has).
  3. SILHOUETTE / VISUAL-HULL veto (the carve): drop any occluded face that
     projects *inside an image but outside the instance silhouette* in any view
     — that geometry is provably outside the visual hull = evidence-contradicting
     = hallucination. Uses the SAME colmap+mask projection as eval_object_mesh.
  4. append surviving faces to recon -> fused mesh.

HONEST SCOPE (matches our agreed framing):
  - This GUARANTEES "no evidence-contradicting geometry" (observation-consistency),
    NOT "no hallucination". Occluded back-surface that projects *inside* the
    silhouette is unverifiable: it is KEPT and FLAGGED, not certified correct.
  - Reported: n_occ (candidates), n_carved (evidence-violating, removed),
    frac_carved (high -> risky object/registration), n_kept (unverifiable fill).

Reuses eval_object_mesh.py (same dir) for camera/mask loading + projection.

Usage (run from /home/elicer/RefineGS):
    conda activate split_and_splat
    python tools/fuse_carve.py --batch \
        --labels 97 98 --seeds 1 2 3 \
        --recon_root output/replica_room0/axis3_sweep/reg_strong \
        --gen_root   ~/Amodal3R/poc_output \
        --data_root  data/replica_room0/masks \
        --gt_mesh    ../room_0/habitat/mesh_semantic.ply \
        --gt_map     gt_map.csv \
        --occ_thresh 0.05
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# reuse eval's colmap/mask machinery (same projection convention)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_object_mesh import load_instance_cameras, load_instance_masks


# ── helpers (mirrored from fuse_generated_recon.py) ───────────────────────────

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(str(path))
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    return m


def chamfer_l1(a: np.ndarray, b: np.ndarray, n: int = 10000) -> float:
    rng = np.random.default_rng(0)
    a_ = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b_ = b[rng.choice(len(b), min(n, len(b)), replace=False)]
    diff = a_[:, None, :] - b_[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    return float((d.min(1).mean() + d.min(0).mean()) / 2)


def load_gt_id_map(csv_path: Path) -> dict:
    m = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m[row["label"]] = int(row["gt_id"])
    return m


def load_gt_object_verts(gt_mesh_path: Path, gt_id: int):
    try:
        import trimesh
        with open(str(gt_mesh_path), "rb") as f:
            data = trimesh.exchange.ply.load_ply(f)
        verts = np.array(data["vertices"], dtype=np.float64)
        fd = data["metadata"]["_ply_raw"]["face"]["data"]
        obj_ids = fd["object_id"]
        face_vids = fd["vertex_indices"]["f1"]
        fm = (obj_ids == gt_id)
        if fm.sum() == 0:
            return None
        return verts[np.unique(face_vids[fm].flatten())]
    except Exception as e:
        print(f"  [WARN] load_gt_object_verts: {e}")
        return None


def filter_gen_faces(gen_mesh, recon_verts, occ_thresh):
    """Keep only gen faces whose mean vertex dist to nearest recon vertex
    exceeds occ_thresh. Returns (subbmesh, n_occ)."""
    gv = np.asarray(gen_mesh.vertices)
    gf = np.asarray(gen_mesh.triangles)
    if len(gf) == 0:
        return o3d.geometry.TriangleMesh(), 0
    rp = o3d.geometry.PointCloud()
    rp.points = o3d.utility.Vector3dVector(recon_verts)
    tree = o3d.geometry.KDTreeFlann(rp)
    vd = np.empty(len(gv))
    for i, pt in enumerate(gv):
        _, _, d2 = tree.search_knn_vector_3d(pt, 1)
        vd[i] = d2[0] ** 0.5
    occ = vd[gf].mean(axis=1) > occ_thresh
    if occ.sum() == 0:
        return o3d.geometry.TriangleMesh(), 0
    sf = gf[occ]
    uv, ni = np.unique(sf, return_inverse=True)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(gv[uv])
    out.triangles = o3d.utility.Vector3iVector(ni.reshape(sf.shape))
    out.compute_vertex_normals()
    return out, int(occ.sum())


# ── the carve: silhouette / visual-hull veto ──────────────────────────────────

def silhouette_violating_vertices(verts, cameras, masks):
    """Per-vertex bool: True = vertex projects INSIDE an image but OUTSIDE the
    instance silhouette in >=1 view -> outside visual hull -> evidence-violating.
    Projection convention identical to eval_object_mesh.classify_visibility."""
    viol = np.zeros(len(verts), dtype=bool)
    in_img_count = np.zeros(len(verts), dtype=np.int32)
    for cam in cameras:
        m = masks.get(cam["stem"])
        if m is None:
            continue
        Xc = verts @ cam["R"].T + cam["t"]
        z = Xc[:, 2]
        ok = z > 1e-6
        u = cam["fx"] * Xc[:, 0] / np.where(ok, z, 1) + cam["cx"]
        v = cam["fy"] * Xc[:, 1] / np.where(ok, z, 1) + cam["cy"]
        Hm, Wm = m.shape
        sy = Hm / cam["H"]; sx = Wm / cam["W"]
        ui = (u * sx).astype(np.int64); vi = (v * sy).astype(np.int64)
        in_img = ok & (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm)
        in_mask = np.zeros(len(verts), dtype=bool)
        in_mask[in_img] = m[vi[in_img], ui[in_img]]
        viol |= (in_img & ~in_mask)      # in image but outside silhouette
        in_img_count += in_img.astype(np.int32)
    return viol, in_img_count


def carve_faces(occ_mesh, cameras, masks):
    """Remove faces with any silhouette-violating vertex. Returns
    (carved_mesh, n_in, n_kept)."""
    gv = np.asarray(occ_mesh.vertices)
    gf = np.asarray(occ_mesh.triangles)
    if len(gf) == 0:
        return occ_mesh, 0, 0
    viol, _ = silhouette_violating_vertices(gv, cameras, masks)
    keep_face = ~viol[gf].any(axis=1)
    n_in = len(gf)
    if keep_face.sum() == 0:
        return o3d.geometry.TriangleMesh(), n_in, 0
    sf = gf[keep_face]
    uv, ni = np.unique(sf, return_inverse=True)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(gv[uv])
    out.triangles = o3d.utility.Vector3iVector(ni.reshape(sf.shape))
    out.compute_vertex_normals()
    return out, n_in, int(keep_face.sum())


# ── per-object fusion ─────────────────────────────────────────────────────────

def fuse_carve(recon_path, gen_path, out_path, colmap_dir, masks_dir,
               occ_thresh=0.05, gt_mesh_path=None, gt_id=None, verbose=True):
    recon = load_mesh(recon_path)
    gen = load_mesh(gen_path)
    rv = np.asarray(recon.vertices)
    if len(rv) == 0 or len(np.asarray(gen.vertices)) == 0:
        print(f"  [ERROR] empty mesh"); return None

    cameras = load_instance_cameras(str(colmap_dir))
    masks = load_instance_masks(str(masks_dir), cameras)

    # 1. occluded candidate faces
    occ_mesh, n_occ = filter_gen_faces(gen, rv, occ_thresh)
    # 2. silhouette / visual-hull veto
    kept_mesh, n_in, n_kept = carve_faces(occ_mesh, cameras, masks)
    n_carved = n_in - n_kept

    # 3. append surviving faces to recon
    if n_kept > 0:
        kv = np.asarray(kept_mesh.vertices)
        kf = np.asarray(kept_mesh.triangles) + len(rv)
        rf = np.asarray(recon.triangles)
        fused = o3d.geometry.TriangleMesh()
        fused.vertices = o3d.utility.Vector3dVector(np.vstack([rv, kv]))
        fused.triangles = o3d.utility.Vector3iVector(np.vstack([rf, kf]))
        fused.compute_vertex_normals()
    else:
        fused = recon
    o3d.io.write_triangle_mesh(str(out_path), fused)

    fv = np.asarray(fused.vertices)
    res = {
        "n_occ": n_occ,
        "n_carved": n_carved,
        "n_kept": n_kept,
        "frac_carved": round(n_carved / n_occ, 3) if n_occ else 0.0,
        "n_fused_verts": len(fv),
        "cd_vs_recon_mm": round(chamfer_l1(fv, rv) * 1000, 1),
    }
    if gt_mesh_path is not None and gt_id is not None and Path(gt_mesh_path).exists():
        gt = load_gt_object_verts(Path(gt_mesh_path), gt_id)
        if gt is not None and len(gt) > 0:
            cd_r = chamfer_l1(rv, gt); cd_f = chamfer_l1(fv, gt)
            res["cd_vs_gt_recon_mm"] = round(cd_r * 1000, 1)
            res["cd_vs_gt_fused_mm"] = round(cd_f * 1000, 1)
            res["cd_improvement_mm"] = round((cd_r - cd_f) * 1000, 1)
    if verbose:
        print(f"  occ={n_occ}  carved(evidence-violating)={n_carved} "
              f"({res['frac_carved']*100:.0f}%)  kept(unverifiable fill)={n_kept}")
        if "cd_vs_gt_fused_mm" in res:
            print(f"  CD vs GT: recon={res['cd_vs_gt_recon_mm']}mm "
                  f"fused={res['cd_vs_gt_fused_mm']}mm "
                  f"Δ={res['cd_improvement_mm']:+.1f}mm")
        print(f"  → {out_path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--recon", type=Path)
    ap.add_argument("--gen", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--colmap_dir", type=Path)
    ap.add_argument("--masks_dir", type=Path)
    ap.add_argument("--recon_root", default="output/replica_room0/axis3_sweep/reg_strong", type=Path)
    ap.add_argument("--gen_root", default="~/Amodal3R/poc_output", type=Path)
    ap.add_argument("--data_root", default="data/replica_room0/masks", type=Path,
                    help="per-label root -> <root>/<label>/{sparse/0,masks}")
    ap.add_argument("--gt_mesh", type=Path, default=None)
    ap.add_argument("--gt_id", type=int, default=None)
    ap.add_argument("--gt_map", type=Path, default=None)
    ap.add_argument("--labels", nargs="+", default=["97", "98", "75"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--occ_thresh", type=float, default=0.05)
    args = ap.parse_args()

    if not args.batch:
        r = fuse_carve(args.recon, args.gen,
                       args.out or args.gen.parent / "mesh_carved.ply",
                       args.colmap_dir, args.masks_dir,
                       occ_thresh=args.occ_thresh,
                       gt_mesh_path=args.gt_mesh, gt_id=args.gt_id)
        if r:
            import json; print(json.dumps(r, indent=2))
        return

    gen_root = Path(str(args.gen_root)).expanduser()
    recon_root = Path(str(args.recon_root))
    data_root = Path(str(args.data_root))
    gt_map = load_gt_id_map(args.gt_map) if args.gt_map else {}
    has_gt = args.gt_mesh is not None

    print(f"\n{'label':>6} {'seed':>5} {'n_occ':>7} {'carved':>7} {'frac%':>6} "
          f"{'kept':>7} {'CD_rcn':>7} {'CD_GTr':>7} {'CD_GTf':>7} {'Δmm':>7}")
    print("-" * 80)
    for label in args.labels:
        rg = list(recon_root.glob(f"{label}/train/ours_7000/fuse_post.ply")) or \
             list(recon_root.glob(f"{label}/train/ours_*/fuse_post.ply"))
        if not rg:
            print(f"  [SKIP] {label}: no recon"); continue
        recon_path = rg[0]
        colmap_dir = data_root / label / "sparse" / "0"
        masks_dir = data_root / label / "masks"
        gt_id = gt_map.get(label, args.gt_id)
        for seed in args.seeds:
            gen_path = gen_root / label / f"seed_{seed}" / "mesh_registered.ply"
            out_path = gen_root / label / f"seed_{seed}" / "mesh_carved.ply"
            if not gen_path.exists():
                print(f"  [SKIP] {label}/seed{seed}: no mesh_registered.ply"); continue
            r = fuse_carve(recon_path, gen_path, out_path, colmap_dir, masks_dir,
                           occ_thresh=args.occ_thresh, gt_mesh_path=args.gt_mesh,
                           gt_id=gt_id, verbose=False)
            if not r:
                continue
            row = (f"{label:>6} {seed:>5} {r['n_occ']:>7} {r['n_carved']:>7} "
                   f"{r['frac_carved']*100:>5.0f}% {r['n_kept']:>7} "
                   f"{r['cd_vs_recon_mm']:>7.1f}")
            if has_gt and "cd_vs_gt_fused_mm" in r:
                row += (f" {r['cd_vs_gt_recon_mm']:>7.1f} {r['cd_vs_gt_fused_mm']:>7.1f} "
                        f"{r['cd_improvement_mm']:>7.1f}")
            print(row)
    print("-" * 80)
    print("frac_carved 높음 = 생성/정렬이 증거를 많이 위반(위험). "
          "kept = 검증 불가 occluded fill(평가에서 accuracy_unobs로 flag).")
    print("다음: eval_object_mesh.py eval 로 mesh_carved.ply visibility-split 평가.")


if __name__ == "__main__":
    main()
