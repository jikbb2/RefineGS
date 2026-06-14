#!/usr/bin/env python3
"""
Fusion PoC (A-2): Screened Poisson mesh fusion.

Combines:
  - TSDF recon mesh   (visible region, high confidence)
  - Registered Amodal3R mesh (occluded region fill-in)

Algorithm:
  1. Sample oriented point clouds from both meshes
  2. For each generated point, compute distance to nearest TSDF point
     → keep only points in "occluded" region (dist > occ_thresh)
  3. Merge: TSDF points (full) + generated points (occluded only)
  4. Screened Poisson reconstruction (open3d, depth=9)
  5. Evaluate fused vs GT using eval_object_mesh harness
  6. Save fused mesh + report CD improvement

Usage (run from /home/elicer/RefineGS):
    conda activate split_and_splat

    # single instance, single seed
    python tools/fuse_generated_recon.py \
        --recon  output/replica_room0/axis3_sweep/reg_strong/97/train/ours_7000/fuse_post.ply \
        --gen    ~/Amodal3R/poc_output/97/seed_1/mesh_registered.ply \
        --out    ~/Amodal3R/poc_output/97/seed_1/mesh_fused.ply \
        --gt_mesh ../room_0/habitat/mesh_semantic.ply \
        --gt_id  72

    # batch: best seed per label
    python tools/fuse_generated_recon.py --batch \
        --recon_root output/replica_room0/axis3_sweep/reg_strong \
        --gen_root   ~/Amodal3R/poc_output \
        --gt_mesh    ../room_0/habitat/mesh_semantic.ply \
        --gt_map     gt_map.csv \
        --labels 97 98 75 --seeds 1 2 3
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ── helpers ──────────────────────────────────────────────────────────────────

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(str(path))
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    if not m.has_triangle_normals():
        m.compute_triangle_normals()
    return m


def sample_pcd_with_normals(mesh: o3d.geometry.TriangleMesh,
                             n: int = 50000) -> o3d.geometry.PointCloud:
    """Sample point cloud; normals come from interpolated vertex normals."""
    pcd = mesh.sample_points_poisson_disk(n, use_triangle_normal=False)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.05, 30))
        pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def occluded_mask(gen_pts: np.ndarray,
                  recon_tree: o3d.geometry.KDTreeFlann,
                  occ_thresh: float) -> np.ndarray:
    """Boolean mask: True = generated point is far from recon (→ occluded region)."""
    mask = np.zeros(len(gen_pts), dtype=bool)
    for i, pt in enumerate(gen_pts):
        [_, _, dist2] = recon_tree.search_knn_vector_3d(pt, 1)
        mask[i] = (dist2[0] ** 0.5) > occ_thresh
    return mask


def chamfer_l1(a: np.ndarray, b: np.ndarray, n: int = 10000) -> float:
    rng = np.random.default_rng(0)
    a_ = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b_ = b[rng.choice(len(b), min(n, len(b)), replace=False)]
    diff = a_[:, None, :] - b_[None, :, :]
    d   = np.linalg.norm(diff, axis=2)
    return float((d.min(1).mean() + d.min(0).mean()) / 2)


def load_gt_id_map(csv_path: Path) -> dict:
    """Read label→gt_id from gt_map.csv."""
    m = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m[row["label"]] = int(row["gt_id"])
    return m


# ── core fusion ───────────────────────────────────────────────────────────────

def fuse(recon_path: Path, gen_path: Path, out_path: Path,
         occ_thresh: float = 0.05,
         n_recon: int = 80000,
         n_gen: int = 80000,
         poisson_depth: int = 9,
         density_trim: float = 0.05,
         gt_mesh_path: Path = None,
         gt_id: int = None,
         verbose: bool = True) -> dict:
    """
    Fuse registered generated mesh into TSDF recon via screened Poisson.

    occ_thresh: distance (m) from TSDF surface above which generated points
                are considered 'occluded' and included in fusion.
                ~5cm = TSDF voxel resolution × 5.  Increase for noisy recons.
    poisson_depth: Poisson octree depth. 9 → ~5mm resolution, 8 → ~1cm.
    density_trim: remove lowest X% density vertices from Poisson output.
    """
    recon_mesh = load_mesh(recon_path)
    gen_mesh   = load_mesh(gen_path)

    rv = np.asarray(recon_mesh.vertices)
    gv = np.asarray(gen_mesh.vertices)
    if len(rv) == 0:
        print(f"  [ERROR] empty recon mesh: {recon_path}")
        return None
    if len(gv) == 0:
        print(f"  [ERROR] empty gen mesh: {gen_path}")
        return None

    # ── 1. Sample PCDs ────────────────────────────────────────────────────────
    recon_pcd = sample_pcd_with_normals(recon_mesh, n_recon)
    gen_pcd   = sample_pcd_with_normals(gen_mesh,   n_gen)

    # ── 2. Filter generated → occluded region only ────────────────────────────
    recon_tree = o3d.geometry.KDTreeFlann(recon_pcd)
    gen_pts    = np.asarray(gen_pcd.points)
    gen_nrm    = np.asarray(gen_pcd.normals)

    mask = occluded_mask(gen_pts, recon_tree, occ_thresh)
    n_occ = mask.sum()
    n_vis = (~mask).sum()

    if verbose:
        print(f"  generated pts: {len(gen_pts)}  "
              f"occluded={n_occ}  visible={n_vis}  "
              f"(occ_thresh={occ_thresh*100:.0f}cm)")

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    if n_occ == 0:
        if verbose:
            print("  [WARN] no occluded points found — using recon only")
        merged = recon_pcd
    else:
        gen_pcd_occ = gen_pcd.select_by_index(np.where(mask)[0])
        merged = recon_pcd + gen_pcd_occ

    # Re-orient normals consistently (Poisson needs this)
    merged.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.1, 30))
    merged.orient_normals_consistent_tangent_plane(30)

    # ── 4. Screened Poisson ───────────────────────────────────────────────────
    fused_mesh, densities = \
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            merged, depth=poisson_depth, scale=1.1, linear_fit=False)

    # Trim low-density outliers (floaters)
    if density_trim > 0 and len(densities) > 0:
        d_arr   = np.asarray(densities)
        thresh  = np.quantile(d_arr, density_trim)
        keep    = d_arr > thresh
        fused_mesh = fused_mesh.select_by_index(
            np.where(keep)[0])

    o3d.io.write_triangle_mesh(str(out_path), fused_mesh)

    fv = np.asarray(fused_mesh.vertices)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    result = {
        "n_recon_pts": len(np.asarray(recon_pcd.points)),
        "n_gen_occ_pts": int(n_occ),
        "n_fused_verts": len(fv),
        "occ_thresh_m": occ_thresh,
    }

    # CD vs recon (alignment quality proxy)
    cd_recon = chamfer_l1(fv, rv)
    result["cd_vs_recon_mm"] = round(cd_recon * 1000, 1)

    # CD vs GT (if provided)
    if gt_mesh_path is not None and gt_id is not None and gt_mesh_path.exists():
        gt_mesh  = load_mesh(gt_mesh_path)
        gt_verts = np.asarray(gt_mesh.vertices)
        if len(gt_verts) > 0:
            cd_gt_recon  = chamfer_l1(rv,  gt_verts)
            cd_gt_fused  = chamfer_l1(fv, gt_verts)
            result["cd_vs_gt_recon_mm"]  = round(cd_gt_recon  * 1000, 1)
            result["cd_vs_gt_fused_mm"]  = round(cd_gt_fused  * 1000, 1)
            result["cd_improvement_mm"]  = round((cd_gt_recon - cd_gt_fused) * 1000, 1)
            result["cd_improvement_pct"] = round(
                (1 - cd_gt_fused / cd_gt_recon) * 100, 1) if cd_gt_recon > 0 else 0

    if verbose:
        print(f"  fused verts: {len(fv)}")
        print(f"  CD vs recon: {result['cd_vs_recon_mm']}mm")
        if "cd_vs_gt_fused_mm" in result:
            print(f"  CD vs GT:  recon={result['cd_vs_gt_recon_mm']}mm  "
                  f"fused={result['cd_vs_gt_fused_mm']}mm  "
                  f"Δ={result['cd_improvement_mm']:+.1f}mm "
                  f"({result['cd_improvement_pct']:+.1f}%)")
        print(f"  → {out_path}")

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true")
    # single mode
    ap.add_argument("--recon",    type=Path)
    ap.add_argument("--gen",      type=Path)
    ap.add_argument("--out",      type=Path)
    ap.add_argument("--gt_mesh",  type=Path, default=None)
    ap.add_argument("--gt_id",    type=int,  default=None)
    # batch mode
    ap.add_argument("--recon_root", default="output/replica_room0/axis3_sweep/reg_strong", type=Path)
    ap.add_argument("--gen_root",   default="~/Amodal3R/poc_output", type=Path)
    ap.add_argument("--gt_map",     type=Path, default=None)
    ap.add_argument("--labels",     nargs="+", default=["97", "98", "75"])
    ap.add_argument("--seeds",      nargs="+", type=int, default=[1, 2, 3])
    # fusion params
    ap.add_argument("--occ_thresh",    type=float, default=0.05,
                    help="Distance (m) to TSDF surface below which gen points are excluded")
    ap.add_argument("--n_pts",         type=int,   default=80000)
    ap.add_argument("--poisson_depth", type=int,   default=9)
    ap.add_argument("--density_trim",  type=float, default=0.05,
                    help="Trim lowest X fraction of Poisson density (removes floaters)")
    args = ap.parse_args()

    kwargs = dict(
        occ_thresh=args.occ_thresh,
        n_recon=args.n_pts,
        n_gen=args.n_pts,
        poisson_depth=args.poisson_depth,
        density_trim=args.density_trim,
    )

    if not args.batch:
        out = args.out or args.gen.parent / "mesh_fused.ply"
        r = fuse(args.recon, args.gen, out,
                 gt_mesh_path=args.gt_mesh, gt_id=args.gt_id,
                 **kwargs)
        if r:
            print(json.dumps(r, indent=2))
        return

    # ── batch ─────────────────────────────────────────────────────────────────
    gen_root   = Path(args.gen_root).expanduser()
    recon_root = Path(args.recon_root)

    gt_map = {}
    if args.gt_map and Path(args.gt_map).exists():
        gt_map = load_gt_id_map(Path(args.gt_map))
    gt_mesh_path = args.gt_mesh

    all_results = []
    has_gt = gt_mesh_path is not None and gt_mesh_path.exists()

    hdr = (f"{'label':>6} {'seed':>5} {'occ_pts':>8} {'CD_recon':>9} "
           + (f"{'CD_GT_rcn':>10} {'CD_GT_fus':>10} {'Δmm':>7} {'Δ%':>6}" if has_gt else ""))
    print("\n" + hdr)
    print("-" * len(hdr))

    for label in args.labels:
        recon_glob = (list(recon_root.glob(f"{label}/train/ours_7000/fuse_post.ply"))
                      or list(recon_root.glob(f"{label}/train/ours_*/fuse_post.ply")))
        if not recon_glob:
            print(f"  [SKIP] label {label}: recon not found")
            continue
        recon_path = recon_glob[0]

        gt_id = gt_map.get(label) or gt_map.get(str(label))

        label_results = []
        for seed in args.seeds:
            gen_path = gen_root / label / f"seed_{seed}" / "mesh_registered.ply"
            out_path = gen_root / label / f"seed_{seed}" / "mesh_fused.ply"
            if not gen_path.exists():
                print(f"  [SKIP] {label}/seed{seed}: mesh_registered.ply not found "
                      f"(run register_generated_to_recon.py first)")
                continue

            r = fuse(recon_path, gen_path, out_path,
                     gt_mesh_path=gt_mesh_path, gt_id=gt_id,
                     verbose=False, **kwargs)
            if r:
                row = (f"{label:>6} {seed:>5} {r['n_gen_occ_pts']:>8} "
                       f"{r['cd_vs_recon_mm']:>9.1f}")
                if has_gt and "cd_vs_gt_fused_mm" in r:
                    row += (f" {r['cd_vs_gt_recon_mm']:>10.1f}"
                            f" {r['cd_vs_gt_fused_mm']:>10.1f}"
                            f" {r['cd_improvement_mm']:>7.1f}"
                            f" {r['cd_improvement_pct']:>5.1f}%")
                print(row)
                label_results.append({"label": label, "seed": seed, **r})
        all_results.extend(label_results)

    # ── per-label summary ─────────────────────────────────────────────────────
    if all_results:
        print("\n── per-label summary (mean over seeds) ──")
        for label in args.labels:
            rs = [r for r in all_results if r["label"] == label]
            if not rs:
                continue
            line = f"  label {label}: CD_vs_recon={np.mean([r['cd_vs_recon_mm'] for r in rs]):.1f}mm"
            if has_gt and all("cd_vs_gt_fused_mm" in r for r in rs):
                line += (f"  CD_GT_recon={np.mean([r['cd_vs_gt_recon_mm'] for r in rs]):.1f}mm"
                         f"  CD_GT_fused={np.mean([r['cd_vs_gt_fused_mm'] for r in rs]):.1f}mm"
                         f"  Δ={np.mean([r['cd_improvement_mm'] for r in rs]):+.1f}mm")
            print(line)

        if has_gt:
            best = min((r for r in all_results if "cd_vs_gt_fused_mm" in r),
                       key=lambda r: r["cd_vs_gt_fused_mm"], default=None)
            if best:
                print(f"\nBest: label={best['label']} seed={best['seed']} "
                      f"CD_GT={best['cd_vs_gt_fused_mm']:.1f}mm  "
                      f"(recon was {best['cd_vs_gt_recon_mm']:.1f}mm, "
                      f"Δ={best['cd_improvement_mm']:+.1f}mm)")

        print("\nFused meshes: <gen_root>/<label>/seed_<k>/mesh_fused.ply")
        print("Eval next: python tools/eval_object_mesh.py --gt_id <id> "
              "--pred_mesh <mesh_fused.ply> --gt_mesh <gt_mesh>")


if __name__ == "__main__":
    main()
