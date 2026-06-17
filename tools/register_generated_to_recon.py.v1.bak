#!/usr/bin/env python3
"""
Registration PoC (A-1): Amodal3R canonical mesh → world-space TSDF recon.

Steps:
  1. Scale normalization (diagonal extent ratio)
  2. Centroid alignment
  3. Rotation search: 8 candidates (y-up→z-up flip × 4 rotations around vertical)
  4. ICP refinement for each candidate → pick best by fitness+RMSE
  5. Save aligned mesh, report Chamfer before/after

Usage (run from /home/elicer/RefineGS with either conda env):
    conda activate split_and_splat
    python tools/register_generated_to_recon.py \
        --gen  ~/Amodal3R/poc_output/75/seed_1/mesh.ply \
        --recon output/replica_room0/axis3_sweep/reg_strong/75/train/ours_7000/fuse_post.ply \
        --out  ~/Amodal3R/poc_output/75/seed_1/mesh_registered.ply

    # batch over all instances and seeds:
    python tools/register_generated_to_recon.py --batch \
        --gen_root  ~/Amodal3R/poc_output \
        --recon_root output/replica_room0/axis3_sweep/reg_strong \
        --labels 97 98 75 --seeds 1 2 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


# ── geometry helpers ──────────────────────────────────────────────────────────

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def build_candidates():
    """32 rotation candidates covering the most common canonical ambiguities.

    Amodal3R may output objects in y-up or z-up canonical frame.
    We try:
      - 8 azimuths (0..315° in 45° steps) × y-up→z-up flip  = 16
      - 8 azimuths × no flip (already z-up)                  = 16
    Total = 32 candidates.
    """
    R_yup_to_zup = rot_x(-np.pi / 2)   # y→z: rotate -90° around X
    R_yup_inv    = rot_x( np.pi / 2)   # y→-z variant
    candidates = []
    for az in range(0, 360, 45):
        R_az = rot_z(np.radians(az))
        candidates.append(R_az @ R_yup_to_zup)   # with y-up flip
        candidates.append(R_az @ R_yup_inv)       # with inverted flip
        candidates.append(R_az)                   # no flip
        candidates.append(R_az @ rot_y(np.pi/2)) # x-up→z-up
    return candidates


def chamfer_l1(a: np.ndarray, b: np.ndarray, n_sample: int = 10000) -> float:
    """Approximate Chamfer-L1 between two point sets (random subsample)."""
    rng = np.random.default_rng(0)
    a_ = a[rng.choice(len(a), min(n_sample, len(a)), replace=False)]
    b_ = b[rng.choice(len(b), min(n_sample, len(b)), replace=False)]

    # a → b
    diff_ab = a_[:, None, :] - b_[None, :, :]  # (Na, Nb, 3)
    dist_ab = np.linalg.norm(diff_ab, axis=2)
    acc = dist_ab.min(axis=1).mean()

    # b → a
    dist_ba = dist_ab.min(axis=0).mean()
    return float((acc + dist_ba) / 2)


# ── single registration ───────────────────────────────────────────────────────

def fpfh_global_register(src_pcd, tgt_pcd, voxel: float = 0.05):
    """FPFH-based global registration (RANSAC). Falls back to None on error."""
    try:
        src_d = src_pcd.voxel_down_sample(voxel)
        tgt_d = tgt_pcd.voxel_down_sample(voxel)
        src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel*2, 30))
        tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel*2, 30))
        src_f = o3d.pipelines.registration.compute_fpfh_feature(
            src_d, o3d.geometry.KDTreeSearchParamHybrid(voxel*5, 100))
        tgt_f = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_d, o3d.geometry.KDTreeSearchParamHybrid(voxel*5, 100))
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_d, tgt_d, src_f, tgt_f,
            mutual_filter=True,
            max_correspondence_distance=voxel*1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel*1.5),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result
    except Exception:
        return None


def icp_refine(src_pcd, tgt_pcd, init_T, dist):
    """Multi-resolution ICP: coarse pass (2× dist) then fine pass (dist)."""
    r1 = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=dist * 2,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    r2 = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=dist,
        init=r1.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return r2


def register(gen_path: Path, recon_path: Path, out_path: Path,
             icp_dist: float = 0.15, verbose: bool = True):
    gen_mesh   = o3d.io.read_triangle_mesh(str(gen_path))
    recon_mesh = o3d.io.read_triangle_mesh(str(recon_path))

    gv = np.asarray(gen_mesh.vertices)
    rv = np.asarray(recon_mesh.vertices)

    if len(gv) == 0 or len(rv) == 0:
        print(f"  [ERROR] empty mesh: gen={len(gv)}, recon={len(rv)}")
        return None

    # ── 1. Scale ──────────────────────────────────────────────────────────────
    gen_diag  = np.linalg.norm(gv.max(0) - gv.min(0))
    recon_diag = np.linalg.norm(rv.max(0) - rv.min(0))
    scale = recon_diag / gen_diag if gen_diag > 1e-6 else 1.0

    gv_scaled = gv * scale
    gen_centroid  = gv_scaled.mean(0)
    recon_centroid = rv.mean(0)
    trans_init = recon_centroid - gen_centroid

    # adaptive ICP distance: larger objects need bigger correspondence radius
    icp_dist_eff = max(icp_dist, scale * 0.12)

    # Chamfer before registration
    gv_init = gv_scaled + trans_init
    cd_before = chamfer_l1(gv_init, rv)

    # ── 2-3. Rotation search + multi-res ICP ─────────────────────────────────
    candidates = build_candidates()
    best = {"fitness": -1, "rmse": 1e9, "T": np.eye(4), "cd": 1e9, "rot_idx": -1}

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(rv)

    for idx, R in enumerate(candidates):
        # apply rotation around centroid, then translate to recon centroid
        gv_rot = (R @ (gv_scaled - gen_centroid).T).T + recon_centroid

        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(gv_rot)

        result = icp_refine(src_pcd, tgt_pcd, np.eye(4), icp_dist_eff)

        if result.fitness > best["fitness"] or (
                result.fitness == best["fitness"] and result.inlier_rmse < best["rmse"]):
            best["fitness"]  = result.fitness
            best["rmse"]     = result.inlier_rmse
            best["T"]        = result.transformation
            best["rot_idx"]  = idx
            src_pcd_final = o3d.geometry.PointCloud(src_pcd)
            src_pcd_final.transform(result.transformation)
            best["gv_aligned"] = np.asarray(src_pcd_final.points)

    # ── 3b. FPFH global registration fallback (if fitness < 0.5) ─────────────
    if best["fitness"] < 0.5:
        voxel = icp_dist_eff * 0.5
        # use best rotation candidate as starting point for FPFH
        R_best_so_far = candidates[best["rot_idx"]]
        gv_rot = (R_best_so_far @ (gv_scaled - gen_centroid).T).T + recon_centroid
        src_pcd_fpfh = o3d.geometry.PointCloud()
        src_pcd_fpfh.points = o3d.utility.Vector3dVector(gv_rot)

        fpfh_result = fpfh_global_register(src_pcd_fpfh, tgt_pcd, voxel=voxel)
        if fpfh_result is not None and fpfh_result.fitness > best["fitness"]:
            # refine with ICP
            result2 = icp_refine(src_pcd_fpfh, tgt_pcd,
                                  fpfh_result.transformation, icp_dist_eff)
            if result2.fitness > best["fitness"]:
                best["fitness"] = result2.fitness
                best["rmse"]    = result2.inlier_rmse
                best["T"]       = result2.transformation
                best["rot_idx"] = best["rot_idx"]  # keep rot_idx (FPFH refines on top)
                src_pcd_fpfh.transform(result2.transformation)
                best["gv_aligned"] = np.asarray(src_pcd_fpfh.points)
                if verbose:
                    print(f"  [FPFH] improved fitness: {result2.fitness:.3f}")

    # ── 4. Save aligned mesh ──────────────────────────────────────────────────
    R_best = candidates[best["rot_idx"]]
    gv_rot_final = (R_best @ (gv_scaled - gen_centroid).T).T + recon_centroid
    gen_mesh_aligned = o3d.geometry.TriangleMesh(gen_mesh)
    gen_mesh_aligned.vertices = o3d.utility.Vector3dVector(gv_rot_final)
    gen_mesh_aligned.transform(best["T"])
    o3d.io.write_triangle_mesh(str(out_path), gen_mesh_aligned)

    cd_after = chamfer_l1(best["gv_aligned"], rv)

    result_dict = {
        "scale": round(scale, 4),
        "rot_candidate": best["rot_idx"],
        "icp_fitness": round(best["fitness"], 4),
        "icp_rmse_m": round(best["rmse"], 4),
        "chamfer_l1_before_m": round(cd_before, 4),
        "chamfer_l1_after_m":  round(cd_after, 4),
        "improvement_pct": round((1 - cd_after / cd_before) * 100, 1) if cd_before > 0 else 0,
    }

    if verbose:
        print(f"  scale={scale:.3f}  icp_dist_eff={icp_dist_eff:.3f}m  rot_idx={best['rot_idx']}")
        print(f"  ICP  fitness={best['fitness']:.3f}  RMSE={best['rmse']*1000:.1f}mm")
        print(f"  Chamfer before={cd_before*1000:.1f}mm  after={cd_after*1000:.1f}mm  "
              f"({result_dict['improvement_pct']:+.1f}%)")
        print(f"  → {out_path}")

    return result_dict


# ── batch ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true")
    # single mode
    ap.add_argument("--gen",   type=Path)
    ap.add_argument("--recon", type=Path)
    ap.add_argument("--out",   type=Path)
    # batch mode
    ap.add_argument("--gen_root",   default="~/Amodal3R/poc_output", type=Path)
    ap.add_argument("--recon_root", default="output/replica_room0/axis3_sweep/reg_strong", type=Path)
    ap.add_argument("--labels", nargs="+", default=["97","98","75"])
    ap.add_argument("--seeds",  nargs="+", type=int, default=[1,2,3])
    ap.add_argument("--icp_dist", type=float, default=0.15,
                    help="ICP max correspondence distance (m)")
    args = ap.parse_args()

    if not args.batch:
        r = register(args.gen, args.recon,
                     args.out or args.gen.parent / "mesh_registered.ply",
                     icp_dist=args.icp_dist)
        if r:
            print(json.dumps(r, indent=2))
        return

    # batch
    gen_root   = Path(args.gen_root).expanduser()
    recon_root = Path(args.recon_root)
    all_results = []

    print(f"\n{'label':>6} {'seed':>5} {'scale':>6} {'fit':>6} "
          f"{'RMSE(mm)':>9} {'CD_bef(mm)':>11} {'CD_aft(mm)':>11} {'improv%':>8}")
    print("-" * 75)

    for label in args.labels:
        iters = "7000" if "reg_strong" in str(recon_root) else "*"
        recon_glob = list(recon_root.glob(
            f"{label}/train/ours_{iters}/fuse_post.ply"))
        if not recon_glob:
            recon_glob = list(recon_root.glob(
                f"{label}/train/ours_*/fuse_post.ply"))
        if not recon_glob:
            print(f"  [SKIP] label {label}: no recon mesh found under {recon_root}/{label}/")
            continue
        recon_path = recon_glob[0]

        for seed in args.seeds:
            gen_path = gen_root / label / f"seed_{seed}" / "mesh.ply"
            out_path = gen_root / label / f"seed_{seed}" / "mesh_registered.ply"
            if not gen_path.exists():
                print(f"  [SKIP] {label}/seed{seed}: gen mesh not found")
                continue

            r = register(gen_path, recon_path, out_path,
                         icp_dist=args.icp_dist, verbose=False)
            if r:
                print(f"{label:>6} {seed:>5} {r['scale']:>6.2f} "
                      f"{r['icp_fitness']:>6.3f} {r['icp_rmse_m']*1000:>9.1f} "
                      f"{r['chamfer_l1_before_m']*1000:>11.1f} "
                      f"{r['chamfer_l1_after_m']*1000:>11.1f} "
                      f"{r['improvement_pct']:>7.1f}%")
                all_results.append({"label": label, "seed": seed, **r})

    # summary
    if all_results:
        print("\n── per-label summary (mean over seeds) ──")
        for label in args.labels:
            rs = [r for r in all_results if r["label"] == label]
            if not rs: continue
            print(f"  label {label}: "
                  f"CD_before={1000*np.mean([r['chamfer_l1_before_m'] for r in rs]):.1f}mm  "
                  f"CD_after={1000*np.mean([r['chamfer_l1_after_m'] for r in rs]):.1f}mm  "
                  f"fitness={np.mean([r['icp_fitness'] for r in rs]):.3f}")

        best_seed = min(all_results, key=lambda r: r["chamfer_l1_after_m"])
        print(f"\nBest overall: label={best_seed['label']} seed={best_seed['seed']} "
              f"CD={best_seed['chamfer_l1_after_m']*1000:.1f}mm")
        print("\nRegistered meshes saved to: <gen_root>/<label>/seed_<k>/mesh_registered.ply")
        print("Next: screened Poisson fusion with free-space veto (A-2)")


if __name__ == "__main__":
    main()
