#!/usr/bin/env python3
"""
Registration (A-1, v2): Amodal3R canonical mesh → world-space TSDF recon.

v2 fixes (vs v1) — addresses mis-registration found in the stage4 diagnostics
(centroid offset 2-10cm, per-axis ext_ratio off, most gen verts >5cm from recon):

  1. ROTATION SELECTION BY CHAMFER, NOT ICP FITNESS.
     v1 picked the rotation candidate with the highest ICP `fitness`, but fitness
     = fraction of correspondences within a *generous* radius (icp_dist_eff up to
     15cm), so wrong rotations (e.g. a table rotated 90° → z_ratio 3.19) could
     win. v2 runs a coarse-to-fine ICP for every candidate and selects by tight
     symmetric Chamfer-L1 on the aligned surfaces.

  2. COARSE-TO-FINE ICP SCHEDULE scaled to the object size (fractions of the
     recon bounding-box diagonal) ending at a tight radius, so ICP can't settle
     in a loose local minimum.

  3. FINAL SCALED ICP (Umeyama, with_scaling=True) after the best rigid pose is
     fixed, to absorb residual *uniform* scale error (e.g. cushion x_ratio 0.80).
     Only accepted if it improves Chamfer.

  4. ext_ratio (per-axis bbox ratio gen/recon) reported for verification — should
     converge to ~1.0 on all three axes when registration is healthy.

Interface (CLI / batch / output keys) is unchanged so the rest of the pipeline
(fuse_generated_recon.py, eval_object_mesh.py) keeps working.

Usage (run from /home/elicer/RefineGS):
    conda activate split_and_splat
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
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def build_candidates():
    """32 rotation candidates covering common canonical ambiguities
    (y-up/z-up/x-up flips × 8 azimuths)."""
    R_yup_to_zup = rot_x(-np.pi / 2)
    R_yup_inv    = rot_x( np.pi / 2)
    candidates = []
    for az in range(0, 360, 45):
        R_az = rot_z(np.radians(az))
        candidates.append(R_az @ R_yup_to_zup)
        candidates.append(R_az @ R_yup_inv)
        candidates.append(R_az)
        candidates.append(R_az @ rot_y(np.pi / 2))
    return candidates


def _pcd(pts):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return p


def chamfer_l1(a_pts, b_pts, n_sample: int = 20000) -> float:
    """Fast symmetric Chamfer-L1 using open3d KDTree distances (subsampled)."""
    rng = np.random.default_rng(0)
    a = np.asarray(a_pts); b = np.asarray(b_pts)
    if len(a) > n_sample:
        a = a[rng.choice(len(a), n_sample, replace=False)]
    if len(b) > n_sample:
        b = b[rng.choice(len(b), n_sample, replace=False)]
    pa, pb = _pcd(a), _pcd(b)
    d_ab = np.asarray(pa.compute_point_cloud_distance(pb))
    d_ba = np.asarray(pb.compute_point_cloud_distance(pa))
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


def ext_ratio(gen_pts, recon_pts):
    """Per-axis bbox extent ratio gen/recon (≈1.0 each axis = good scale+rot)."""
    g = np.asarray(gen_pts); r = np.asarray(recon_pts)
    ge = g.max(0) - g.min(0)
    re = r.max(0) - r.min(0)
    return ge / np.maximum(re, 1e-9)


# ── ICP ─────────────────────────────────────────────────────────────────────

def _icp(src, tgt, init_T, dist, with_scaling=False, iters=60):
    return o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=dist,
        init=init_T,
        estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint(with_scaling),
        criteria=o3d.pipelines.registration.
            ICPConvergenceCriteria(max_iteration=iters))


def multi_res_icp(src, tgt, init_T, schedule, with_scaling_last=False):
    """Coarse-to-fine ICP over a list of decreasing correspondence distances."""
    T = init_T
    res = None
    for i, dist in enumerate(schedule):
        sc = with_scaling_last and (i == len(schedule) - 1)
        res = _icp(src, tgt, T, dist, with_scaling=sc)
        T = res.transformation
    return res


# ── single registration ──────────────────────────────────────────────────────

def register(gen_path: Path, recon_path: Path, out_path: Path,
             icp_dist: float = 0.15, verbose: bool = True):
    gen_mesh   = o3d.io.read_triangle_mesh(str(gen_path))
    recon_mesh = o3d.io.read_triangle_mesh(str(recon_path))

    gv = np.asarray(gen_mesh.vertices)
    rv = np.asarray(recon_mesh.vertices)
    if len(gv) == 0 or len(rv) == 0:
        print(f"  [ERROR] empty mesh: gen={len(gv)}, recon={len(rv)}")
        return None

    # ── 1. uniform pre-scale by bbox diagonal ratio ───────────────────────────
    gen_diag   = np.linalg.norm(gv.max(0) - gv.min(0))
    recon_diag = np.linalg.norm(rv.max(0) - rv.min(0))
    scale = recon_diag / gen_diag if gen_diag > 1e-6 else 1.0

    gv_scaled     = gv * scale
    gen_centroid  = gv_scaled.mean(0)
    recon_centroid = rv.mean(0)

    # coarse-to-fine schedule scaled to object size (fractions of recon diagonal)
    schedule = [f * recon_diag for f in (0.15, 0.06, 0.025)]
    schedule = [max(d, 0.01) for d in schedule]

    # Chamfer before (centroid-aligned only)
    cd_before = chamfer_l1(gv_scaled - gen_centroid + recon_centroid, rv)

    # downsampled target for ICP speed (full rv kept for final metrics)
    voxel = max(recon_diag * 0.01, 0.005)
    tgt_full = _pcd(rv)
    tgt_icp  = tgt_full.voxel_down_sample(voxel)

    # ── 2. rotation search, selected by Chamfer (not fitness) ─────────────────
    candidates = build_candidates()
    best = {"cd": 1e9, "rot_idx": -1, "fitness": 0.0, "rmse": 1e9,
            "aligned": None}

    for idx, R in enumerate(candidates):
        gv_rot = (R @ (gv_scaled - gen_centroid).T).T + recon_centroid
        src_full = _pcd(gv_rot)
        src_icp  = src_full.voxel_down_sample(voxel)

        res = multi_res_icp(src_icp, tgt_icp, np.eye(4), schedule,
                            with_scaling_last=False)

        src_aligned = _pcd(gv_rot)
        src_aligned.transform(res.transformation)
        gv_aligned = np.asarray(src_aligned.points)
        cd = chamfer_l1(gv_aligned, rv)

        if cd < best["cd"]:
            best.update(cd=cd, rot_idx=idx, fitness=res.fitness,
                        rmse=res.inlier_rmse, aligned=gv_aligned)

    # ── 3. final scaled-ICP refinement (absorb residual uniform scale) ────────
    src_ref = _pcd(best["aligned"])
    src_ref_icp = src_ref.voxel_down_sample(voxel)
    res_s = multi_res_icp(src_ref_icp, tgt_icp, np.eye(4),
                          schedule[-2:], with_scaling_last=True)
    src_ref.transform(res_s.transformation)
    gv_scaled_aligned = np.asarray(src_ref.points)
    cd_scaled = chamfer_l1(gv_scaled_aligned, rv)
    used_scaled = cd_scaled < best["cd"]
    final_aligned = gv_scaled_aligned if used_scaled else best["aligned"]
    cd_after = min(cd_scaled, best["cd"])

    # ── 4. save aligned mesh (rebuild from final vertex positions) ────────────
    gen_mesh_aligned = o3d.geometry.TriangleMesh(gen_mesh)
    gen_mesh_aligned.vertices = o3d.utility.Vector3dVector(final_aligned)
    gen_mesh_aligned.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out_path), gen_mesh_aligned)

    er = ext_ratio(final_aligned, rv)
    result_dict = {
        "scale": round(float(scale), 4),
        "rot_candidate": best["rot_idx"],
        "icp_fitness": round(float(best["fitness"]), 4),
        "icp_rmse_m": round(float(best["rmse"]), 4),
        "chamfer_l1_before_m": round(float(cd_before), 4),
        "chamfer_l1_after_m":  round(float(cd_after), 4),
        "improvement_pct": round((1 - cd_after / cd_before) * 100, 1)
                           if cd_before > 0 else 0,
        "ext_ratio": [round(float(x), 3) for x in er],
        "scaled_icp_used": bool(used_scaled),
    }

    if verbose:
        print(f"  scale={scale:.3f}  rot_idx={best['rot_idx']}  "
              f"sched={[round(s,3) for s in schedule]}")
        print(f"  ICP  fitness={best['fitness']:.3f}  RMSE={best['rmse']*1000:.1f}mm")
        print(f"  ext_ratio (x,y,z) = ({er[0]:.2f}, {er[1]:.2f}, {er[2]:.2f})"
              f"   scaled_icp={'yes' if used_scaled else 'no'}")
        print(f"  Chamfer before={cd_before*1000:.1f}mm  after={cd_after*1000:.1f}mm  "
              f"({result_dict['improvement_pct']:+.1f}%)")
        print(f"  → {out_path}")

    return result_dict


# ── batch ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--gen",   type=Path)
    ap.add_argument("--recon", type=Path)
    ap.add_argument("--out",   type=Path)
    ap.add_argument("--gen_root",   default="~/Amodal3R/poc_output", type=Path)
    ap.add_argument("--recon_root", default="output/replica_room0/axis3_sweep/reg_strong", type=Path)
    ap.add_argument("--labels", nargs="+", default=["97", "98", "75"])
    ap.add_argument("--seeds",  nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--icp_dist", type=float, default=0.15,
                    help="(kept for CLI compat; schedule is now size-relative)")
    args = ap.parse_args()

    if not args.batch:
        r = register(args.gen, args.recon,
                     args.out or args.gen.parent / "mesh_registered.ply",
                     icp_dist=args.icp_dist)
        if r:
            print(json.dumps(r, indent=2))
        return

    gen_root   = Path(str(args.gen_root)).expanduser()
    recon_root = Path(str(args.recon_root))
    all_results = []

    print(f"\n{'label':>6} {'seed':>5} {'scale':>6} {'fit':>6} {'RMSE(mm)':>9} "
          f"{'CD_bef':>8} {'CD_aft':>8} {'improv%':>8} {'ext_ratio(x,y,z)':>22} {'sc':>3}")
    print("-" * 100)

    for label in args.labels:
        iters = "7000" if "reg_strong" in str(recon_root) else "*"
        recon_glob = list(recon_root.glob(f"{label}/train/ours_{iters}/fuse_post.ply"))
        if not recon_glob:
            recon_glob = list(recon_root.glob(f"{label}/train/ours_*/fuse_post.ply"))
        if not recon_glob:
            print(f"  [SKIP] label {label}: no recon mesh under {recon_root}/{label}/")
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
                er = r["ext_ratio"]
                print(f"{label:>6} {seed:>5} {r['scale']:>6.2f} "
                      f"{r['icp_fitness']:>6.3f} {r['icp_rmse_m']*1000:>9.1f} "
                      f"{r['chamfer_l1_before_m']*1000:>8.1f} "
                      f"{r['chamfer_l1_after_m']*1000:>8.1f} "
                      f"{r['improvement_pct']:>7.1f}% "
                      f"({er[0]:>5.2f},{er[1]:>5.2f},{er[2]:>5.2f}) "
                      f"{'Y' if r['scaled_icp_used'] else '-':>3}")
                all_results.append({"label": label, "seed": seed, **r})

    if all_results:
        print("\n── per-label summary (mean over seeds) ──")
        for label in args.labels:
            rs = [r for r in all_results if r["label"] == label]
            if not rs:
                continue
            mean_er = np.mean([r["ext_ratio"] for r in rs], axis=0)
            print(f"  label {label}: "
                  f"CD_before={1000*np.mean([r['chamfer_l1_before_m'] for r in rs]):.1f}mm  "
                  f"CD_after={1000*np.mean([r['chamfer_l1_after_m'] for r in rs]):.1f}mm  "
                  f"ext_ratio=({mean_er[0]:.2f},{mean_er[1]:.2f},{mean_er[2]:.2f})")

        best_seed = min(all_results, key=lambda r: r["chamfer_l1_after_m"])
        print(f"\nBest overall: label={best_seed['label']} seed={best_seed['seed']} "
              f"CD={best_seed['chamfer_l1_after_m']*1000:.1f}mm")
        print("\nRegistered meshes → <gen_root>/<label>/seed_<k>/mesh_registered.ply")
        print("Verify: ext_ratio should be ~1.0 on all axes; re-run diag_registration.py.")
        print("Next: fuse_generated_recon.py --method append (occ_thresh now meaningful).")


if __name__ == "__main__":
    main()
