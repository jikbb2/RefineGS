#!/usr/bin/env python3
"""
diag_registration.py — Quantify alignment quality of registered Amodal3R meshes
vs the TSDF recon, to decide whether bad fusion results are caused by
mis-registration (alignment) rather than by the generated geometry itself.

For each label/seed it reports, comparing recon <-> mesh_registered.ply:
  - centroid offset (mm)          : systematic translation error
  - bbox extent ratio (gen/recon) : scale mismatch  (should be ~1.0 per axis)
  - gen->recon NN distance        : how far the generated *visible* surface sits
                                    from recon. If registration is good, a large
                                    fraction of gen verts (the front/visible part
                                    overlapping recon) should be within ~1-2 cm.
  - recon->gen NN distance        : reverse coverage.
  - %within {1,2,5}cm             : the key registration health numbers.

Interpretation:
  * High overlap (e.g. >60% of gen within 2cm) + centroid offset < ~1cm
    + extent ratio ~1.0  -> registration is fine; bad fusion is the generated
    geometry's fault (object/back-side hallucination).
  * Low overlap (most gen verts >5cm from recon) + big centroid offset or
    extent ratio far from 1.0 -> registration is broken; fix alignment first.

No open3d dependency required (uses trimesh + scipy cKDTree). Headless-safe.
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_verts(path: Path) -> np.ndarray:
    m = trimesh.load(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return np.asarray(m.vertices, dtype=np.float64)


def sample_surface(path: Path, n: int) -> np.ndarray:
    """Sample points on the surface for distance stats (area-weighted)."""
    m = trimesh.load(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    if len(m.faces) == 0:
        return np.asarray(m.vertices, dtype=np.float64)
    pts, _ = trimesh.sample.sample_surface(m, n)
    return np.asarray(pts, dtype=np.float64)


def stats(d):
    d = np.asarray(d)
    return dict(
        median_mm=float(np.median(d) * 1000),
        mean_mm=float(np.mean(d) * 1000),
        p90_mm=float(np.percentile(d, 90) * 1000),
        within_1cm=float((d < 0.01).mean() * 100),
        within_2cm=float((d < 0.02).mean() * 100),
        within_5cm=float((d < 0.05).mean() * 100),
    )


def diagnose(recon_path: Path, gen_path: Path, n: int = 80000):
    r_v = load_verts(recon_path)
    g_v = load_verts(gen_path)
    r_pts = sample_surface(recon_path, n)
    g_pts = sample_surface(gen_path, n)

    # centroid offset (use surface samples for robustness)
    centroid_off = np.linalg.norm(r_pts.mean(0) - g_pts.mean(0)) * 1000  # mm

    # bbox extent ratio per axis
    r_ext = r_pts.max(0) - r_pts.min(0)
    g_ext = g_pts.max(0) - g_pts.min(0)
    ext_ratio = g_ext / np.maximum(r_ext, 1e-9)

    # NN distances
    r_tree = cKDTree(r_pts)
    g_tree = cKDTree(g_pts)
    d_g2r, _ = r_tree.query(g_pts)   # gen -> recon
    d_r2g, _ = g_tree.query(r_pts)   # recon -> gen

    return dict(
        centroid_off_mm=centroid_off,
        ext_ratio=ext_ratio,
        n_recon_v=len(r_v),
        n_gen_v=len(g_v),
        g2r=stats(d_g2r),
        r2g=stats(d_r2g),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_root",
                    default="output/replica_room0/axis3_sweep/reg_strong", type=Path)
    ap.add_argument("--gen_root", default="~/Amodal3R/poc_output", type=Path)
    ap.add_argument("--labels", nargs="+", default=["97", "98"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--recon_name",
                    default="train/ours_7000/fuse_post.ply",
                    help="recon mesh path under <recon_root>/<label>/")
    ap.add_argument("--gen_name", default="mesh_registered.ply",
                    help="registered gen mesh under <gen_root>/<label>/seed_<k>/")
    ap.add_argument("--n", type=int, default=80000)
    args = ap.parse_args()

    gen_root = Path(str(args.gen_root)).expanduser()
    recon_root = Path(str(args.recon_root)).expanduser()

    print(f"{'label':>5} {'seed':>4} {'cOff_mm':>8} {'ext_ratio(x,y,z)':>22} "
          f"{'g2r_med':>8} {'g<1cm%':>7} {'g<2cm%':>7} {'g<5cm%':>7} "
          f"{'r2g_med':>8}")
    print("-" * 90)
    for label in args.labels:
        recon_path = recon_root / label / Path(args.recon_name)
        if not recon_path.exists():
            print(f"  [SKIP] {label}: recon not found at {recon_path}")
            continue
        for seed in args.seeds:
            gen_path = gen_root / label / f"seed_{seed}" / args.gen_name
            if not gen_path.exists():
                print(f"  [SKIP] {label}/seed{seed}: {gen_path} not found")
                continue
            d = diagnose(recon_path, gen_path, args.n)
            er = d["ext_ratio"]
            print(f"{label:>5} {seed:>4} {d['centroid_off_mm']:>8.1f} "
                  f"({er[0]:>5.2f},{er[1]:>5.2f},{er[2]:>5.2f})    "
                  f"{d['g2r']['median_mm']:>8.1f} "
                  f"{d['g2r']['within_1cm']:>7.1f} "
                  f"{d['g2r']['within_2cm']:>7.1f} "
                  f"{d['g2r']['within_5cm']:>7.1f} "
                  f"{d['r2g']['median_mm']:>8.1f}")
    print("-" * 90)
    print("Healthy registration ~  cOff<10mm, ext_ratio~1.0/axis, g<2cm% high (>50).")
    print("Broken registration ~  big cOff, ext_ratio off, most gen verts >5cm (g<2cm% low).")


if __name__ == "__main__":
    main()
