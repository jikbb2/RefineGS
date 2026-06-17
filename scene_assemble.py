#!/usr/bin/env python3
"""
Scene assembly — per-object 결과를 하나의 scene으로 합침 (직접 concat).

모든 per-object 모델이 *같은 scene COLMAP(sparse/0)* 으로 학습돼 world 좌표계를 공유하므로,
객체별 Gaussian(point_cloud.ply) / mesh(fuse_completed|fuse_post.ply)를 그대로 이어붙이면 scene이 됨.
객체별로 최신 iteration / 완성본을 선택. 끝난 객체만(존재하는 것만) 합치므로 부분 실행에도 동작.

출력:
  scene_gaussians.ply  — SuperSplat 등에서 볼 수 있는 합쳐진 Gaussian scene
  scene_mesh.ply       — 합쳐진 geometry mesh

의존: numpy, plyfile(Gaussian), trimesh(mesh). split_and_splat env.

실행:
    conda activate split_and_splat
    python scene_assemble.py --root output/replica_room0_v2/refinegs_full \
        --out_gauss output/replica_room0_v2/scene_gaussians.ply \
        --out_mesh  output/replica_room0_v2/scene_mesh.ply
"""
import argparse, glob, os, re
import numpy as np


def latest_iter_ply(gid_dir):
    cands = glob.glob(os.path.join(gid_dir, "point_cloud", "iteration_*", "point_cloud.ply"))
    if not cands:
        return None
    return max(cands, key=lambda p: int(re.search(r"iteration_(\d+)", p).group(1)))


def best_mesh(gid_dir):
    its = glob.glob(os.path.join(gid_dir, "train", "ours_*"))
    if not its:
        return None
    it = max(its, key=lambda p: int(re.search(r"ours_(\d+)", p).group(1)))
    for name in ("fuse_completed.ply", "fuse_post.ply"):
        f = os.path.join(it, name)
        if os.path.exists(f):
            return f
    return None


def assemble_gaussians(gid_dirs, out):
    from plyfile import PlyData, PlyElement
    parts = []
    for d in gid_dirs:
        f = latest_iter_ply(d)
        if not f:
            continue
        parts.append((os.path.basename(d), PlyData.read(f)["vertex"].data, f))
    if not parts:
        print("  [gaussians] none found"); return
    base_dt = parts[0][1].dtype
    keep = [p for p in parts if p[1].dtype == base_dt]
    if len(keep) != len(parts):
        print(f"  [gaussians] WARN: {len(parts)-len(keep)} objects skipped (different property schema)")
    merged = np.concatenate([p[1] for p in keep])
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(out)
    print(f"  [gaussians] {len(keep)} objects, {len(merged)} splats → {out}")


def assemble_mesh(gid_dirs, out):
    import trimesh
    ms = []
    for d in gid_dirs:
        f = best_mesh(d)
        if not f:
            continue
        m = trimesh.load(f, process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        if len(m.vertices):
            ms.append(m)
    if not ms:
        print("  [mesh] none found"); return
    scene = trimesh.util.concatenate(ms)
    scene.export(out)
    print(f"  [mesh] {len(ms)} objects, {len(scene.vertices)} verts → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="refinegs_full (per-object 출력 루트)")
    ap.add_argument("--out_gauss", default=None)
    ap.add_argument("--out_mesh", default=None)
    args = ap.parse_args()
    gid_dirs = sorted([d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)])
    print(f"object dirs: {len(gid_dirs)}")
    if args.out_gauss:
        os.makedirs(os.path.dirname(args.out_gauss) or ".", exist_ok=True)
        print("== assemble gaussians =="); assemble_gaussians(gid_dirs, args.out_gauss)
    if args.out_mesh:
        os.makedirs(os.path.dirname(args.out_mesh) or ".", exist_ok=True)
        print("== assemble mesh =="); assemble_mesh(gid_dirs, args.out_mesh)
    if not (args.out_gauss or args.out_mesh):
        print("--out_gauss / --out_mesh 중 하나 이상 지정")
    print("\nGaussian scene은 SuperSplat에서, mesh scene은 MeshLab/미리보기에서 확인.")


if __name__ == "__main__":
    main()
