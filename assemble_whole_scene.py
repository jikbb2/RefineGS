#!/usr/bin/env python3
"""RefineGS — 전체 씬 조립 (모든 객체 recon + Amodal3R gen + base).

원래 계획: per-object Amodal3R gen → mesh_to_surfels → base ⊕ 모든 recon ⊕ 모든 gen surfel 합성
          → 안 보이는 곳 See3D → 전체 씬 joint 학습.
이 스크립트는 그 *합치는 과정*(Gaussian 레벨)을 수행한다.

각 객체:
  recon = output/<scene>/refinegs_fix/<gid>/point_cloud/iteration_*/point_cloud.ply
  gen   = ~/Amodal3R/poc_output/<gid>/seed_1/mesh_registered_clean.ply  → mesh_to_surfels → surfel
  base 의 객체 bbox(=recon extent+pad) 안을 carve(중복 테이블 방지) 후 recon·gen 삽입.

tag(id_0): base=0, 모든 recon=1, 모든 gen=2  (See3D hole 은 gen=2 ∧ ¬obs).

실행:
  python assemble_whole_scene.py --base output/replica_room0/scene_base/point_cloud/iteration_30000/point_cloud.ply \
    --scene replica_room0_v2 --recon_root output/replica_room0_v2/refinegs_fix \
    --gen_root ~/Amodal3R/poc_output --pad 0.05 \
    --out output/replica_room0_v2/scene_whole/point_cloud/iteration_1/point_cloud.ply
  # --gids "1,8,24,27" 로 객체 한정 가능(기본: recon+gen 둘 다 있는 모든 gid 자동)

Deps: numpy, plyfile. (mesh_to_surfels.py 를 subprocess 로 호출)
"""
import argparse, os, glob, sys, subprocess, tempfile
import numpy as np
from plyfile import PlyData, PlyElement


def load(path):
    v = PlyData.read(path)["vertex"]
    return np.asarray(v.data), [p.name for p in v.properties]


def xyz(data):
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--scene", default="replica_room0_v2")
    ap.add_argument("--recon_root", required=True)
    ap.add_argument("--gen_root", default=os.path.expanduser("~/Amodal3R/poc_output"))
    ap.add_argument("--gids", default="", help="콤마구분. 비우면 recon+gen 둘 다 있는 gid 자동")
    ap.add_argument("--mesh_to_surfels", default="mesh_to_surfels.py")
    ap.add_argument("--n_samples", type=int, default=200000)
    ap.add_argument("--pad", type=float, default=0.05, help="(미사용 — proximity carve로 대체)")
    ap.add_argument("--carve_dist", type=float, default=0.04,
                    help="base 점이 recon 표면에서 이 거리(m) 이내면 제거(중복 객체 표면만). bbox carve 대체")
    ap.add_argument("--max_extent", type=float, default=3.0,
                    help="recon 최대 extent가 이보다 크면 broken(whole-scene)으로 보고 제외")
    ap.add_argument("--tmp", default=os.path.expanduser("~/tmp"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)

    # 대상 gid
    if a.gids.strip():
        gids = [g.strip() for g in a.gids.split(",")]
    else:
        gids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.gen_root, "*"))
                      if os.path.isdir(p))

    base, bprops = load(a.base)
    base_xyz = xyz(base)
    recon_xyz_all = []                       # proximity carve용
    recon_chunks, gen_chunks, used = [], [], []

    for gid in gids:
        recon_g = sorted(glob.glob(os.path.join(a.recon_root, gid, "point_cloud/iteration_*/point_cloud.ply")))
        mesh_g = sorted(glob.glob(os.path.join(a.recon_root, gid, "train/ours_*/fuse_post.ply")))
        gen_mesh = os.path.join(a.gen_root, gid, "seed_1", "mesh_registered_clean.ply")
        if not recon_g or not mesh_g or not os.path.exists(gen_mesh):
            print(f"[skip] gid {gid}: recon={bool(recon_g)} mesh={bool(mesh_g)} gen={os.path.exists(gen_mesh)}")
            continue
        recon, rprops = load(recon_g[-1])
        if set(rprops) != set(bprops):
            print(f"[skip] gid {gid}: recon 스키마 불일치"); continue
        # 메쉬(fuse_post) bbox 로 broken 판정 + recon 점군 crop(floater 제거)
        mdata, _ = load(mesh_g[-1]); mxyz = xyz(mdata)
        mext = mxyz.max(0) - mxyz.min(0)
        if mext.max() > a.max_extent:
            print(f"[skip] gid {gid}: mesh extent {mext.round(2)} > {a.max_extent} (broken)"); continue
        lo_c, hi_c = mxyz.min(0) - a.pad, mxyz.max(0) + a.pad
        rkeep = np.all((xyz(recon) >= lo_c) & (xyz(recon) <= hi_c), axis=1)
        recon = recon[rkeep]      # 메쉬 bbox 밖 floater 제거
        # gen mesh → surfel
        surfel = os.path.join(a.tmp, f"gensurf_{gid}.ply")
        r = subprocess.run([sys.executable, a.mesh_to_surfels, "--mesh", gen_mesh,
                            "--out", surfel, "--n_samples", str(a.n_samples)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(surfel):
            print(f"[skip] gid {gid}: mesh_to_surfels 실패\n{r.stderr[-300:]}"); continue
        gen, gprops = load(surfel)
        if set(gprops) != set(bprops):
            print(f"[skip] gid {gid}: gen 스키마 불일치"); continue
        recon_xyz_all.append(xyz(recon))         # crop된 recon
        recon_chunks.append(recon.astype(base.dtype, copy=True))
        gen_chunks.append(gen.astype(base.dtype, copy=True))
        used.append(gid)
        print(f"[ok] gid {gid}: recon {len(recon)} (mesh ext {mext.round(2)}) + gen {len(gen)}")

    if not used:
        raise SystemExit("조립할 객체 0 — recon/gen 경로 확인")

    # proximity carve: base 점이 recon 표면 가까이(<carve_dist)면 제거(중복 객체 표면만, 바닥/벽 보존)
    from scipy.spatial import cKDTree
    rall = np.concatenate(recon_xyz_all)
    print(f"proximity carve: recon {len(rall)} 점 KDTree, base {len(base_xyz)} 질의...")
    tree = cKDTree(rall)
    d, _ = tree.query(base_xyz, k=1, distance_upper_bound=a.carve_dist*4, workers=-1)
    carve = d < a.carve_dist
    base_keep = base[~carve].copy()
    if "id_0" in bprops:
        base_keep["id_0"] = 0
        for c in recon_chunks: c["id_0"] = 1
        for c in gen_chunks:   c["id_0"] = 2

    merged = np.concatenate([base_keep] + recon_chunks + gen_chunks)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(a.out)
    print(f"\n조립: 객체 {len(used)}개 {used}")
    print(f"  proximity carve: {int(carve.sum())} 제거 → base {len(base_keep)} (바닥/벽 보존)")
    print(f"  + recon {sum(len(c) for c in recon_chunks)} (tag1) + gen {sum(len(c) for c in gen_chunks)} (tag2)")
    print(f"  = {len(merged)} → {a.out}")
    print("검수: 렌더에서 각 객체가 *하나씩*만(중복 없이) 보이면 carve 성공.")


if __name__ == "__main__":
    main()
