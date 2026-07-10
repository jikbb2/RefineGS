#!/usr/bin/env python3
"""STAGE 6/6.5 입력 v3 — weight를 목표 객체 실루엣과 교차 (focus<0.5 문제 수정).

soft_in(warp_gt_to_pose 출력)에서:
  nv_out     : gen=view(실픽셀), weight = known ∧ 객체실루엣   ← STAGE 6 감독을 객체에 집중
  see3d_out  : view 그대로,      weight = unobs ∧ 객체실루엣   ← 생성/감독 대상을 객체에 한정
뷰 채택 기준도 객체 기준으로: (known∧sil)/sil ≥ --min_known_obj → NV,
                             (unobs∧sil)/sil ≥ --min_unobs_obj → See3D.

  python make_stage6_inputs.py --soft_in ~/See3D/dataset/stage6/soft_in_v2 \
    --recon_root output/replica_room0_v2/refinegs_full \
    --nv_out ~/See3D/dataset/stage6/nv_v3 --see3d_out ~/See3D/dataset/stage6/see3d_in_v3

Deps: numpy, PIL, open3d, scipy. (stage6_qa.py와 같은 투영 로직)
"""
import os
import re
import glob
import shutil
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation


def target_from_pose(rec):
    wvt = np.asarray(rec["world_view_transform"], float)
    M = wvt.T
    R, t = M[:3, :3], M[:3, 3]
    W, H = int(rec["width"]), int(rec["height"])
    fx = W / (2 * np.tan(float(rec["FoVx"]) / 2))
    fy = H / (2 * np.tan(float(rec["FoVy"]) / 2))
    return R, t, fx, fy, W / 2.0, H / 2.0, W, H


def obj_silhouette(verts, rec, dil=6):
    R, t, fx, fy, cx, cy, W, H = target_from_pose(rec)
    pc = verts @ R.T + t
    z = pc[:, 2]
    ok = z > 1e-3
    u = np.clip(fx * pc[ok, 0] / z[ok] + cx, 0, W - 1).astype(np.int64)
    v = np.clip(fy * pc[ok, 1] / z[ok] + cy, 0, H - 1).astype(np.int64)
    m = np.zeros((H, W), bool)
    m[v, u] = True
    return binary_dilation(m, iterations=dil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_in", required=True)
    ap.add_argument("--recon_root", required=True)
    ap.add_argument("--nv_out", required=True)
    ap.add_argument("--see3d_out", required=True)
    ap.add_argument("--n_verts", type=int, default=30000)
    ap.add_argument("--min_known_obj", type=float, default=0.15,
                    help="객체 실루엣 중 known 비율 ≥ 이 값이면 NV 채택")
    ap.add_argument("--min_unobs_obj", type=float, default=0.10,
                    help="객체 실루엣 중 unobs 비율 ≥ 이 값이면 See3D 채택")
    args = ap.parse_args()

    soft = os.path.expanduser(args.soft_in)
    nv = os.path.expanduser(args.nv_out)
    sd = os.path.expanduser(args.see3d_out)
    for d in (nv, sd):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

    recs = {int(r["idx"]): (r.item() if hasattr(r, "item") and not isinstance(r, dict) else r)
            for r in np.load(os.path.join(soft, "poses.npz"), allow_pickle=True)["records"]}

    mesh_cache = {}
    def verts_of(gid):
        if gid not in mesh_cache:
            p = sorted(glob.glob(os.path.join(args.recon_root, gid, "train", "ours_*", "fuse_post.ply")))[-1]
            mesh_cache[gid] = np.asarray(
                o3d.io.read_triangle_mesh(p).sample_points_uniformly(args.n_verts).points)
        return mesh_cache[gid]

    n_nv = n_sd = 0
    for wp in sorted(glob.glob(soft + "/weight_*.png")):
        i = int(os.path.basename(wp)[7:11])
        rec = recs.get(i)
        if rec is None:
            continue
        m = re.match(r"g(.+?)_o", str(rec.get("stem", "")))
        if not m:
            continue
        gid = m.group(1)

        w = np.array(Image.open(wp).convert("L"))
        sil = obj_silhouette(verts_of(gid), rec)
        if sil.shape != w.shape:
            sil = np.array(Image.fromarray(sil).resize((w.shape[1], w.shape[0]), Image.NEAREST)) > 0
        known_o = (w == 255) & sil
        unobs_o = (w == 128) & sil
        s = max(sil.sum(), 1)
        r_known, r_unobs = known_o.sum() / s, unobs_o.sum() / s

        vp = os.path.join(soft, f"view_{i:04d}.jpg")
        if r_known >= args.min_known_obj:
            shutil.copy(vp, os.path.join(nv, f"gen_{i:04d}.jpg"))
            Image.fromarray((known_o * 255).astype(np.uint8)).save(
                os.path.join(nv, f"weight_{i:04d}.png"))
            n_nv += 1
        if r_unobs >= args.min_unobs_obj:
            shutil.copy(vp, os.path.join(sd, f"view_{i:04d}.jpg"))
            Image.fromarray((unobs_o * 255).astype(np.uint8)).save(
                os.path.join(sd, f"weight_{i:04d}.png"))
            n_sd += 1
        print(f"[{i:04d}] g{gid:>3}  obj-known {r_known:.2f}  obj-unobs {r_unobs:.2f}"
              f"  → NV {'O' if r_known>=args.min_known_obj else '-'}"
              f" / See3D {'O' if r_unobs>=args.min_unobs_obj else '-'}")

    shutil.copy(os.path.join(soft, "poses.npz"), os.path.join(nv, "poses.npz"))
    shutil.copy(os.path.join(soft, "poses.npz"), os.path.join(sd, "poses.npz"))
    print(f"\nNV {n_nv}뷰 → {nv}\nSee3D {n_sd}뷰 → {sd}")


if __name__ == "__main__":
    main()
