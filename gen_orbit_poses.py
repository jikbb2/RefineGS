#!/usr/bin/env python3
"""STAGE 6 (dense) — 약점 객체 중심 orbit 포즈 생성 → poses.npz (모델 로드 불필요).

각 gid의 fuse_post.ply 중심/반경으로 orbit 카메라를 합성하고, occluder mesh(raycast)로
벽 뒤/가려진 포즈를 제거. 출력 poses.npz 는 warp_gt_to_pose.py --poses 로 직결.

  python gen_orbit_poses.py \
    --recon_root output/replica_room0_v2/refinegs_full \
    --gids 20,6,2,5,22,31,0 \
    --model_cams output/replica_room0_v2/scene_whole_dense_reg/cameras.json \
    --occluder output/replica_room0_v2/scene_mono_reg/train/ours_30000/fuse_post.ply \
    --n_per_obj 16 --elev_min -10 --elev_max 40 --radius_scale 2.5 --up_axis 2 \
    --out ~/See3D/dataset/stage6/poses.npz

RefineGS repo 루트에서 실행 (utils.graphics_utils 사용).
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
import open3d as o3d
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_root", required=True)
    ap.add_argument("--gids", required=True, help="콤마구분 대상 객체")
    ap.add_argument("--model_cams", required=True, help="intrinsic 템플릿용 cameras.json")
    ap.add_argument("--occluder", default="", help="free-space 필터 mesh (예: mono fuse_post)")
    ap.add_argument("--n_per_obj", type=int, default=16)
    ap.add_argument("--elev_min", type=float, default=-10)
    ap.add_argument("--elev_max", type=float, default=40)
    ap.add_argument("--radius_scale", type=float, default=2.5, help="객체 반경 대비 orbit 반경 배율")
    ap.add_argument("--radius_min", type=float, default=0.7)
    ap.add_argument("--radius_max", type=float, default=2.2)
    ap.add_argument("--up_axis", type=int, default=2)
    ap.add_argument("--min_clear", type=float, default=0.25,
                    help="카메라 위치가 occluder 표면에서 이 거리(m) 미만이면 reject (내부 박힘 방지)")
    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # intrinsic 템플릿 (첫 카메라)
    cam0 = json.load(open(args.model_cams))[0]
    W, H = int(cam0["width"]), int(cam0["height"])
    FoVx = 2 * np.arctan(W / (2 * cam0["fx"]))
    FoVy = 2 * np.arctan(H / (2 * cam0["fy"]))
    proj = getProjectionMatrix(args.znear, args.zfar, FoVx, FoVy).transpose(0, 1)

    rc, occ_tree, occ_bb = None, None, None
    if args.occluder and os.path.exists(args.occluder):
        m = o3d.io.read_triangle_mesh(args.occluder)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        from scipy.spatial import cKDTree
        vv = np.asarray(m.vertices)
        occ_tree = cKDTree(vv)                        # 카메라가 가구/벽 '내부'에 박히는 것 방지
        occ_bb = (vv.min(0) + 0.3, vv.max(0) - 0.3)   # 방 내부 bbox (벽 밖 포즈 차단)
        print(f"free-space 필터: {args.occluder}")
        print(f"방 내부 bbox: {np.round(occ_bb[0],2)} ~ {np.round(occ_bb[1],2)}")

    up = np.zeros(3); up[args.up_axis] = 1.0
    a0, a1 = [j for j in range(3) if j != args.up_axis]

    records, idx = [], 0
    for gid in args.gids.split(","):
        gid = gid.strip()
        mp = sorted(glob.glob(os.path.join(args.recon_root, gid, "train", "ours_*", "fuse_post.ply")))
        if not mp:
            print(f"[skip] gid {gid}: fuse_post 없음"); continue
        mesh = o3d.io.read_triangle_mesh(mp[-1])
        v = np.asarray(mesh.vertices)
        ctr = (v.max(0) + v.min(0)) / 2
        obj_r = float(np.linalg.norm(v.max(0) - v.min(0)) / 2)
        rad = float(np.clip(obj_r * args.radius_scale, args.radius_min, args.radius_max))

        kept = 0
        for i in range(args.n_per_obj):
            f = i / max(args.n_per_obj - 1, 1)
            az = np.deg2rad(360.0 * f)
            el = np.deg2rad(args.elev_min + (args.elev_max - args.elev_min) * f)
            offs = np.zeros(3)
            offs[a0] = np.cos(el) * np.cos(az)
            offs[a1] = np.cos(el) * np.sin(az)
            offs[args.up_axis] = np.sin(el)
            pos = ctr + rad * offs

            if occ_bb is not None:    # 방 밖 포즈 reject (벽 뒤에서 known=0 되는 원인 차단)
                if np.any(pos < occ_bb[0]) or np.any(pos > occ_bb[1]):
                    continue
            if occ_tree is not None:  # 카메라 위치가 메쉬 표면 25cm 이내(내부/밀착) → reject
                if occ_tree.query(pos)[0] < args.min_clear:
                    continue
            if rc is not None:  # 카메라→객체 중심이 막히면 reject
                d = ctr - pos; dist = np.linalg.norm(d)
                ray = o3d.core.Tensor([[*pos.astype(np.float32), *(d / dist).astype(np.float32)]],
                                      dtype=o3d.core.Dtype.Float32)
                t_hit = float(rc.cast_rays(ray)["t_hit"].numpy()[0])
                if np.isfinite(t_hit) and t_hit < dist - obj_r * 1.2:
                    continue

            fwd = ctr - pos; fwd /= np.linalg.norm(fwd) + 1e-9
            right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-9
            down = np.cross(fwd, right)
            Rc2w = np.stack([right, down, fwd], axis=1)
            t = -Rc2w.T @ pos
            wvt = torch.tensor(getWorld2View2(Rc2w, t)).transpose(0, 1).float()
            fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            records.append(dict(idx=idx, stem=f"g{gid}_o{i:02d}",
                                world_view_transform=wvt.numpy(),
                                full_proj_transform=fpt.numpy(),
                                FoVx=float(FoVx), FoVy=float(FoVy), width=W, height=H))
            idx += 1; kept += 1
        print(f"gid {gid}: center {np.round(ctr,2)} r={obj_r:.2f} orbit_r={rad:.2f} → {kept}/{args.n_per_obj} pose")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, records=np.array(records, dtype=object))
    print(f"\n총 {len(records)} pose → {out}")


if __name__ == "__main__":
    main()
