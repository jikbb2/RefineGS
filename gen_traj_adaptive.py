#!/usr/bin/env python3
"""객체별 적응 점진 궤적 — 관측 콘에서 출발해 unseen으로 하강 (known=0 시작 문제 해결).

각 gid에 대해:
  1) masks/<gid>/images 의 프레임 stem → COLMAP 카메라 위치 = 그 객체의 실제 관측 방향/거리
  2) 관측 평균 방향(az0, el0)·중앙값 거리에서 궤적 시작 (앵커 프레임: known 보장)
  3) azimuth 는 az0 부터 sweep, elevation 은 el0 → elev_min 으로 점진 하강 → unseen 진입
  프레임 순서 = 체인 순서 (See3D chunk carry 앵커와 정합)

  python gen_traj_adaptive.py \
    --recon_root output/replica_room0_v2/refinegs_full \
    --masks_root data/replica_room0_v2/masks \
    --colmap data/replica_room0_v2/sparse/0 \
    --gids 0,1,10,11,12,14,15,16,17,18,19,2,20,22,23,24,27,28,3,31,32,34,35,36,37,38,4,5,6,7,8 \
    --model_cams output/replica_room0_v2/scene_whole_dense_reg/cameras.json \
    --occluder output/replica_room0_v2/scene_mono_reg/train/ours_30000/fuse_post.ply \
    --n_anchor 3 --n_desc 12 --elev_min -10 --sweep_az 300 \
    --out ~/See3D/dataset/stage6/poses_adapt.npz

RefineGS repo 루트에서 실행. Deps: numpy, torch, open3d, scipy + repo utils.
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
import open3d as o3d
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from warp_gt_to_pose import read_colmap, cam_center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_root", required=True)
    ap.add_argument("--masks_root", required=True)
    ap.add_argument("--stems_dir", default="", help="clean_stems.py 출력 폴더 (gid.txt) — 관측 방향 계산에 정화 목록 사용")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--gids", required=True)
    ap.add_argument("--model_cams", required=True)
    ap.add_argument("--occluder", default="")
    ap.add_argument("--n_anchor", type=int, default=3, help="관측 콘 안 앵커 프레임 수(known 보장)")
    ap.add_argument("--n_desc", type=int, default=12, help="하강 프레임 수(unseen 진입)")
    ap.add_argument("--elev_min", type=float, default=-10)
    ap.add_argument("--sweep_az", type=float, default=300, help="하강 중 azimuth 총 선회각(도)")
    ap.add_argument("--radius_min", type=float, default=0.7)
    ap.add_argument("--radius_max", type=float, default=1.6)
    ap.add_argument("--min_clear", type=float, default=0.25)
    ap.add_argument("--up_axis", type=int, default=2)
    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cam0 = json.load(open(args.model_cams))[0]
    W, H = int(cam0["width"]), int(cam0["height"])
    FoVx = 2 * np.arctan(W / (2 * cam0["fx"]))
    FoVy = 2 * np.arctan(H / (2 * cam0["fy"]))
    proj = getProjectionMatrix(args.znear, args.zfar, FoVx, FoVy).transpose(0, 1)

    cams = read_colmap(args.colmap)
    center_of = {c["stem"]: cam_center(c["R"], c["t"]) for c in cams}
    cam_of = {c["stem"]: c for c in cams}

    rc, occ_tree, occ_bb = None, None, None
    if args.occluder and os.path.exists(args.occluder):
        m = o3d.io.read_triangle_mesh(args.occluder)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        from scipy.spatial import cKDTree
        vv = np.asarray(m.vertices)
        occ_tree = cKDTree(vv)
        occ_bb = (vv.min(0) + 0.3, vv.max(0) - 0.3)
        print(f"필터: free-space + min_clear + 방 bbox {np.round(occ_bb[0],2)}~{np.round(occ_bb[1],2)}")

    up = np.zeros(3); up[args.up_axis] = 1.0
    a0, a1 = [j for j in range(3) if j != args.up_axis]

    records, idx = [], 0
    for gid in args.gids.split(","):
        gid = gid.strip()
        mp = sorted(glob.glob(os.path.join(args.recon_root, gid, "train", "ours_*", "fuse_post.ply")))
        if not mp:
            print(f"[skip] gid {gid}: fuse_post 없음"); continue
        v = np.asarray(o3d.io.read_triangle_mesh(mp[-1]).vertices)
        ctr = (v.max(0) + v.min(0)) / 2
        obj_r = float(np.linalg.norm(v.max(0) - v.min(0)) / 2)

        # 이 객체를 관측한 실측 카메라들 → 관측 방향/거리
        # --stems_dir 가 있으면 3D-일관 정화 목록(clean_stems.py) 사용 — 오염 프레임 배제
        sf = os.path.join(os.path.expanduser(args.stems_dir), f"{gid}.txt") if args.stems_dir else ""
        if sf and os.path.exists(sf):
            stems = [ln.strip() for ln in open(sf) if ln.strip()]
        else:
            stems = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(args.masks_root, gid, "images", "*"))]
        obs = np.array([center_of[s] for s in stems if s in center_of])
        if len(obs) < 3:
            print(f"[skip] gid {gid}: 관측 카메라 매칭 {len(obs)}개"); continue
        dirs = obs - ctr
        dist = np.linalg.norm(dirs, axis=1)
        dn = dirs / dist[:, None]
        mdir = dn.mean(0); mdir /= np.linalg.norm(mdir) + 1e-9
        el0 = np.degrees(np.arcsin(np.clip(mdir[args.up_axis], -1, 1)))
        az0 = np.degrees(np.arctan2(mdir[a1], mdir[a0]))
        rad = float(np.clip(np.median(dist), args.radius_min, args.radius_max))

        # ── 실측 앵커: 정화 stem 중 균등 K장의 '실제 카메라 포즈' 그대로 (known 보장) ──
        kept = 0
        ks = [round(j * (len(stems) - 1) / max(args.n_anchor - 1, 1)) for j in range(args.n_anchor)]
        anchor_pos = None
        for j, k in enumerate(ks):
            c = cam_of.get(stems[k])
            if c is None:
                continue
            M = np.eye(4); M[:3, :3] = c["R"]; M[:3, 3] = c["t"]   # colmap w2c
            wvt = torch.tensor(M.T, dtype=torch.float32)            # 저장 규약 = w2c^T
            fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            records.append(dict(idx=idx, stem=f"g{gid}_o{j:02d}",
                                world_view_transform=wvt.numpy(),
                                full_proj_transform=fpt.numpy(),
                                FoVx=float(FoVx), FoVy=float(FoVy), width=W, height=H))
            idx += 1; kept += 1
            anchor_pos = center_of[stems[k]]

        # ── 하강 시작 방향: 마지막 실측 앵커의 위치 기준 (메쉬 중심 오류에 강건) ──
        if anchor_pos is not None:
            dv = anchor_pos - ctr; dvn = dv / (np.linalg.norm(dv) + 1e-9)
            el0 = np.degrees(np.arcsin(np.clip(dvn[args.up_axis], -1, 1)))
            az0 = np.degrees(np.arctan2(dvn[a1], dvn[a0]))

        for i in range(args.n_desc):     # 하강: el0 → elev_min, azimuth 선회
            f = i / max(args.n_desc - 1, 1)
            az = az0 + args.sweep_az * f
            el = el0 + (args.elev_min - el0) * f
            azr, elr = np.deg2rad(az), np.deg2rad(el)
            offs = np.zeros(3)
            offs[a0] = np.cos(elr) * np.cos(azr)
            offs[a1] = np.cos(elr) * np.sin(azr)
            offs[args.up_axis] = np.sin(elr)
            pos = ctr + rad * offs

            if occ_bb is not None and (np.any(pos < occ_bb[0]) or np.any(pos > occ_bb[1])):
                continue
            if occ_tree is not None and occ_tree.query(pos)[0] < args.min_clear:
                continue
            if rc is not None:
                d = ctr - pos; dd = np.linalg.norm(d)
                ray = o3d.core.Tensor([[*pos.astype(np.float32), *(d / dd).astype(np.float32)]],
                                      dtype=o3d.core.Dtype.Float32)
                t_hit = float(rc.cast_rays(ray)["t_hit"].numpy()[0])
                if np.isfinite(t_hit) and t_hit < dd - obj_r * 1.2:
                    continue

            fwd = ctr - pos; fwd /= np.linalg.norm(fwd) + 1e-9
            right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-9
            down = np.cross(fwd, right)
            Rc2w = np.stack([right, down, fwd], axis=1)
            t = -Rc2w.T @ pos
            wvt = torch.tensor(getWorld2View2(Rc2w, t)).transpose(0, 1).float()
            fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            records.append(dict(idx=idx, stem=f"g{gid}_o{args.n_anchor + i:02d}",
                                world_view_transform=wvt.numpy(),
                                full_proj_transform=fpt.numpy(),
                                FoVx=float(FoVx), FoVy=float(FoVy), width=W, height=H))
            idx += 1; kept += 1
        print(f"gid {gid:>3}: obs {len(obs)}뷰  az0 {az0:6.1f}°  el0 {el0:5.1f}°  r={rad:.2f} → "
              f"{kept}/{args.n_anchor + args.n_desc} (실측앵커 {min(args.n_anchor, kept)})")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, records=np.array(records, dtype=object))
    print(f"\n총 {len(records)} pose → {out}")


if __name__ == "__main__":
    main()
