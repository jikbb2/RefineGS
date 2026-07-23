#!/usr/bin/env python3
"""VGGT joint pointmap 정합 — {실측 GT 프레임 + SEVA 생성 뷰}을 한 forward 로 공동 추론.

단안 depth 의 뷰별 독립 추정(scale-shift 불일치) 문제를 VGGT 의 공동 추론으로 해소.
좌표 정렬: VGGT 추정 카메라(실측 프레임) ↔ 우리 COLMAP 포즈 로 Sim(3) → world 로 이동.
필터: depth_conf + unknown 볼륨 근접 + TSDF 표면 밖 → prior 점군.

  conda activate vggt
  python make_vggt_points.py \
    --scene_dir ~/prior/seva/obj6 \
    --samples <seva_repo>/work_dirs/demo/img2trajvid/obj6/samples-rgb \
    --colmap data/replica_room0_v2/sparse/0 \
    --tsdf output/replica_room0_v2/refinegs_full/6/train/ours_7000/fuse_post.ply \
    --unknown ~/prior/obj6/unknown.ply \
    --out ~/prior/obj6/vggt_points.ply

전제: scene_dir/transforms.json 의 train frames = 실측(파일이 실제 GT), test frames = 생성 pose,
      samples/NNN.png = test 순서 생성 이미지.
Deps: vggt, torch, open3d, numpy, PIL, (repo) warp_gt_to_pose.
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from warp_gt_to_pose import read_colmap, cam_center


def umeyama_sim3(src, dst):
    """src → dst Sim(3) (R,s,t). src,dst: (N,3) 대응점."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    s = (sig * [1, 1, d]).sum() / (S ** 2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return R, s, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--tsdf", required=True)
    ap.add_argument("--unknown", required=True)
    ap.add_argument("--conf_thr", type=float, default=2.0, help="depth_conf 이 값 미만 픽셀 제외")
    ap.add_argument("--near_unknown", type=float, default=0.12)
    ap.add_argument("--surf_band", type=float, default=0.03)
    ap.add_argument("--max_pts", type=int, default=200000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sdir = os.path.expanduser(args.scene_dir)
    tr = json.load(open(os.path.join(sdir, "transforms.json")))
    spl = json.load(open(sorted(glob.glob(os.path.join(sdir, "train_test_split_*.json")))[0]))
    train_ids, test_ids = spl["train_ids"], spl["test_ids"]
    frames = tr["frames"]

    # 배치 이미지 목록: 실측(train) 실제 파일 + 생성(test) samples
    samp = sorted(glob.glob(os.path.join(os.path.expanduser(args.samples), "*.png")))
    assert samp, f"생성 이미지 없음: {args.samples}"
    img_paths, is_real = [], []
    for i in train_ids:
        img_paths.append(os.path.join(sdir, frames[i]["file_path"])); is_real.append(True)
    for k, i in enumerate(test_ids[:len(samp)]):
        img_paths.append(samp[k]); is_real.append(False)
    is_real = np.array(is_real)
    print(f"VGGT 입력: 실측 {is_real.sum()} + 생성 {(~is_real).sum()} = {len(img_paths)}뷰")

    # ── VGGT 추론 ──
    dev = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev)
    images = load_and_preprocess_images(img_paths).to(dev)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        im = images[None]
        toks, ps = model.aggregator(im)
        pose_enc = model.camera_head(toks)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, im.shape[-2:])
        depth, depth_conf = model.depth_head(toks, im, ps)
    extr = extr.squeeze(0).float().cpu().numpy()      # (V,3,4) world→cam (OpenCV)
    intr = intr.squeeze(0).float().cpu().numpy()
    depth = depth.squeeze(0).float().cpu().numpy()    # (V,H,W,1) or (V,H,W)
    dconf = depth_conf.squeeze(0).float().cpu().numpy()
    pmap = unproject_depth_map_to_point_map(depth, extr, intr)   # (V,H,W,3) VGGT 좌표
    print(f"VGGT depth {depth.shape}, conf 범위 [{dconf.min():.2f},{dconf.max():.2f}]")

    # ── Sim(3) 정렬: VGGT 실측 카메라중심 ↔ COLMAP 실측 카메라중심 ──
    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    src, dst = [], []
    for bi, real in enumerate(is_real):
        if not real:
            continue
        stem = os.path.splitext(os.path.basename(frames[train_ids[np.sum(is_real[:bi+1])-1]]["file_path"]))[0] \
            if False else os.path.splitext(os.path.basename(img_paths[bi]))[0]
        # VGGT 카메라 중심(world→cam 역): C = -R^T t
        R, t = extr[bi, :3, :3], extr[bi, :3, 3]
        src.append(-R.T @ t)
        c = cams.get(stem)
        assert c is not None, f"COLMAP 에 실측 프레임 없음: {stem}"
        dst.append(cam_center(c["R"], c["t"]))
    src, dst = np.stack(src), np.stack(dst)
    Rs, s, ts = umeyama_sim3(src, dst)
    err = np.linalg.norm((s * (src @ Rs.T) + ts) - dst, axis=1)
    print(f"Sim(3) 정렬: 실측 {len(src)}점, scale {s:.3f}, 잔차 {err.mean()*1000:.1f}mm")

    # ── 생성 뷰 점군 → world, 필터 ──
    tm = o3d.io.read_triangle_mesh(os.path.expanduser(args.tsdf))
    tsdf_tree = cKDTree(np.asarray(tm.vertices))
    uk_tree = cKDTree(np.asarray(o3d.io.read_point_cloud(os.path.expanduser(args.unknown)).points))

    P_all, C_all = [], []
    for bi, real in enumerate(is_real):
        if real:
            continue
        pm = pmap[bi].reshape(-1, 3)
        cf = dconf[bi].reshape(-1)
        rgb = np.asarray(Image.open(img_paths[bi]).convert("RGB").resize(
            (depth.shape[2], depth.shape[1]))).reshape(-1, 3) / 255.0
        ok = cf > args.conf_thr
        pm, rgb = pm[ok], rgb[ok]
        Pw = s * (pm @ Rs.T) + ts                      # VGGT → world
        d_uk, _ = uk_tree.query(Pw, workers=-1)
        d_ts, _ = tsdf_tree.query(Pw, workers=-1)
        keep = (d_uk < args.near_unknown) & (d_ts > args.surf_band)
        if keep.sum() == 0:
            continue
        P_all.append(Pw[keep]); C_all.append(rgb[keep])
    assert P_all, "필터 후 점 0 — conf_thr/near_unknown 완화"
    P = np.concatenate(P_all); C = np.concatenate(C_all)
    if len(P) > args.max_pts:
        sel = np.random.choice(len(P), args.max_pts, replace=False)
        P, C = P[sel], C[sel]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_point_cloud(out, pc)
    print(f"\nVGGT prior 점군 {len(P)} → {out}")
    print(f"  centroid {P.mean(0).round(3).tolist()}  extent {(P.max(0)-P.min(0)).round(3).tolist()}")


if __name__ == "__main__":
    main()
