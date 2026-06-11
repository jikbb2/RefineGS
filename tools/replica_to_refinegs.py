#!/usr/bin/env python3
#
# RefineGS - tools/replica_to_refinegs.py
# ---------------------------------------------------------------------------
# NICE-SLAM Replica (GT pose + GT depth) → RefineGS(Split&Splat) 입력 포맷 변환
#
# Replica(nice-slam) 입력:
#   <scene>/results/frameXXXXXX.jpg     RGB (1200x680)
#   <scene>/results/depthXXXXXX.png     uint16 depth (meters = png / 6553.5)
#   <scene>/traj.txt                    줄당 4x4 c2w (camera-to-world), row-major 16값
#   intrinsics: fx=fy=600, cx=599.5, cy=339.5
#
# 출력 (data/<out_name>/):
#   images/frameXXXXXX.jpg              (심볼릭 링크, subsample 적용)
#   depth/frameXXXXXX_pred.npy          float32 meters (mask_propagation 이 읽는 이름)
#   sparse/0/cameras.txt                PINHOLE 1200 680 600 600 599.5 339.5
#   sparse/0/images.txt                 per-frame world→cam qvec+tvec (COLMAP 규약)
#   sparse/0/points3D.txt               (빈 헤더 — pycolmap 로드용)
#   sparse/0/points3D.ply               dense 점군 (GT depth 역투영, world+RGB) = ① 결정
#
# 이유: mask_propagation.py 가 sparse/0 를 pycolmap 으로 읽어 포즈를 얻고,
#       dataset_readers 는 points3D.ply 로 GS init. Replica 는 GT 라 SfM/스케일정렬 불필요.
# ---------------------------------------------------------------------------

import os
import argparse
import numpy as np
from PIL import Image

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False


def rotmat2qvec(R):
    """COLMAP 규약 qvec=(qw,qx,qy,qz). (colmap_loader 와 동일)"""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def load_traj(path):
    """traj.txt → list of 4x4 c2w (camera-to-world)."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.array([float(v) for v in line.split()], dtype=np.float64)
            assert vals.size == 16, f"traj 줄당 16값이어야 함 (got {vals.size})"
            poses.append(vals.reshape(4, 4))
    return poses


def write_ply(path, xyz, rgb):
    """xyz (N,3) float, rgb (N,3) uint8 → binary PLY."""
    if HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float64) / 255.0))
        o3d.io.write_point_cloud(path, pcd)
        return
    # open3d 없을 때 fallback (plyfile)
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attrs = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def main():
    ap = argparse.ArgumentParser(description="Replica(nice-slam) → RefineGS 변환")
    ap.add_argument("--replica_scene", required=True, help="예: /path/Replica/room0")
    ap.add_argument("--out_dir", required=True, help="예: ./data/replica_room0")
    ap.add_argument("--subsample", type=int, default=10, help="N프레임마다 1개")
    ap.add_argument("--fx", type=float, default=600.0)
    ap.add_argument("--fy", type=float, default=600.0)
    ap.add_argument("--cx", type=float, default=599.5)
    ap.add_argument("--cy", type=float, default=339.5)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=680)
    ap.add_argument("--depth_scale", type=float, default=6553.5, help="png/scale = meters")
    ap.add_argument("--pixel_stride", type=int, default=8, help="점군용 픽셀 stride")
    ap.add_argument("--voxel", type=float, default=0.02, help="점군 voxel downsample (m)")
    ap.add_argument("--max_points", type=int, default=1_500_000, help="points3D.ply 최대 점")
    ap.add_argument("--colmap_max_points", type=int, default=300_000,
                    help="points3D.txt 최대 점 (pycolmap/mask_propagation 이 읽음)")
    ap.add_argument("--depth_trunc", type=float, default=12.0, help="이보다 먼 depth 무시(m)")
    ap.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    results = os.path.join(args.replica_scene, "results")
    traj_path = os.path.join(args.replica_scene, "traj.txt")
    assert os.path.isdir(results), f"results/ 없음: {results}"
    assert os.path.exists(traj_path), f"traj.txt 없음: {traj_path}"

    poses = load_traj(traj_path)
    n_total = len(poses)
    sel = list(range(0, n_total, args.subsample))
    print(f"총 {n_total} 프레임 중 {len(sel)} 선택 (subsample={args.subsample})")

    images_dir = os.path.join(args.out_dir, "images")
    depth_dir = os.path.join(args.out_dir, "depth")
    sparse_dir = os.path.join(args.out_dir, "sparse", "0")
    for d in (images_dir, depth_dir, sparse_dir):
        os.makedirs(d, exist_ok=True)

    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    W, H = args.width, args.height

    # ---- cameras.txt ----
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # 점군 누적 (픽셀 그리드 미리 계산)
    us = np.arange(0, W, args.pixel_stride)
    vs = np.arange(0, H, args.pixel_stride)
    uu, vv = np.meshgrid(us, vs)  # (h',w')
    uu_f = uu.reshape(-1).astype(np.float64)
    vv_f = vv.reshape(-1).astype(np.float64)

    all_xyz = []
    all_rgb = []

    img_lines = []
    for out_idx, i in enumerate(sel, start=1):
        stem = f"frame{i:06d}"
        rgb_src = os.path.join(results, stem + ".jpg")
        depth_src = os.path.join(results, "depth%06d.png" % i)
        if not os.path.exists(rgb_src) or not os.path.exists(depth_src):
            print(f"  [skip] 누락: {stem}")
            continue

        c2w = poses[i]
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        qvec = rotmat2qvec(R)
        # images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME  +  빈 줄(points2D)
        img_lines.append(
            f"{out_idx} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
            f"{t[0]} {t[1]} {t[2]} 1 {stem}.jpg\n\n")

        # 이미지 링크/복사
        dst_img = os.path.join(images_dir, stem + ".jpg")
        if not os.path.lexists(dst_img):
            if args.link_mode == "symlink":
                os.symlink(os.path.abspath(rgb_src), dst_img)
            else:
                from shutil import copyfile
                copyfile(rgb_src, dst_img)

        # depth → meters → _pred.npy
        depth_png = np.array(Image.open(depth_src)).astype(np.float32)
        depth_m = depth_png / args.depth_scale
        np.save(os.path.join(depth_dir, stem + "_pred.npy"), depth_m)

        # dense 점군 누적 (GT depth 역투영)
        rgb = np.array(Image.open(rgb_src).convert("RGB"))
        d = depth_m[vv.reshape(-1), uu.reshape(-1)]
        valid = (d > 0) & (d < args.depth_trunc)
        if valid.sum() == 0:
            continue
        dv = d[valid]
        xc = (uu_f[valid] - cx) / fx * dv
        yc = (vv_f[valid] - cy) / fy * dv
        zc = dv
        Xcam = np.stack([xc, yc, zc], axis=1)                  # (M,3) OpenCV
        Xworld = (c2w[:3, :3] @ Xcam.T).T + c2w[:3, 3]
        cols = rgb[vv.reshape(-1)[valid], uu.reshape(-1)[valid]]
        all_xyz.append(Xworld.astype(np.float32))
        all_rgb.append(cols.astype(np.uint8))

    # images.txt 기록
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for line in img_lines:
            f.write(line)

    # ---- 최종 dense 점군 (voxel 다운샘플) ----
    xyz = np.concatenate(all_xyz, axis=0).astype(np.float64)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)
    print(f"역투영 점 수(raw): {xyz.shape[0]}")
    if HAS_O3D and args.voxel > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
        pcd = pcd.voxel_down_sample(args.voxel)
        xyz = np.asarray(pcd.points)
        rgb = np.clip(np.asarray(pcd.colors) * 255.0, 0, 255).astype(np.uint8)
        print(f"voxel({args.voxel}m) 다운샘플 후: {xyz.shape[0]} 점")
    if xyz.shape[0] > args.max_points:
        idx = np.random.choice(xyz.shape[0], args.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    # points3D.ply (GS init, dense)
    write_ply(os.path.join(sparse_dir, "points3D.ply"), xyz.astype(np.float32), rgb)
    print(f"points3D.ply: {xyz.shape[0]} 점")

    # points3D.txt — ★ pycolmap 이 읽어 recon.points3D 구성 (mask_propagation 의 sparse_pcd).
    #   비워두면 recon.points3D 가 0개 → o3d Vector3dVector(빈배열) RuntimeError 발생.
    n_txt = min(xyz.shape[0], args.colmap_max_points)
    sub = (np.random.choice(xyz.shape[0], n_txt, replace=False)
           if xyz.shape[0] > n_txt else np.arange(xyz.shape[0]))
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for k, j in enumerate(sub, start=1):
            x, y, z = xyz[j]
            r, g, b = rgb[j]
            f.write(f"{k} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0\n")
    print(f"points3D.txt: {n_txt} 점 (pycolmap 로드용)")

    print(f"\n변환 완료 → {args.out_dir}")
    print("다음: auto_seg.py --scene <out_name> → mask_propagation.py → prepare_folder → smoke_test")


if __name__ == "__main__":
    main()
