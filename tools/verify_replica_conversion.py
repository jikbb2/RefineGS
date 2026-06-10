#!/usr/bin/env python3
#
# RefineGS - tools/verify_replica_conversion.py
# ---------------------------------------------------------------------------
# replica_to_refinegs.py 출력 검증:
#   1) sparse/0 (cameras.txt/images.txt/points3D.ply) 존재·개수
#   2) (선택) pycolmap 로 로드되는지 — mask_propagation 이 이 경로를 씀
#   3) 한 뷰에 points3D.ply 재투영 → RGB 위 오버레이 저장 (포즈/intrinsic/점군 정합 시각 확인)
#
# 사용:
#   python tools/verify_replica_conversion.py --data ./data/replica_room0 [--frame 0]
#   → data/replica_room0/_verify_reproj.png 생성. 점들이 장면 구조와 겹치면 정합 OK.
# ---------------------------------------------------------------------------

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])


def read_cameras_txt(path):
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            e = line.split()
            # CAMERA_ID MODEL W H fx fy cx cy
            fx, fy, cx, cy = map(float, e[4:8])
            W, H = int(e[2]), int(e[3])
            return fx, fy, cx, cy, W, H
    raise RuntimeError("no camera in cameras.txt")


def read_images_txt(path):
    """반환: list of (name, qvec(4), tvec(3)) — pose 줄만(토큰 10개)."""
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            e = line.split()
            if len(e) >= 10 and e[0].isdigit() and e[9].count(".") <= 1 and not e[1].replace(".", "").replace("-", "").isalpha():
                try:
                    qvec = np.array(list(map(float, e[1:5])))
                    tvec = np.array(list(map(float, e[5:8])))
                    name = e[9]
                    out.append((name, qvec, tvec))
                except ValueError:
                    continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="예: ./data/replica_room0")
    ap.add_argument("--frame", type=int, default=0, help="검증할 프레임 인덱스(선택된 것 중 N번째)")
    ap.add_argument("--n_points", type=int, default=8000, help="오버레이에 찍을 점 수")
    args = ap.parse_args()

    sparse = os.path.join(args.data, "sparse", "0")
    ply = os.path.join(sparse, "points3D.ply")
    cams = os.path.join(sparse, "cameras.txt")
    imgs = os.path.join(sparse, "images.txt")

    print("== 파일 존재 ==")
    for p in (cams, imgs, ply):
        print(f"  {'OK ' if os.path.exists(p) else 'MISSING'} {p}")

    fx, fy, cx, cy, W, H = read_cameras_txt(cams)
    print(f"intrinsics: fx={fx} fy={fy} cx={cx} cy={cy} ({W}x{H})")

    images = read_images_txt(imgs)
    print(f"#images(poses): {len(images)}")

    # points3D.ply 로드
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply)
        xyz = np.asarray(pcd.points)
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    except Exception:
        from plyfile import PlyData
        pl = PlyData.read(ply)['vertex']
        xyz = np.vstack([pl['x'], pl['y'], pl['z']]).T
        rgb = None
    print(f"#points: {xyz.shape[0]}, bbox min={xyz.min(0).round(2)} max={xyz.max(0).round(2)}")

    # (선택) pycolmap 로드 확인 — mask_propagation 이 이 경로를 사용
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(sparse)
        print(f"[pycolmap] OK — images={recon.num_images()} cameras={recon.num_cameras()} points={recon.num_points3D()}")
    except Exception as e:
        print(f"[pycolmap] 로드 실패: {e}")
        print("  → 'colmap model_converter --input_path {0} --output_path {0} --output_type BIN' 로 bin 변환 시도".format(sparse))

    # 재투영 오버레이
    name, qvec, tvec = images[args.frame]
    R = qvec2rotmat(qvec)                      # world→cam
    Xc = (R @ xyz.T).T + tvec                  # (N,3)
    z = Xc[:, 2]
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    vis = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    print(f"frame '{name}': 가시 점 {vis.sum()}/{xyz.shape[0]}  ({100*vis.mean():.1f}%)")

    img_path = os.path.join(args.data, "images", name)
    if not os.path.exists(img_path):  # 확장자 다르면 stem 으로 탐색
        stem = os.path.splitext(name)[0]
        import glob
        hits = glob.glob(os.path.join(args.data, "images", stem + ".*"))
        img_path = hits[0] if hits else None
    bg = np.array(Image.open(img_path).convert("RGB")) if img_path else np.zeros((H, W, 3), np.uint8)

    uu, vv = u[vis], v[vis]
    cc = (rgb[vis] / 255.0) if rgb is not None else 'lime'
    if len(uu) > args.n_points:
        sub = np.random.choice(len(uu), args.n_points, replace=False)
        uu, vv = uu[sub], vv[sub]
        cc = cc[sub] if rgb is not None else cc
    plt.figure(figsize=(12, 7))
    plt.imshow(bg)
    plt.scatter(uu, vv, s=2, c=cc, marker='.')
    plt.title(f"reproject {name} | visible {vis.sum()} pts")
    plt.axis('off')
    out_png = os.path.join(args.data, "_verify_reproj.png")
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"\n오버레이 저장 → {out_png}")
    print("점들이 RGB의 장면 구조(가구·벽 등)와 겹쳐 보이면 포즈/intrinsic/점군 정합 OK ✅")
    print("점이 어긋나거나 한쪽에 뭉치면 → convention(예: c2w/w2c) 의심")


if __name__ == "__main__":
    main()
