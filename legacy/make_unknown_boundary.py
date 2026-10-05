#!/usr/bin/env python3
"""TSDF 메쉬의 열린 경계(boundary)에서 unknown 방향 추출 — SDF 불필요.

관측 메쉬(TSDF)의 열린 edge = 관측이 끊긴 지점. 그 경계 정점과 '표면이 이어질 방향'
(경계 edge 에 수직이고 삼각형 바깥쪽)이 곧 미관측이 시작되는 곳/방향.
free-space 검증으로 '이미 비어있음이 확인된' 방향은 제외.

  python make_unknown_boundary.py \
    --tsdf output/replica_room0_v2/refinegs_full/6/train/ours_7000/fuse_post.ply \
    --carve_depth_dir ~/carve_depth_mono \
    --out ~/prior/obj6/unknown.ply

출력: 경계 점군(법선 = 표면이 이어질 방향) → make_seva_scene 의 --unknown 입력.
Deps: numpy, open3d, scipy.
"""
import os
import glob
import argparse
import numpy as np
import open3d as o3d


def boundary_edges(faces):
    """열린 edge(한 삼각형에만 속함) 목록."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsdf", required=True)
    ap.add_argument("--carve_depth_dir", default="")
    ap.add_argument("--offset", type=float, default=0.03,
                    help="경계에서 이 거리(m) 바깥으로 unknown 점을 배치")
    ap.add_argument("--free_margin", type=float, default=0.05)
    ap.add_argument("--min_pts", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = o3d.io.read_triangle_mesh(os.path.expanduser(args.tsdf))
    m.compute_vertex_normals()
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    VN = np.asarray(m.vertex_normals)
    assert len(F) > 0, "빈 메쉬"

    be = boundary_edges(F)
    print(f"열린 경계 edge {len(be)}개 / 삼각형 {len(F)}")
    if len(be) == 0:
        print("경계 없음 (watertight) — unknown 0"); return

    # 경계 정점과 '이어질 방향': edge 방향 ⟂ + 표면 법선 ⟂ = 표면 접평면에서 바깥쪽
    bv = np.unique(be.reshape(-1))
    P0 = V[be[:, 0]]; P1 = V[be[:, 1]]
    emid = (P0 + P1) / 2
    edir = P1 - P0; edir /= (np.linalg.norm(edir, axis=1, keepdims=True) + 1e-9)
    nrm = (VN[be[:, 0]] + VN[be[:, 1]]) / 2
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    out_dir = np.cross(edir, nrm)
    out_dir /= (np.linalg.norm(out_dir, axis=1, keepdims=True) + 1e-9)

    # 바깥 방향 부호 결정: 메쉬 중심에서 멀어지는 쪽
    ctr_mesh = V.mean(0)
    flip = ((emid - ctr_mesh) * out_dir).sum(1) < 0
    out_dir[flip] = -out_dir[flip]

    U = emid + out_dir * args.offset      # 경계 바깥으로 살짝 밀어낸 지점 = 미관측 시작점
    UN = out_dir

    # free-space 검증
    if args.carve_depth_dir:
        files = sorted(glob.glob(os.path.join(os.path.expanduser(args.carve_depth_dir), "*.npz")))
        freed = np.zeros(len(U), bool)
        for fi, f in enumerate(files):
            if freed.all():
                break
            z = np.load(f)
            depth = z["depth"].astype(np.float32)
            fx, fy = float(z["fx"]), float(z["fy"])
            cx, cy = float(z["cx"]), float(z["cy"])
            c2w = z["c2w"].astype(np.float32)
            R, t = c2w[:3, :3], c2w[:3, 3]
            H, W = depth.shape
            pc = (U - t) @ R
            zc = pc[:, 2]
            ok = zc > 1e-3
            u = np.full(len(U), -1); v = np.full(len(U), -1)
            u[ok] = np.round(fx * pc[ok, 0] / zc[ok] + cx).astype(np.int64)
            v[ok] = np.round(fy * pc[ok, 1] / zc[ok] + cy).astype(np.int64)
            inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inb.any():
                continue
            ds = np.zeros(len(U), np.float32)
            ds[inb] = depth[v[inb], u[inb]]
            freed |= inb & (ds > 1e-3) & (zc < ds - args.free_margin)
            if fi % 300 == 0:
                print(f"  view {fi}/{len(files)}: free {freed.mean()*100:.1f}%")
        U, UN = U[~freed], UN[~freed]
        print(f"free-space 제외 후 unknown {len(U)}")

    if len(U) < args.min_pts:
        print(f"unknown {len(U)} < min_pts — 경계가 거의 닫혀 있음(관측 충분)"); return

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(U)
    pc.normals = o3d.utility.Vector3dVector(UN)
    pc.paint_uniform_color([1, 0.3, 0])
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_point_cloud(out, pc)
    d = UN.mean(0); d /= np.linalg.norm(d) + 1e-9
    print(f"\nunknown 경계 {len(U)}점 → {out}")
    print(f"  centroid {np.round(U.mean(0),3).tolist()}  평균 방향 {np.round(d,3).tolist()}")


if __name__ == "__main__":
    main()
