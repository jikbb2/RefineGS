#!/usr/bin/env python3
"""미관측 '볼륨' 기반 unknown 추출 — 경계(빙산의 일각) 문제 해결. SDF 불필요.

객체 bbox 를 복셀로 채우고 각 복셀을 3분류:
  surface : TSDF 메쉬 표면 근처            → 관측된 표면
  free    : 어떤 카메라의 (카메라~depth) 구간을 통과 → 비었음이 확인
  unknown : 둘 다 아님                     → 어떤 카메라도 들여다본 적 없는 볼륨
unknown 볼륨의 '껍질'(free 와 맞닿는 면)이 곧 SEVA 가 채워야 할 표면이고,
그 면의 법선(free 쪽 = 카메라가 접근 가능한 방향)이 궤적 방향을 준다.

  python make_unknown_volume.py \
    --tsdf output/replica_room0_v2/refinegs_full/6/train/ours_7000/fuse_post.ply \
    --carve_depth_dir ~/carve_depth_mono \
    --voxel 0.02 --pad 0.25 \
    --out ~/prior/obj6/unknown.ply

출력: 껍질 점군(법선 = free 방향) + 통계(unknown 볼륨 크기 = refinement 필요량 지표).
Deps: numpy, open3d, scipy.
"""
import os
import glob
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, binary_erosion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsdf", required=True, help="관측 메쉬(TSDF fuse_post)")
    ap.add_argument("--carve_depth_dir", required=True, help="dump_scene_depth 출력(자유공간 판정)")
    ap.add_argument("--voxel", type=float, default=0.02, help="복셀 크기(m)")
    ap.add_argument("--pad", type=float, default=0.25, help="객체 bbox 확장(m) — 주변 미관측 포함")
    ap.add_argument("--surf_band", type=float, default=0.03, help="TSDF 표면으로 간주할 거리(m)")
    ap.add_argument("--free_margin", type=float, default=0.04, help="depth 표면 앞 이 거리까지 free")
    ap.add_argument("--min_component", type=int, default=30, help="이보다 작은 unknown 덩어리 무시(복셀 수)")
    ap.add_argument("--max_pts", type=int, default=60000)
    ap.add_argument("--other_meshes", nargs="*", default=[],
                    help="다른 객체 메쉬 glob(예: 'output/.../refinegs_full/*/train/ours_*/fuse_post.ply'). "
                         "이들에 더 가까운 복셀은 unknown 에서 제외(꽃병 등 타 객체 배제)")
    ap.add_argument("--other_band", type=float, default=0.08,
                    help="다른 객체 표면에서 이 거리(m) 이내면 그 객체 소관으로 간주")
    ap.add_argument("--near_self", type=float, default=0.0,
                    help=">0 이면 이 객체 표면에서 이 거리(m) 밖 복셀은 unknown 에서 제외(주변 잡음 차단)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = o3d.io.read_triangle_mesh(os.path.expanduser(args.tsdf))
    V = np.asarray(m.vertices)
    assert len(V) > 0, "빈 메쉬"
    lo, hi = V.min(0) - args.pad, V.max(0) + args.pad
    vs = args.voxel
    dims = np.maximum(np.ceil((hi - lo) / vs).astype(int), 1)
    print(f"객체 bbox {np.round(lo,2)}~{np.round(hi,2)}  복셀 {dims.tolist()} = {np.prod(dims):,}개")

    gx, gy, gz = np.meshgrid(*[np.arange(d) for d in dims], indexing="ij")
    centers = lo + (np.stack([gx, gy, gz], -1).reshape(-1, 3) + 0.5) * vs   # (N,3)

    # 1) surface: TSDF 표면 근처
    d_surf, _ = cKDTree(V).query(centers, workers=-1)
    surface = d_surf < args.surf_band
    print(f"surface 복셀 {surface.sum():,} ({surface.mean()*100:.1f}%)")

    # 2) free: 어떤 카메라의 (카메라~depth) 구간 통과
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.carve_depth_dir), "*.npz")))
    assert files, f"carve depth 없음: {args.carve_depth_dir}"
    free = np.zeros(len(centers), bool)
    for fi, f in enumerate(files):
        z = np.load(f)
        depth = z["depth"].astype(np.float32)
        fx, fy = float(z["fx"]), float(z["fy"])
        cx, cy = float(z["cx"]), float(z["cy"])
        c2w = z["c2w"].astype(np.float32)
        R, t = c2w[:3, :3], c2w[:3, 3]
        H, W = depth.shape
        pc = (centers - t) @ R
        zc = pc[:, 2]
        ok = zc > 1e-3
        if not ok.any():
            continue
        u = np.full(len(centers), -1); v = np.full(len(centers), -1)
        u[ok] = np.round(fx * pc[ok, 0] / zc[ok] + cx).astype(np.int64)
        v[ok] = np.round(fy * pc[ok, 1] / zc[ok] + cy).astype(np.int64)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inb.any():
            continue
        ds = np.zeros(len(centers), np.float32)
        ds[inb] = depth[v[inb], u[inb]]
        free |= inb & (ds > 1e-3) & (zc < ds - args.free_margin)
        if fi % 300 == 0:
            print(f"  view {fi}/{len(files)}: free {free.mean()*100:.1f}%")
    print(f"free 복셀 {free.sum():,} ({free.mean()*100:.1f}%)")

    # 2b) 다른 객체(other_meshes) 근처는 이 객체의 refinement 대상 아님 → 제외
    other = np.zeros(len(centers), bool)
    if args.other_meshes:
        import glob as _g
        opaths = []
        for pat in args.other_meshes:
            opaths += [p for p in _g.glob(os.path.expanduser(pat))
                       if os.path.abspath(p) != os.path.abspath(os.path.expanduser(args.tsdf))]
        OV = []
        for p in opaths:
            om = o3d.io.read_triangle_mesh(p)
            if len(om.vertices):
                ov = np.asarray(om.vertices)
                if np.all(ov.max(0) > lo) and np.all(ov.min(0) < hi):    # bbox 겹치는 것만
                    OV.append(ov)
        if OV:
            OVa = np.concatenate(OV)
            d_o, _ = cKDTree(OVa).query(centers, workers=-1)
            d_self, _ = cKDTree(V).query(centers, workers=-1)
            other = (d_o < args.other_band) & (d_o < d_self)   # 다른 객체에 더 가까운 복셀
            print(f"다른 객체({len(OV)}개) 근처 제외: {other.sum():,} 복셀")

    # 3) unknown = 나머지 (+ 이 객체 근방으로 제한 옵션)
    unknown = ~(surface | free | other)
    if args.near_self > 0:
        far = d_surf > args.near_self
        print(f"객체 표면 {args.near_self}m 밖 제외: {(unknown & far).sum():,} 복셀")
        unknown &= ~far
    U3 = unknown.reshape(dims)
    print(f"unknown 복셀 {unknown.sum():,} ({unknown.mean()*100:.1f}%)  "
          f"= 볼륨 {unknown.sum()*vs**3:.4f} m³")

    # 연결성분 필터 (작은 잡음 덩어리 제거)
    try:
        from scipy.ndimage import label
        lab, n = label(U3)
        if n > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep_lbl = np.where(sizes >= args.min_component)[0]
            U3 = np.isin(lab, keep_lbl)
            print(f"연결성분 {n}개 → {len(keep_lbl)}개 유지 (≥{args.min_component}복셀)")
    except Exception as e:
        print(f"[warn] 연결성분 필터 skip: {e}")

    # 4) 껍질: unknown 이면서 free 와 맞닿는 복셀 → SEVA 가 채울 표면
    F3 = free.reshape(dims)
    free_dil = binary_dilation(F3, iterations=1)
    shell = U3 & free_dil
    print(f"껍질(free 접촉) 복셀 {shell.sum():,}")
    if shell.sum() == 0:
        shell = U3 & ~binary_erosion(U3, iterations=1)      # fallback: 볼륨 외곽
        print(f"  fallback 외곽 껍질 {shell.sum():,}")
    if shell.sum() == 0:
        print("unknown 껍질 없음 — 관측 충분"); return

    # 껍질 점 + 법선(free 방향: free 복셀 쪽으로 향하는 gradient)
    idx = np.argwhere(shell)
    P = lo + (idx + 0.5) * vs
    Ffloat = F3.astype(np.float32)
    gxx, gyy, gzz = np.gradient(Ffloat)
    N = np.stack([gxx[tuple(idx.T)], gyy[tuple(idx.T)], gzz[tuple(idx.T)]], -1)
    nn = np.linalg.norm(N, axis=1, keepdims=True)
    bad = nn[:, 0] < 1e-6
    if bad.any():                                            # gradient 0 → 볼륨 중심에서 바깥
        c = P.mean(0)
        N[bad] = P[bad] - c
        nn = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.maximum(nn, 1e-9)

    if len(P) > args.max_pts:
        sel = np.random.choice(len(P), args.max_pts, replace=False)
        P, N = P[sel], N[sel]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.normals = o3d.utility.Vector3dVector(N)
    pc.paint_uniform_color([1, 0.3, 0])
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_point_cloud(out, pc)
    d = N.mean(0); d /= np.linalg.norm(d) + 1e-9
    print(f"\nunknown 껍질 {len(P)}점 → {out}")
    print(f"  centroid {np.round(P.mean(0),3).tolist()}  평균 접근방향 {np.round(d,3).tolist()}")
    print(f"  (refinement 필요량 지표: unknown 볼륨 {U3.sum()*vs**3:.4f} m³)")


if __name__ == "__main__":
    main()
