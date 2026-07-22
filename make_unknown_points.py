#!/usr/bin/env python3
"""객체의 unknown(미관측) 표면 추출 — TSDF(관측) vs SDF(완성) 차분 + free-space 검증.

unknown = SDF 정점 중
  (1) TSDF 표면에서 --tau 밖            → 관측되지 않은 표면
  (2) 어떤 카메라도 통과해 본 적 없음    → 인공 뚜껑(항아리 입구 막힘) 배제
법선은 SDF 메쉬 정점 법선(바깥 방향) → 궤적 설계에서 '봐야 할 방향'으로 사용.

  python make_unknown_points.py \
    --tsdf output/replica_room0_v2/refinegs_full/6/train/ours_7000/fuse_post.ply \
    --sdf  output/replica_room0_v2/refinegs_full/6/train/ours_7000/sdf_obj_post.ply \
    --carve_depth_dir ~/carve_depth_mono \
    --out ~/See3D/dataset/obj6/unknown.ply

Deps: numpy, open3d, scipy.
"""
import os
import glob
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsdf", required=True, help="관측 메쉬(TSDF fuse_post)")
    ap.add_argument("--sdf", required=True, help="완성 메쉬(SDF post)")
    ap.add_argument("--carve_depth_dir", default="", help="dump_scene_depth 출력 — free-space 검증")
    ap.add_argument("--tau", type=float, default=0.03, help="TSDF 표면에서 이 거리(m) 밖 = 미관측")
    ap.add_argument("--free_margin", type=float, default=0.05,
                    help="카메라~depth 표면 사이 이 여유(m) 안쪽을 통과한 점 = 자유공간(제외)")
    ap.add_argument("--px_per_view", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tm = o3d.io.read_triangle_mesh(os.path.expanduser(args.tsdf))
    sm = o3d.io.read_triangle_mesh(os.path.expanduser(args.sdf))
    assert len(tm.vertices) and len(sm.vertices), "메쉬 비어있음"
    sm.compute_vertex_normals()
    SV = np.asarray(sm.vertices)
    SN = np.asarray(sm.vertex_normals)
    TV = np.asarray(tm.vertices)

    # (1) TSDF 에서 먼 정점 = 미관측 후보
    d, _ = cKDTree(TV).query(SV, workers=-1)
    cand = d > args.tau
    print(f"SDF 정점 {len(SV)} → TSDF 에서 {args.tau}m 밖: {cand.sum()} ({cand.mean()*100:.1f}%)")

    # (2) free-space 검증: 카메라→depth 표면 사이를 지나간 점은 '표면 없음'이 확인된 곳 → 제외
    keep = cand.copy()
    if args.carve_depth_dir:
        files = sorted(glob.glob(os.path.join(os.path.expanduser(args.carve_depth_dir), "*.npz")))
        assert files, f"carve depth 없음: {args.carve_depth_dir}"
        P = SV[cand]
        freed = np.zeros(len(P), bool)
        for fi, f in enumerate(files):
            if freed.all():
                break
            z = np.load(f)
            depth = z["depth"].astype(np.float32)
            fx, fy = float(z["fx"]), float(z["fy"])
            cx, cy = float(z["cx"]), float(z["cy"])
            c2w = z["c2w"].astype(np.float32)
            R = c2w[:3, :3]; t = c2w[:3, 3]
            H, W = depth.shape
            pc = (P - t) @ R                      # world → camera
            zc = pc[:, 2]
            ok = zc > 1e-3
            u = np.full(len(P), -1); v = np.full(len(P), -1)
            u[ok] = np.round(fx * pc[ok, 0] / zc[ok] + cx).astype(np.int64)
            v[ok] = np.round(fy * pc[ok, 1] / zc[ok] + cy).astype(np.int64)
            inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inb.any():
                continue
            dsurf = np.zeros(len(P), np.float32)
            dsurf[inb] = depth[v[inb], u[inb]]
            # 카메라~표면 사이(여유 margin) 를 지나감 = 그 지점은 빈 공간
            freed |= inb & (dsurf > 1e-3) & (zc < dsurf - args.free_margin)
            if fi % 200 == 0:
                print(f"  view {fi}/{len(files)}: free 확인 {freed.mean()*100:.1f}%")
        idx = np.where(cand)[0]
        keep[idx[freed]] = False
        print(f"free-space 제외: {freed.sum()} → unknown {keep.sum()}")

    U, UN = SV[keep], SN[keep]
    if len(U) == 0:
        print("unknown 0 — tau/free_margin 조정 필요"); return
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(U)
    pc.normals = o3d.utility.Vector3dVector(UN)
    pc.paint_uniform_color([1, 0.3, 0])
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_point_cloud(out, pc)
    ctr = U.mean(0); nmean = UN.mean(0); nmean /= np.linalg.norm(nmean) + 1e-9
    print(f"\nunknown {len(U)}점 → {out}")
    print(f"  centroid {np.round(ctr,3).tolist()}  평균 법선 {np.round(nmean,3).tolist()}")
    print("  (평균 법선 = 이 면들이 향하는 방향 → 궤적 카메라를 그쪽에 배치)")


if __name__ == "__main__":
    main()
