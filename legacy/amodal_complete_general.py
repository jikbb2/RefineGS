#!/usr/bin/env python3
"""
일반 객체 관측-일관 완성 (비평면 포함, 재학습 0).

평면 가정 없이: recon 점+법선으로 screened Poisson → 구멍을 보간으로 메움.
단 (a) density-trim으로 저신뢰 외삽 제거, (b) 관측 support 거리 trim으로 recon 점에서
support_dist 초과로 떨어진 부분(=뒷면/미관측 외삽) 제거 → *관측에 둘러싸인 구멍만* 채움
= observation-consistent(hallucination 아님). 평면형(테이블)·곡면형 모두 동작.

aggregate용: 모든 per-object recon에 적용 후 eval_object_mesh batch --auto_match로
recon vs completed 비교 → 수십 객체 평균에서 확실한 차이.

의존: numpy, open3d, scipy. split_and_splat env.

실행(단일):
    python amodal_complete_general.py \
        --recon_ply <obj>/train/ours_7000/fuse_post.ply \
        --out_ply <obj>/train/ours_7000/fuse_completed.ply \
        --poisson_depth 9 --density_trim 0.1 --support_dist 0.05
"""
import argparse, os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_ply", required=True)
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--poisson_depth", type=int, default=9)
    ap.add_argument("--density_trim", type=float, default=0.1,
                    help="하위 X 분위 density(저신뢰 외삽) 제거")
    ap.add_argument("--support_dist", type=float, default=0.05,
                    help="recon 점에서 이보다 먼 완성면 제거(관측 support 밖=외삽)")
    ap.add_argument("--n_sample", type=int, default=200000)
    args = ap.parse_args()

    m = o3d.io.read_triangle_mesh(args.recon_ply)
    if len(m.vertices) == 0:
        print("[ERROR] empty recon"); return
    # 점+법선 (mesh면 표면 샘플, 법선 보장)
    if len(m.triangles) > 0:
        m.compute_vertex_normals()
        pcd = m.sample_points_poisson_disk(min(args.n_sample, max(2000, len(m.vertices))))
        if not pcd.has_normals():
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.05, 30))
    else:
        pcd = o3d.io.read_point_cloud(args.recon_ply)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.05, 30))
        pcd.orient_normals_consistent_tangent_plane(30)
    rec_pts = np.asarray(pcd.points)
    print(f"recon pts={len(rec_pts)}")

    # screened Poisson → 구멍 보간(watertight)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.poisson_depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    v = np.asarray(mesh.vertices)
    keep = np.ones(len(v), bool)
    # (a) density-trim: 저신뢰(데이터 멂) 외삽 제거
    if args.density_trim > 0 and len(dens) > 0:
        keep &= dens > np.quantile(dens, args.density_trim)
    # (b) 관측 support trim: recon 점에서 먼 완성면 제거(뒷면/미관측 외삽 차단)
    d2rec = cKDTree(rec_pts).query(v)[0]
    keep &= d2rec <= args.support_dist
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(args.out_ply, mesh)

    vv = np.asarray(mesh.vertices)
    added = (cKDTree(rec_pts).query(vv)[0] > args.support_dist*0.5).sum()
    print(f"completed verts={len(vv)} (recon에서 떨어진 보간면 ~{added})")
    print(f"저장: {args.out_ply}")
    print("aggregate: 모든 객체에 적용 후 eval_object_mesh batch --auto_match로 recon vs completed 비교.")


if __name__ == "__main__":
    main()
