#!/usr/bin/env python3
"""견고한 정합: Amodal3R(canonical, 색 있음) → recon(scene 좌표).

global FPFH + RANSAC(전역 coarse) → ICP(미세) → Sim3.
변환을 '메시 객체'에 직접 적용해 vertex color 보존(재생성 안 함).

실행:
  python register_robust.py --gen mesh.ply --recon fuse_post.ply --out mesh_reg.ply
"""
import argparse
import numpy as np
import open3d as o3d


def fpfh(pcd, voxel):
    d = pcd.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return d, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000, help="샘플 점 수")
    ap.add_argument("--icp_scale", type=float, default=2.0, help="ICP dist = voxel*이값")
    a = ap.parse_args()

    gen = o3d.io.read_triangle_mesh(a.gen)      # 색 포함 로드
    recon = o3d.io.read_triangle_mesh(a.recon)
    gen.compute_vertex_normals(); recon.compute_vertex_normals()
    has_color = gen.has_vertex_colors()

    # 1) 거친 uniform 스케일 (gen canonical → recon 크기)
    ge = gen.get_axis_aligned_bounding_box().get_extent()
    re = recon.get_axis_aligned_bounding_box().get_extent()
    s = float(np.linalg.norm(re) / max(np.linalg.norm(ge), 1e-9))
    gen.scale(s, center=gen.get_center())

    # 2) 포인트 샘플 + voxel/FPFH
    gp = gen.sample_points_uniformly(a.n)
    rp = recon.sample_points_uniformly(a.n)
    diag = float(np.linalg.norm(re))
    voxel = max(diag / 60.0, 1e-4)
    gd, gf = fpfh(gp, voxel); rd, rf = fpfh(rp, voxel)

    # 3) RANSAC 전역 정합
    dist = voxel * 1.5
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        gd, rd, gf, rf, True, dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
    gen.transform(res.transformation)           # coarse 적용 (색 보존)

    # 4) ICP 미세 (point-to-plane)
    gp2 = gen.sample_points_uniformly(a.n)
    gp2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    rp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    icp = o3d.pipelines.registration.registration_icp(
        gp2, rp, voxel * a.icp_scale, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    gen.transform(icp.transformation)           # 미세 적용 (색 보존)

    gen.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, gen)      # 색 포함 저장

    ge2 = gen.get_axis_aligned_bounding_box().get_extent()
    print(f"scale={s:.3f}  ransac_fit={res.fitness:.3f}  icp_fit={icp.fitness:.3f} "
          f"icp_rmse={icp.inlier_rmse*1000:.1f}mm  color={'yes' if has_color else 'NO'}")
    print(f"ext_ratio(gen/recon)={(ge2/np.maximum(re,1e-9)).round(2)}  -> {a.out}")


if __name__ == "__main__":
    main()
