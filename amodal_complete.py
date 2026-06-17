#!/usr/bin/env python3
"""
축2+3 best — amodal observation-consistent surface completion (재학습 0, 마스크 불필요).

문제: modal 마스크가 객체 위 물체(램프 밑받침 등)를 제외 → recon 표면에 구멍.
해법: 관측된 평면 + 관측된 외곽(convex hull) 안에서, recon이 비운 구멍을 평면 보간으로 채움.
      이 채움은 *관측된 표면에 둘러싸인* occlusion 영역이라 hallucination이 아니라
      증거-일관 보간(observation-consistent). GT로 구멍 영역 completion 정량 비교.

자기완결: recon만 입력(평면·외곽을 recon에서 추출). SAM 마스크/재학습 불필요.
한계(정직): *관측 표면에 둘러싸인 평면형 occlusion 구멍*만 채움. 둘러싸이지 않은
            미관측(다리 등)은 다루지 않음(그건 hallucination 없이는 불가).

의존: numpy, open3d, scipy, trimesh. split_and_splat env.

실행:
    conda activate split_and_splat
    python amodal_complete.py \
        --recon_ply output/replica_room0/axis3_sweep/reg_strong/98/train/ours_7000/fuse_post.ply \
        --gt_mesh ../room_0/habitat/mesh_semantic.ply --gt_id 7 \
        --res 0.01 --fill_thresh 0.03 --plane_dist 0.012 --plane_band 0.02 \
        --out_dir ~/amodal_98
"""
import argparse, os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, ConvexHull, Delaunay


def load_gt_verts(gt_mesh, gt_id):
    import trimesh
    with open(gt_mesh, "rb") as f:
        data = trimesh.exchange.ply.load_ply(f)
    verts = np.asarray(data["vertices"], np.float64)
    fd = data["metadata"]["_ply_raw"]["face"]["data"]
    fm = (fd["object_id"] == gt_id)
    return verts[np.unique(fd["vertex_indices"]["f1"][fm].flatten())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_ply", required=True)
    ap.add_argument("--gt_mesh", required=True); ap.add_argument("--gt_id", type=int, required=True)
    ap.add_argument("--res", type=float, default=0.01, help="채움 격자 간격(m)")
    ap.add_argument("--fill_thresh", type=float, default=0.03, help="recon 점이 이보다 멀면 구멍으로 간주")
    ap.add_argument("--plane_dist", type=float, default=0.012, help="RANSAC 평면 inlier 거리")
    ap.add_argument("--plane_band", type=float, default=0.02, help="GT를 tabletop(on-plane)으로 볼 거리")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = o3d.io.read_triangle_mesh(args.recon_ply)
    xyz = np.asarray(m.vertices) if len(m.vertices) else np.asarray(o3d.io.read_point_cloud(args.recon_ply).points)
    gt = load_gt_verts(args.gt_mesh, args.gt_id)
    print(f"recon pts={len(xyz)}  GT={len(gt)}")

    # 1) 지배 평면(테이블 상판) RANSAC
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(xyz)
    plane, inliers = pcd.segment_plane(args.plane_dist, ransac_n=3, num_iterations=2000)
    a, b, c, d = plane; n = np.array([a, b, c]); n = n/np.linalg.norm(n)
    inl = xyz[inliers]
    print(f"plane inliers={len(inl)}/{len(xyz)}  normal={n.round(3)}")

    # 2) 평면 2D 기저
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u); v = np.cross(n, u)
    origin = inl.mean(0)
    def to2d(P): q = P - origin; return np.stack([q@u, q@v], 1)
    def to3d(P2): return origin + P2[:,0:1]*u + P2[:,1:2]*v
    inl2d = to2d(inl)

    # 3) 관측 외곽(convex hull) + 격자
    hull = ConvexHull(inl2d); dela = Delaunay(inl2d[hull.vertices])
    lo, hi = inl2d.min(0), inl2d.max(0)
    gx = np.arange(lo[0], hi[0], args.res); gy = np.arange(lo[1], hi[1], args.res)
    GX, GY = np.meshgrid(gx, gy); grid = np.stack([GX.ravel(), GY.ravel()], 1)
    inside = dela.find_simplex(grid) >= 0                          # 관측 외곽 안
    covered = cKDTree(inl2d).query(grid)[0] < args.fill_thresh      # recon이 이미 덮음
    holes = inside & ~covered                                       # 외곽 안 + recon 비움 = occlusion 구멍
    comp3d = to3d(grid[holes])
    print(f"외곽 안 격자={inside.sum()}  recon 덮음={covered.sum()}  채운 구멍점={len(comp3d)}")

    # 저장
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "completed.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(comp3d)))
    union = np.vstack([xyz, comp3d])
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "union.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(union)))

    # 4) 평가: GT completion (GT->nearest), tabletop(on-plane) vs off-plane(다리 등)
    dist_plane = np.abs((gt - origin) @ n)
    on = dist_plane < args.plane_band
    d_rec = cKDTree(xyz).query(gt)[0]
    d_uni = cKDTree(union).query(gt)[0]
    def mm(x): return x.mean()*1000
    print(f"\n{'group':>22} {'n':>6} {'recon(mm)':>11} {'union(mm)':>11} {'Δ':>8}")
    for name, sel in [("on-plane(tabletop)", on), ("off-plane(legs/side)", ~on),
                      ("on-plane HOLE(d_rec>3cm)", on & (d_rec > 0.03)), ("ALL", np.ones(len(gt), bool))]:
        if sel.sum() == 0: continue
        r, uu = mm(d_rec[sel]), mm(d_uni[sel])
        print(f"{name:>22} {sel.sum():>6} {r:>11.1f} {uu:>11.1f} {r-uu:>8.1f}")
    print(f"\n저장: {args.out_dir} (completed.ply, union.ply)")
    print("판정: on-plane HOLE 행에서 union이 recon보다 크게 줄면(Δ↑) = 관측에 둘러싸인 occlusion "
          "표면을 hallucination 없이 복원 = best 방법 가치. off-plane(다리)은 안 변함(정직한 경계).")


if __name__ == "__main__":
    main()
