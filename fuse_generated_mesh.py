#!/usr/bin/env python3
"""3D-native 생성 메쉬(TRELLIS/Hunyuan3D/VARCO)를 우리 TSDF 에 9-DoF 정합하고,
미관측 영역만 생성으로 채워 완결 메쉬를 만든다. 관측 = hard constraint.

설계(Indoor scene reconstruction prior model 세션 기준):
  1) 9-DoF 정합: 회전3 + 이동3 + 축별(anisotropic) 스케일3.
     - up 정렬 + 다중 yaw init → visible(TSDF 상판) 대응 alternating trimmed ICP.
     - 상판이 anchor 라 상판:다리 비율 자동 고정.
  2) render-and-compare 진단: known pose 에서 silhouette IoU · depth err 로 정합 검증.
  3) SDF fusion(hard constraint):
     - 관측 표면(TSDF) 전량 유지.
     - 생성점은 (TSDF 에서 먼 미관측) AND (free-space carve 통과) 인 것만 이식.
       → 관측 영역 침범 + 관측-빈공간 floater 를 둘 다 차단.
     - screened Poisson 으로 경계 blend.

  python fuse_generated_mesh.py \
    --recon output/.../6/train/ours_7000/fuse_post.ply --gen trellis_obj6.glb \
    --gen_up y --world_up z \
    --colmap data/replica_room0_v2/sparse/0 --gid 6 \
    --masks_root data/replica_room0_v2/masks --stems ~/See3D/.../6.txt \
    --out output/.../6/fused.ply --save_aligned

Deps: numpy, open3d, scipy, opencv, PIL, (repo) warp_gt_to_pose.
"""
import os, glob, argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

try:
    from warp_gt_to_pose import read_colmap, cam_center
    from PIL import Image
except Exception:
    read_colmap = None


# ---------------- Sim3 / 9-DoF ----------------
def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd(D.T @ S / len(src))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    s = (d.sum() / (S ** 2).sum() * len(src)) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def rot_axis(axis, deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def up_align_R(gen_up, world_up):
    e = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    a = np.array(e[gen_up], float); b = np.array(e[world_up], float)
    v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def aniso_icp(G, dst_tree, R_pts, R, t, S, iters=40, trim=0.6):
    """9-DoF: p' = R @ (S⊙p) + t. 강체와 축별 스케일을 교대로 갱신."""
    for _ in range(iters):
        cur = (R @ (G * S).T).T + t
        dist, idx = dst_tree.query(cur)
        k = max(30, int(len(G) * trim))
        keep = np.argsort(dist)[:k]
        src, dstp = G[keep], R_pts[idx[keep]]
        _, R, t = umeyama(src * S, dstp, with_scale=False)      # 강체(스케일 고정)
        q = (R.T @ (dstp - t).T).T                               # 정준계 목표
        S = (src * q).sum(0) / ((src * src).sum(0) + 1e-9)       # 축별 스케일
        S = np.clip(S, 0.3 * np.mean(S), 3 * np.mean(S))
    cur = (R @ (G * S).T).T + t
    d, _ = dst_tree.query(cur)
    k = max(30, int(len(G) * trim))
    rmse = np.sqrt((np.sort(d)[:k] ** 2).mean())                 # trimmed RMSE
    return R, t, S, rmse


def apply9(P, R, t, S):
    return (R @ (P * S).T).T + t


# ---------------- 카메라 유틸(carve / render-compare) ----------------
def load_cams(colmap, masks_root, gid, stems_file, images_dir):
    cams = {c["stem"]: c for c in read_colmap(colmap)}
    if stems_file and os.path.exists(os.path.expanduser(stems_file)):
        stems = [l.strip() for l in open(os.path.expanduser(stems_file)) if l.strip()]
    else:
        stems = sorted(cams)
    out = []
    for s in stems:
        if s not in cams:
            continue
        c = cams[s]
        mp = os.path.join(masks_root, gid, "masks", s + ".png")
        m = None
        if os.path.exists(mp):
            a = np.array(Image.open(mp))
            m = (a[..., 3] if (a.ndim == 3 and a.shape[2] == 4)
                 else np.array(Image.open(mp).convert("L"))) > 0
        out.append((c, m))
    return out


def freespace_carve(G, cams, recon_mesh, margin=0.01):
    """관측 카메라가 '빈 공간'으로 본 곳의 생성점 제거(space carving)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(recon_mesh))
    carved = np.zeros(len(G), bool)
    for c, _ in cams:
        R, t = c["R"], c["t"]; C = cam_center(R, t)
        pc = (R @ G.T).T + t                                   # cam 좌표
        z = pc[:, 2]
        u = c["fx"] * pc[:, 0] / np.where(z > 1e-6, z, 1e9) + c["cx"]
        v = c["fy"] * pc[:, 1] / np.where(z > 1e-6, z, 1e9) + c["cy"]
        infr = (z > 1e-6) & (u >= 0) & (u < c["W"]) & (v >= 0) & (v < c["H"])
        d = G - C; dist = np.linalg.norm(d, axis=1)
        dirn = d / (dist[:, None] + 1e-9)
        rays = np.concatenate([np.broadcast_to(C.astype(np.float32), d.shape),
                               dirn.astype(np.float32)], 1)
        th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        # 관측 표면이 생성점보다 뒤 → 생성점은 관측-빈공간에 뜬 것 → carve
        carved |= infr & np.isfinite(th) & (th > dist + margin)
    return carved


def render_compare(gen_aligned, cams, max_views=6):
    """정합 진단: known pose 에서 silhouette IoU · depth L1(overlap)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gen_aligned))
    ious, derrs = [], []
    used = [c for c in cams if c[1] is not None][:max_views]
    for c, m in used:
        W, H = c["W"], c["H"]; R, t = c["R"], c["t"]; C = cam_center(R, t)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        dc = np.stack([(uu - c["cx"]) / c["fx"], (vv - c["cy"]) / c["fy"],
                       np.ones_like(uu, float)], -1).reshape(-1, 3)
        dw = (R.T @ dc.T).T                                    # cam→world 방향(R: world→cam)
        dwn = dw / (np.linalg.norm(dw, axis=1, keepdims=True) + 1e-9)
        rays = np.concatenate([np.broadcast_to(C.astype(np.float32), dwn.shape),
                               dwn.astype(np.float32)], 1)
        th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)
        sil = np.isfinite(th)
        inter = (sil & m).sum(); union = (sil | m).sum()
        ious.append(inter / max(union, 1))
    return float(np.mean(ious)) if ious else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen_up", default="y", choices=["x", "y", "z"])
    ap.add_argument("--world_up", default="z", choices=["x", "y", "z"])
    ap.add_argument("--n_sample", type=int, default=60000)
    ap.add_argument("--graft_dist", type=float, default=0.0, help="0=자동(TSDF간격×3)")
    ap.add_argument("--poisson_depth", type=int, default=9)
    ap.add_argument("--save_aligned", action="store_true")
    # carve / render-compare 용(선택)
    ap.add_argument("--colmap", default="")
    ap.add_argument("--masks_root", default="")
    ap.add_argument("--gid", default="")
    ap.add_argument("--stems", default="")
    ap.add_argument("--images", default="")
    ap.add_argument("--carve_margin", type=float, default=0.01)
    args = ap.parse_args()

    recon = o3d.io.read_triangle_mesh(args.recon)
    gen = o3d.io.read_triangle_mesh(args.gen)
    assert len(recon.vertices) and len(gen.vertices), "메쉬 로드 실패"
    recon.compute_vertex_normals(); gen.compute_vertex_normals()

    rp = (recon.sample_points_poisson_disk(args.n_sample) if len(recon.triangles)
          else recon.sample_points_uniformly(args.n_sample))
    gp = gen.sample_points_uniformly(args.n_sample)
    R_pts, G_pts = np.asarray(rp.points), np.asarray(gp.points)
    dst_tree = cKDTree(R_pts)

    # --- 9-DoF 정합: up 정렬 + yaw 8-init ---
    R0 = up_align_R(args.gen_up, args.world_up)
    best = None
    for yaw in range(0, 360, 45):
        Ry = rot_axis(args.world_up, yaw) @ R0
        gu = (Ry @ G_pts.T).T
        s0 = np.linalg.norm(R_pts.std(0)) / (np.linalg.norm(gu.std(0)) + 1e-9)
        S = np.array([s0, s0, s0]); t = R_pts.mean(0) - (Ry @ (G_pts * S).T).T.mean(0)
        R, t, S, rmse = aniso_icp(G_pts, dst_tree, R_pts, Ry, t, S, iters=20, trim=0.5)
        if best is None or rmse < best[-1]:
            best = (R, t, S, rmse)
    R, t, S, _ = best
    R, t, S, rmse = aniso_icp(G_pts, dst_tree, R_pts, R, t, S, iters=50, trim=0.6)
    print(f"[정합] 9-DoF  scale=({S[0]:.3f},{S[1]:.3f},{S[2]:.3f})  "
          f"trimmed-RMSE={rmse*1000:.1f}mm")

    gv = np.asarray(gen.vertices)
    gen.vertices = o3d.utility.Vector3dVector(apply9(gv, R, t, S))
    gen.compute_vertex_normals()
    G_world = apply9(G_pts, R, t, S)

    # --- 카메라 기반 진단 + carve(선택) ---
    cams = None
    if args.colmap and read_colmap is not None:
        cams = load_cams(args.colmap, args.masks_root, args.gid, args.stems, args.images)
        iou = render_compare(gen, cams)
        print(f"[진단] render-compare silhouette IoU={iou:.3f} (1.0에 가까울수록 정합 우수)")

    # --- 융합: 미관측 이식(거리) ∩ free-space carve 통과 ---
    med = np.median(dst_tree.query(R_pts[np.random.choice(len(R_pts), 2000)], k=2)[0][:, 1])
    tau = args.graft_dist if args.graft_dist > 0 else med * 3
    gdist, _ = dst_tree.query(G_world)
    graft = gdist > tau                                        # TSDF 미커버(미관측 후보)
    if cams is not None:
        carved = freespace_carve(G_world, cams, recon, margin=args.carve_margin)
        graft &= ~carved                                      # 관측-빈공간 floater 제거
        print(f"[융합] graft τ={tau*1000:.1f}mm  carve 제거 {carved.sum()}  "
              f"최종 이식 {graft.sum()}/{len(G_world)} ({graft.mean()*100:.0f}%)")
    else:
        print(f"[융합] graft τ={tau*1000:.1f}mm  이식 {graft.sum()} (carve 미적용)")

    gp_w = gp.select_by_index(np.where(graft)[0])
    gp_w.points = o3d.utility.Vector3dVector(G_world[graft])
    for pc in (rp, gp_w):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pc.orient_normals_consistent_tangent_plane(30)

    fused = rp + gp_w
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        fused, depth=args.poisson_depth)
    mesh.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), 0.02))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"→ 완결 메쉬: {args.out}  (정점 {len(mesh.vertices)})")
    if args.save_aligned:
        p = args.out.replace(".ply", "_gen_aligned.ply")
        o3d.io.write_triangle_mesh(p, gen); print(f"→ 정합 생성메쉬: {p}")


if __name__ == "__main__":
    main()
