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
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import cv2

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


def aniso_icp(G, dst_tree, R_pts, R, t, S, iters=40, trim=0.6, isotropic=False):
    """정합 ICP: p' = R @ (S⊙p) + t. isotropic=True 면 등방 Sim3(S 스칼라)."""
    for _ in range(iters):
        cur = (R @ (G * S).T).T + t
        dist, idx = dst_tree.query(cur)
        k = max(30, int(len(G) * trim))
        keep = np.argsort(dist)[:k]
        src, dstp = G[keep], R_pts[idx[keep]]
        if isotropic:
            s, R, t = umeyama(src, dstp, with_scale=True)        # 등방 Sim3
            S = np.array([s, s, s])
        else:
            _, R, t = umeyama(src * S, dstp, with_scale=False)   # 강체(스케일 고정)
            q = (R.T @ (dstp - t).T).T                           # 정준계 목표
            S = (src * q).sum(0) / ((src * src).sum(0) + 1e-9)   # 축별 스케일
            S = np.clip(S, 0.3 * np.mean(S), 3 * np.mean(S))
    cur = (R @ (G * S).T).T + t
    d, _ = dst_tree.query(cur)
    k = max(30, int(len(G) * trim))
    rmse = np.sqrt((np.sort(d)[:k] ** 2).mean())                 # trimmed RMSE
    return R, t, S, rmse


def apply9(P, R, t, S):
    return (R @ (P * S).T).T + t


def refine_pose_rc(gen_canon_mesh, cams, R, t, S, isotropic=False,
                   lr_w=192, max_views=6, maxiter=120):
    """render-and-compare 포즈 미세정합: 여러 뷰 실루엣 IoU 최대화.
    평면(상판) 관측의 yaw·수직 모호성을 실루엣으로 해소. 광선을 canonical 로
    역변환해 고정 메쉬에 raycast(저해상도) → 빠른 미분자유 최적화."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gen_canon_mesh))
    # 뷰별 저해상도 광선 원점/방향(world) + 다운스케일 마스크 사전계산
    views = []
    for c, m in [x for x in cams if x[1] is not None][:max_views]:
        sc = lr_w / c["W"]; W = lr_w; H = int(round(c["H"] * sc))
        fx, fy = c["fx"] * sc, c["fy"] * sc; cx, cy = c["cx"] * sc, c["cy"] * sc
        Rc, tc = c["R"], c["t"]; C = cam_center(Rc, tc)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        dcam = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        dwn = (Rc.T @ dcam.T).T                                   # world 방향
        md = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        views.append((C.astype(np.float64), dwn, md))

    def unpack(x):
        R_ = Rotation.from_rotvec(x[:3]).as_matrix(); t_ = x[3:6]
        S_ = np.exp(x[6:7].repeat(3)) if isotropic else np.exp(x[6:9])
        return R_, t_, S_

    def loss(x):
        R_, t_, S_ = unpack(x); Sinv = 1.0 / S_; Rt = R_.T
        tot = 0.0
        for C, dwn, md in views:
            o_c = (Sinv * (Rt @ (C - t_)))                        # canonical 원점
            d_c = (Sinv * (Rt @ dwn.T).T)                         # canonical 방향
            rays = np.concatenate([np.broadcast_to(o_c.astype(np.float32), d_c.shape),
                                   d_c.astype(np.float32)], 1)
            th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            sil = np.isfinite(th).reshape(md.shape)
            inter = (sil & md).sum(); union = (sil | md).sum()
            tot += 1.0 - inter / max(union, 1)
        return tot / len(views)

    x0 = np.concatenate([Rotation.from_matrix(R).as_rotvec(), t,
                         (np.log(S).mean(keepdims=True) if isotropic else np.log(S))])
    iou0 = 1 - loss(x0)
    res = minimize(loss, x0, method="Powell", options={"maxiter": maxiter, "xtol": 1e-4})
    R_, t_, S_ = unpack(res.x); iou1 = 1 - res.fun
    print(f"[정합] render-compare 최적화 IoU {iou0:.3f} → {iou1:.3f}")
    return R_, t_, S_


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


def isolate_target(G, keep, recon_tree, eps=0.05, min_points=50, tau=0.12):
    """문맥 생성 시 딸려온 다른 객체(소파 등) 제거: 생성점을 DBSCAN 클러스터링해
    obj TSDF 에서 먼 클러스터(=다른 객체)를 버린다. 타깃은 관측 TSDF 에 인접."""
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return keep
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(G[idx])
    lab = np.array(pc.cluster_dbscan(eps=eps, min_points=min_points))
    out = keep.copy()
    for L in set(lab.tolist()):
        sel = lab == L
        near = recon_tree.query(G[idx[sel]])[0].min() if sel.any() else 1e9
        if L == -1 or near > tau:                       # 노이즈/원거리 클러스터 → 제거
            out[idx[sel]] = False
    return out


def visual_hull_carve(G, cams, recon_mesh, margin=0.01):
    """occlusion-aware 실루엣 carve: 관측 표면에 가려지지 않았는데 마스크 밖으로
    투영되는 생성점 = 실루엣과 모순 = 할루시네이션 → 제거.
    가려진(occluded) 곳은 판단 보류 → 테이블 아래 다리 등 미관측은 보존."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(recon_mesh))
    carved = np.zeros(len(G), bool)
    for c, m in cams:
        if m is None:
            continue
        R, t = c["R"], c["t"]; C = cam_center(R, t)
        pc = (R @ G.T).T + t; z = pc[:, 2]
        zz = np.where(z > 1e-6, z, 1e9)
        u = (c["fx"] * pc[:, 0] / zz + c["cx"])
        v = (c["fy"] * pc[:, 1] / zz + c["cy"])
        infr = (z > 1e-6) & (u >= 0) & (u < c["W"]) & (v >= 0) & (v < c["H"])
        # 관측 표면이 생성점보다 앞 → occluded(판단 보류)
        d = G - C; dist = np.linalg.norm(d, axis=1)
        dirn = d / (dist[:, None] + 1e-9)
        rays = np.concatenate([np.broadcast_to(C.astype(np.float32), d.shape),
                               dirn.astype(np.float32)], 1)
        th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        occluded = np.isfinite(th) & (th < dist - margin)
        # 마스크 안/밖
        ui = np.clip(u, 0, c["W"] - 1).astype(int)
        vi = np.clip(v, 0, c["H"] - 1).astype(int)
        inside = np.zeros(len(G), bool)
        inside[infr] = m[vi[infr], ui[infr]]
        carved |= infr & ~occluded & ~inside          # 보이는데 실루엣 밖 → carve
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


def load_gen_with_color(path):
    """생성 메쉬 로드 + 텍스처를 정점색으로 베이킹(trimesh). 반환 (o3d_mesh, trimesh|None).
    o3d 는 텍스처를 정점색으로 안 옮기므로, trimesh 의 to_color 로 UV 텍스처를 정점색으로 굽는다.
    trimesh 객체는 텍스처 보존 export(.glb/.obj)용으로 함께 반환."""
    p = os.path.expanduser(path)
    try:
        import trimesh
        tm = trimesh.load(p, process=False, force="mesh")
        V = np.asarray(tm.vertices, np.float64)
        F = np.asarray(tm.faces, np.int32)
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                      o3d.utility.Vector3iVector(F))
        try:
            vc = np.asarray(tm.visual.to_color().vertex_colors)[:, :3].astype(np.float64) / 255.0
            if len(vc) == len(V):
                m.vertex_colors = o3d.utility.Vector3dVector(vc)
                print(f"[색] 텍스처→정점색 베이킹 완료 ({len(V)}정점)")
        except Exception as e:
            print(f"[색] 텍스처 베이킹 실패({e}) — 색 없이 진행")
        m.compute_vertex_normals()
        return m, tm
    except Exception as e:
        print(f"[색] trimesh 로드 실패({e}) — o3d 로드(텍스처 무시)")
        m = o3d.io.read_triangle_mesh(p)
        m.compute_vertex_normals()
        return m, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", default="", help="Poisson 미리보기 메쉬 경로(메쉬 낼 때만)")
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
    ap.add_argument("--isolate", action="store_true",
                    help="문맥 생성 시 딸려온 다른 객체(소파 등) 클러스터 제거")
    ap.add_argument("--iso_eps", type=float, default=0.05, help="DBSCAN eps(m)")
    ap.add_argument("--iso_tau", type=float, default=0.12, help="타깃 TSDF 인접 임계(m)")
    ap.add_argument("--isotropic", action="store_true",
                    help="등방 스케일(Sim3). 생성 비율이 정확하면 anisotropic overfit 방지")
    ap.add_argument("--refine", action="store_true",
                    help="render-and-compare 로 포즈 미세정합(평면 관측 yaw·수직 모호성 해소)")
    ap.add_argument("--overwrite", action="store_true",
                    help="기존 산출물 덮어쓰기. 미지정 시 기존 파일 있으면 타임스탬프 붙여 보존")
    ap.add_argument("--export_points", default="",
                    help="carve 통과 미관측 prior 점군(구멍 가능) PLY 경로")
    ap.add_argument("--dense_points", default="",
                    help="정합된 생성 메쉬 전체를 carve 없이 dense 샘플한 점군 PLY 경로")
    ap.add_argument("--dense_n", type=int, default=300000, help="dense 샘플 점 수")
    ap.add_argument("--dense_graft_only", action="store_true",
                    help="dense 점군에서 관측 표면 근접점 제거(이중 표면 방지, 구멍은 안 냄)")
    ap.add_argument("--dense_snap", action="store_true",
                    help="관측 band 내 생성점을 제거 대신 관측 표면으로 거리가중 끌어당김(seam 완화)")
    ap.add_argument("--snap_band", type=float, default=0.0, help="스냅/제거 band(m). 0=자동(TSDF간격×3)")
    ap.add_argument("--no_mesh", action="store_true", help="Poisson 미리보기 메쉬 생략")
    args = ap.parse_args()
    if not (args.out or args.export_points or args.dense_points):
        ap.error("--out / --export_points / --dense_points 중 하나는 필요합니다")
    if args.out and args.no_mesh:
        args.out = ""                                    # no_mesh 면 메쉬 경로 무시

    # 출력 경로 안전 처리(기존 산출물 보존)
    out_path = os.path.expanduser(args.out) if args.out else None
    if out_path and os.path.exists(out_path) and not args.overwrite:
        import time
        out_path = out_path[:-4] + "_" + time.strftime("%Y%m%d_%H%M%S") + ".ply"
        print(f"[안전] 기존 파일 보존 → 새 경로: {out_path}")

    recon = o3d.io.read_triangle_mesh(args.recon)
    gen, gen_tm = load_gen_with_color(args.gen)          # 텍스처→정점색 베이킹
    assert len(recon.vertices) and len(gen.vertices), "메쉬 로드 실패"
    recon.compute_vertex_normals()

    rp = (recon.sample_points_poisson_disk(args.n_sample) if len(recon.triangles)
          else recon.sample_points_uniformly(args.n_sample))
    gp = gen.sample_points_uniformly(args.n_sample)
    R_pts, G_pts = np.asarray(rp.points), np.asarray(gp.points)
    dst_tree = cKDTree(R_pts)

    # 카메라 먼저 로드(RC 정합·carve 공용)
    cams = None
    if args.colmap and read_colmap is not None:
        cams = load_cams(args.colmap, args.masks_root, args.gid, args.stems, args.images)

    # --- 정합: up 정렬 + yaw 8-init + (an)isotropic ICP ---
    R0 = up_align_R(args.gen_up, args.world_up)
    best = None
    for yaw in range(0, 360, 45):
        Ry = rot_axis(args.world_up, yaw) @ R0
        gu = (Ry @ G_pts.T).T
        s0 = np.linalg.norm(R_pts.std(0)) / (np.linalg.norm(gu.std(0)) + 1e-9)
        S = np.array([s0, s0, s0]); t = R_pts.mean(0) - (Ry @ (G_pts * S).T).T.mean(0)
        R, t, S, rmse = aniso_icp(G_pts, dst_tree, R_pts, Ry, t, S,
                                  iters=20, trim=0.5, isotropic=args.isotropic)
        if best is None or rmse < best[-1]:
            best = (R, t, S, rmse)
    R, t, S, _ = best
    R, t, S, rmse = aniso_icp(G_pts, dst_tree, R_pts, R, t, S,
                              iters=50, trim=0.6, isotropic=args.isotropic)
    print(f"[정합] {'iso' if args.isotropic else '9-DoF'}  "
          f"scale=({S[0]:.3f},{S[1]:.3f},{S[2]:.3f})  trimmed-RMSE={rmse*1000:.1f}mm")

    # render-and-compare 미세정합(평면 관측 모호성 해소) — 변환 전 canonical 메쉬 사용
    if args.refine and cams is not None:
        R, t, S = refine_pose_rc(gen, cams, R, t, S, isotropic=args.isotropic)

    gv = np.asarray(gen.vertices)
    gen.vertices = o3d.utility.Vector3dVector(apply9(gv, R, t, S))
    gen.compute_vertex_normals()
    G_world = apply9(G_pts, R, t, S)

    if cams is not None:
        print(f"[진단] 최종 silhouette IoU={render_compare(gen, cams):.3f}")

    # dense 점군: 정합된 생성 메쉬 전체를 carve 없이 조밀 샘플(train→SDF prior)
    if args.dense_points:
        dpc = gen.sample_points_uniformly(number_of_points=args.dense_n)
        D = np.asarray(dpc.points)
        if args.dense_snap or args.dense_graft_only:
            med = np.median(dst_tree.query(
                R_pts[np.random.choice(len(R_pts), 2000)], k=2)[0][:, 1])
            band = args.snap_band if args.snap_band > 0 else med * 3
            d, idx = dst_tree.query(D)
            if args.dense_snap:                          # band 내 → 관측으로 거리가중 스냅
                w = np.clip(1 - d / band, 0, 1)[:, None]
                D = D + w * (R_pts[idx] - D)
                dpc.points = o3d.utility.Vector3dVector(D)
                print(f"[dense] band {band*1000:.0f}mm 내 {(d < band).sum()}점 관측으로 끌어당김")
            else:                                        # 근접점 제거(구멍 X)
                keep = d > band
                dpc = dpc.select_by_index(np.where(keep)[0])
                print(f"[dense] 관측근접 제거 {(~keep).sum()} → {len(dpc.points)}점")
        dpc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        dpc.orient_normals_consistent_tangent_plane(30)
        dp = os.path.expanduser(args.dense_points)
        os.makedirs(os.path.dirname(dp) or ".", exist_ok=True)
        o3d.io.write_point_cloud(dp, dpc)
        print(f"→ dense 생성 점군: {dp}  ({len(dpc.points)}점, 법선 포함, carve 없음)")

    # dense 만 요청되면 carve/융합 생략
    need_fusion = bool(args.export_points) or (out_path and not args.no_mesh)
    if not need_fusion:
        return

    # --- 융합: 미관측 이식(거리) ∩ free-space carve 통과 ---
    med = np.median(dst_tree.query(R_pts[np.random.choice(len(R_pts), 2000)], k=2)[0][:, 1])
    tau = args.graft_dist if args.graft_dist > 0 else med * 3
    gdist, _ = dst_tree.query(G_world)
    graft = gdist > tau                                        # TSDF 미커버(미관측 후보)
    if cams is not None:
        free = freespace_carve(G_world, cams, recon, margin=args.carve_margin)    # 빈공간 floater
        vh = visual_hull_carve(G_world, cams, recon, margin=args.carve_margin)    # 실루엣 모순
        remove = free | vh
        graft &= ~remove
        msg = (f"[융합] graft τ={tau*1000:.1f}mm  free-carve {free.sum()}  "
               f"vhull-carve {vh.sum()}")
        if args.isolate:
            before = graft.sum()
            graft = isolate_target(G_world, graft, dst_tree,
                                   eps=args.iso_eps, tau=args.iso_tau)
            msg += f"  isolate 제거 {before - graft.sum()}"
        print(msg + f"  최종 이식 {graft.sum()}/{len(G_world)} ({graft.mean()*100:.0f}%)")
    else:
        print(f"[융합] graft τ={tau*1000:.1f}mm  이식 {graft.sum()} (carve 미적용)")

    gp_w = gp.select_by_index(np.where(graft)[0])
    gp_w.points = o3d.utility.Vector3dVector(G_world[graft])
    for pc in (rp, gp_w):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pc.orient_normals_consistent_tangent_plane(30)

    # 미관측 prior 점군 export (augment_init_ply → train 최적화용)
    if args.export_points:
        pp = os.path.expanduser(args.export_points)
        os.makedirs(os.path.dirname(pp) or ".", exist_ok=True)
        o3d.io.write_point_cloud(pp, gp_w)
        print(f"→ 미관측 prior 점군: {pp}  ({len(gp_w.points)}점, 법선 포함)  "
              f"augment_init_ply 입력용")

    if not args.no_mesh and out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fused = rp + gp_w
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            fused, depth=args.poisson_depth)
        mesh.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), 0.02))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(out_path, mesh)
        print(f"→ 완결 메쉬(미리보기): {out_path}  (정점 {len(mesh.vertices)})")
    if args.save_aligned:
        base = (out_path or os.path.expanduser(args.export_points)
                or os.path.expanduser(args.dense_points))
        if gen_tm is not None:                            # 텍스처 보존 .glb 로 저장
            p = base[:-4] + "_gen_aligned.glb"
            tm2 = gen_tm.copy()
            tm2.vertices = apply9(np.asarray(gen_tm.vertices), R, t, S)
            tm2.export(p)
            print(f"→ 정합 생성메쉬(텍스처 유지): {p}")
        else:
            p = base[:-4] + "_gen_aligned.ply"
            o3d.io.write_triangle_mesh(p, gen); print(f"→ 정합 생성메쉬: {p}")


if __name__ == "__main__":
    main()
