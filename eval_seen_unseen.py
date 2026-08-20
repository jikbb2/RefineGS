#!/usr/bin/env python3
"""seen / unseen 분리 평가 — prior 가 '관측을 지키면서 미관측만 채웠는가'를 정량화.

단일 chamfer/F-score 는 "관측이 오염된 것"과 "생성이 부정확한 것"을 구분하지 못한다.
가시성 oracle 로 GT 점과 recon 점에 '동일한 규칙'으로 라벨을 붙이고 지표를 영역별로 쪼갠다.
oracle 은 3D 위치만의 함수(입력 카메라에서 보이는가)라 어떤 점 집합에도 같게 적용되고,
GT 라벨이 방법과 무관하게 고정되므로 baseline/ours 비교가 성립한다.

  기본 oracle = GT 씬 메쉬 raycasting(--vis_source gt_mesh):
    depth 파일 포맷·스케일 규약에 의존하지 않고 "입력 카메라 포즈에서 GT 씬이 보이는가"
    로만 정의 → 데이터셋 무관, 논문 프로토콜로 기술하기 좋다.
    (--vis_source gt_depth 로 depth 맵 oracle 도 선택 가능)

  라벨 (뷰별 판정 후 min_views 합의):
    seen      |z - d_gt| < margin   → 그 뷰에서 첫 표면 = 실제로 관측됨
    free      z < d_gt - margin     → 관측된 빈 공간 (recon 점만 해당; 확실한 오류)
    unseen    그 외(가려짐/FOV 밖)  → 정보 없음 = 생성이 채워야 할 영역

  지표:
    accuracy   (recon→GT)  : seen 영역은 baseline 과 같아야 함(관측 보존)
    completion (GT→recon)  : unseen 영역이 개선돼야 함(prior 기여)
    F@thr                  : precision=recon 점 기준, recall=GT 점 기준, 영역별

논문 주장 형태: "seen accuracy 유지 + unseen completion 개선".

  python eval_seen_unseen.py --gt_mesh gt_obj1.ply \
    --gt_scene_mesh /home/elicer/room_0/habitat/mesh_semantic.ply \
    --recon output/.../1/train/ours_7000/fuse_post.ply \
    --recon2 output/.../1/train/ours_7000/fused_prior.ply \
    --colmap data/replica_room0_v2/sparse/0 --gid 1 \
    --stems ~/See3D/dataset/stage6/clean_stems/1.txt
"""
import os
import glob
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image

try:
    from warp_gt_to_pose import read_colmap, cam_center
except Exception:                                    # repo 밖에서 실행 시
    read_colmap = None

    def cam_center(R, t):
        return -R.T @ t


# --------------------------------------------------------------------------
def load_gt_depth(depth_dir, stem, scale):
    """GT depth 로드(sdf_distill_depth.load_gt_depth 와 동일 규약)."""
    for c in (stem.replace("frame", "depth"), stem, stem + "_depth"):
        for ext in (".png", ".npy"):
            p = os.path.join(os.path.expanduser(depth_dir), c + ext)
            if not os.path.exists(p):
                continue
            if ext == ".npy":
                return np.load(p).astype(np.float32)
            return np.array(Image.open(p)).astype(np.float32) / scale
    return None


def raycast_depth(scene, cam, ds):
    """GT 씬 메쉬를 카메라 포즈에서 ray casting → z-depth (권장 가시성 oracle).

    depth 맵 파일·스케일 규약에 의존하지 않고 '입력 카메라 포즈에서 GT 씬이 보이는가'
    만으로 가시성을 정의하므로 데이터셋 무관하게 성립 — 논문 프로토콜로 기술하기 좋다.
    방향벡터를 정규화하지 않으면(z성분=1) t_hit 이 곧 카메라 z-depth 라 규약이 일치한다.
    ※ occlusion 은 '씬 전체'가 결정하므로 scene 은 객체가 아닌 전체 GT 메쉬여야 한다.
    """
    W = int(np.ceil(cam["W"] / ds)); H = int(np.ceil(cam["H"] / ds))
    fx, fy = cam["fx"] / ds, cam["fy"] / ds
    cx, cy = cam["cx"] / ds, cam["cy"] / ds
    R, t = cam["R"], cam["t"]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dcam = np.stack([(uu - cx) / fx, (vv - cy) / fy,
                     np.ones_like(uu, float)], -1).reshape(-1, 3)
    dwn = (R.T @ dcam.T).T                             # world 방향(비정규화)
    C = cam_center(R, t)
    rays = np.concatenate([np.broadcast_to(C.astype(np.float32), dwn.shape),
                           dwn.astype(np.float32)], 1)
    th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)
    return np.where(np.isfinite(th), th, 0.0).astype(np.float32)


def load_mask(masks_root, gid, stem):
    p = os.path.join(masks_root, str(gid), "masks", stem + ".png")
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p))
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., 3]
    elif a.ndim == 3:
        a = np.array(Image.open(p).convert("L"))
    if a.max() <= 1:
        return a > 0
    if (a == 188).any():
        return a == 188                              # amodal 규약: 188=visible
    return a > 127


def build_views(args):
    """카메라 + GT depth(+마스크) 뷰 버퍼. 다운스케일로 메모리/속도 확보."""
    assert read_colmap is not None, "warp_gt_to_pose 임포트 실패 — repo 루트에서 실행하세요"
    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    if args.stems and os.path.exists(os.path.expanduser(args.stems)):
        stems = [l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()]
    else:
        stems = sorted(cams)
    stems = [s for s in stems if s in cams]
    if args.n_views > 0 and len(stems) > args.n_views:      # 균등 간격 서브샘플
        idx = np.linspace(0, len(stems) - 1, args.n_views).round().astype(int)
        stems = [stems[i] for i in np.unique(idx)]

    ds = max(1, args.ds)
    rc_scene = None
    if args.vis_source == "gt_mesh":
        sp = os.path.expanduser(args.gt_scene_mesh or args.gt_mesh)
        sm = o3d.io.read_triangle_mesh(sp)
        assert len(sm.triangles), f"씬 메쉬 로드 실패(삼각형 없음): {sp}"
        rc_scene = o3d.t.geometry.RaycastingScene()
        rc_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sm))
        print(f"[oracle] GT 메쉬 raycasting — {os.path.basename(sp)} "
              f"(tri {len(sm.triangles)})" +
              ("" if args.gt_scene_mesh else "  ⚠ 객체 메쉬 사용 중 — 타 객체에 의한 "
                                            "가림이 반영되지 않음. --gt_scene_mesh 권장"))
    else:
        print("[oracle] GT depth 맵 (--vis_source gt_depth)")

    views, n_mask = [], 0
    for s in stems:
        c = cams[s]
        if rc_scene is not None:
            dg = raycast_depth(rc_scene, c, ds)
        else:
            dg = load_gt_depth(args.gt_depth_dir, s, args.gt_depth_scale)
            dg = dg[::ds, ::ds] if dg is not None else None
        if dg is None:
            continue
        m = load_mask(args.masks_root, args.gid, s) if args.masks_root else None
        if m is not None and m.shape != dg.shape:
            m = np.array(Image.fromarray(m.astype(np.uint8))
                         .resize((dg.shape[1], dg.shape[0]), Image.NEAREST)) > 0
        v = {"R": c["R"], "t": c["t"],
             "fx": c["fx"] / ds, "fy": c["fy"] / ds,
             "cx": c["cx"] / ds, "cy": c["cy"] / ds,
             "dgt": dg}
        v["H"], v["W"] = v["dgt"].shape
        if m is not None:
            v["mask"] = m if m.shape == dg.shape else m[::ds, ::ds]
            n_mask += 1
        views.append(v)
    print(f"[views] {len(views)}뷰, 마스크 {n_mask}뷰, ds={ds}")
    assert views, "가시성 뷰 0개 — --vis_source / 경로·파일명 규약 확인"
    return views


def classify(P, views, margin, min_views, use_mask):
    """점 라벨: seen(첫 표면 일치) / free(관측된 빈 공간) / unseen(정보 없음).
    반환 (seen: bool, free: bool) — free 는 seen 이 아닌 점에만 True."""
    n_seen = np.zeros(len(P), np.int32)
    n_free = np.zeros(len(P), np.int32)
    for v in views:
        Xc = P @ v["R"].T + v["t"]
        z = Xc[:, 2]
        zz = np.maximum(z, 1e-6)
        u = v["fx"] * Xc[:, 0] / zz + v["cx"]
        w = v["fy"] * Xc[:, 1] / zz + v["cy"]
        infr = (z > 0.05) & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
        ui = np.clip(u, 0, v["W"] - 1).astype(int)
        wi = np.clip(w, 0, v["H"] - 1).astype(int)
        d = v["dgt"][wi, ui]
        ok = infr & (d > 0.01)
        if use_mask and "mask" in v:
            ok = ok & v["mask"][wi, ui]
        n_seen += (ok & (np.abs(z - d) < margin)).astype(np.int32)
        n_free += (ok & (z < d - margin)).astype(np.int32)
    seen = n_seen >= min_views
    free = (~seen) & (n_free >= min_views)
    return seen, free


def sample(mesh_path, n):
    m = o3d.io.read_triangle_mesh(os.path.expanduser(mesh_path))
    assert len(m.vertices), f"메쉬 로드 실패: {mesh_path}"
    if len(m.triangles) == 0:
        return np.asarray(m.vertices)
    pc = m.sample_points_uniformly(number_of_points=n)
    return np.asarray(pc.points)


def _stat(d):
    return (float(d.mean() * 1000), float(np.median(d) * 1000)) if len(d) else (float("nan"),) * 2


def report(name, R, G, gs, gf, thresholds, views, args):
    """recon 점군 R, GT 점군 G, GT 라벨(gs=seen) 로 영역별 지표 출력."""
    rs, rf = classify(R, views, args.margin, args.min_views, args.use_mask)
    dR = cKDTree(G).query(R, workers=-1)[0]          # accuracy 용 (recon→GT)
    dG = cKDTree(R).query(G, workers=-1)[0]          # completion 용 (GT→recon)

    print(f"\n===== {name} =====")
    print(f"  점 구성  recon: seen {rs.mean()*100:5.1f}%  free위반 {rf.mean()*100:5.1f}%  "
          f"unseen {(~rs & ~rf).mean()*100:5.1f}%   |  GT: seen {gs.mean()*100:.1f}%")
    for lab, rm, gm in (("ALL   ", np.ones(len(R), bool), np.ones(len(G), bool)),
                        ("SEEN  ", rs, gs),
                        ("UNSEEN", ~rs, ~gs)):
        am, amd = _stat(dR[rm])
        cm, cmd = _stat(dG[gm])
        line = (f"  [{lab}] accuracy {am:7.2f}mm (med {amd:6.2f})   "
                f"completion {cm:7.2f}mm (med {cmd:6.2f})")
        print(line)
        for thr in thresholds:
            p = float((dR[rm] < thr).mean()) if rm.sum() else float("nan")
            r = float((dG[gm] < thr).mean()) if gm.sum() else float("nan")
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            print(f"        F@{thr*100:.1f}cm {f:.4f} (P {p:.4f} / R {r:.4f})")
    if rf.any():
        am, _ = _stat(dR[rf])
        print(f"  [FREE 위반] {int(rf.sum())}점 ({rf.mean()*100:.1f}%) — "
              f"관측된 빈 공간의 표면, accuracy {am:.2f}mm  ※ 명백한 오류")
    return dict(seen_acc=_stat(dR[rs])[0], unseen_comp=_stat(dG[~gs])[0],
                free_pct=float(rf.mean() * 100))


def main():
    ap = argparse.ArgumentParser(description="seen/unseen 분리 평가")
    ap.add_argument("--gt_mesh", required=True, help="해당 객체의 GT 메쉬(추출본)")
    ap.add_argument("--recon", required=True, help="비교 A (보통 fuse_post.ply)")
    ap.add_argument("--recon2", default="", help="비교 B (보통 fused_prior.ply)")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--gid", required=True)
    ap.add_argument("--masks_root", default="")
    ap.add_argument("--stems", default="")
    ap.add_argument("--vis_source", default="gt_mesh", choices=["gt_mesh", "gt_depth"],
                    help="가시성 oracle. gt_mesh=GT 씬 메쉬 raycasting(권장, depth 파일 불요) "
                         "/ gt_depth=depth 맵 파일")
    ap.add_argument("--gt_scene_mesh", default="",
                    help="[gt_mesh] occlusion 판정용 '씬 전체' GT 메쉬(예: mesh_semantic.ply). "
                         "미지정 시 --gt_mesh 사용(타 객체 가림 미반영)")
    ap.add_argument("--gt_depth_dir", default="", help="[gt_depth] depth 맵 폴더")
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--n_views", type=int, default=120, help="사용할 뷰 수(균등 서브샘플, 0=전체)")
    ap.add_argument("--ds", type=int, default=2, help="depth/mask 다운스케일")
    ap.add_argument("--n_sample", type=int, default=200000)
    ap.add_argument("--margin", type=float, default=0.015,
                    help="가시 판정 허용오차(m) — GT depth 노이즈+이산화 여유")
    ap.add_argument("--min_views", type=int, default=1,
                    help="seen 판정 최소 뷰 수(1=한 뷰라도 봤으면 관측)")
    ap.add_argument("--use_mask", action="store_true",
                    help="객체 마스크도 가시 조건에 포함(인접 객체 depth 혼입 방지)")
    ap.add_argument("--thresholds", default="0.005,0.01,0.02")
    args = ap.parse_args()

    thr = [float(x) for x in args.thresholds.split(",")]
    if args.vis_source == "gt_depth" and not args.gt_depth_dir:
        ap.error("--vis_source gt_depth 에는 --gt_depth_dir 가 필요합니다")
    views = build_views(args)

    G = sample(args.gt_mesh, args.n_sample)
    gs, _ = classify(G, views, args.margin, args.min_views, args.use_mask)
    print(f"[GT] {len(G)}점 — seen {gs.mean()*100:.1f}% / unseen {(~gs).mean()*100:.1f}%")
    if gs.mean() > 0.98:
        print("  ⚠ unseen 이 거의 없음 — margin 과다 또는 뷰/depth 매칭 확인")

    a = report("A: " + os.path.basename(args.recon),
               sample(args.recon, args.n_sample), G, gs, None, thr, views, args)
    if args.recon2:
        b = report("B: " + os.path.basename(args.recon2),
                   sample(args.recon2, args.n_sample), G, gs, None, thr, views, args)
        print("\n===== A → B 변화 (원하는 방향: seen acc 유지, unseen comp 감소) =====")
        print(f"  seen accuracy      {a['seen_acc']:7.2f} → {b['seen_acc']:7.2f} mm  "
              f"({b['seen_acc']-a['seen_acc']:+.2f}, 0 에 가까울수록 관측 보존)")
        print(f"  unseen completion  {a['unseen_comp']:7.2f} → {b['unseen_comp']:7.2f} mm  "
              f"({b['unseen_comp']-a['unseen_comp']:+.2f}, 음수가 prior 기여)")
        print(f"  free 위반 비율     {a['free_pct']:6.2f}% → {b['free_pct']:6.2f}%  "
              f"({b['free_pct']-a['free_pct']:+.2f}%p, 낮을수록 좋음)")


if __name__ == "__main__":
    main()