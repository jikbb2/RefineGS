#!/usr/bin/env python3
"""RefineGS 객체 → ShapeR 입력 pkl 변환기.

ShapeR(Meta FAIR, arXiv 2601.11514)은 posed multi-view + **metric sparse point cloud**를
조건으로 SDF를 생성한다. 우리가 겪은 세 문제를 동시에 겨냥한다:
  - shape 오차: 포인트 클라우드가 1급 조건 → 생성 형상이 관측 기하에 앵커됨
  - 필드 추출 해킹: SDF 디코더 출력 → sign-fix / shell_delta / flood-fill 전부 불필요
  - 접합부 끊김: 관측점 근처를 지나도록 조건화 → seam 감소

스키마는 ShapeR `dataset/shaper_dataset.py` + `dataset/image_processor.py` 를 읽고 매핑.
필수 키(SLAM 경로, strategy="cluster"):
  points_model            (N,3) torch  오브젝트 프레임 metric 포인트
  bounds                  (3,)  torch  half-extent → scale = 0.9/max(bounds)
  inv_dist_std, dist_std  (N,)  torch  포인트별 불확실성(작을수록 신뢰)
  image_data              list[bytes]  인코딩된 이미지(PIL 로 열림, "L" 변환)
  Ts_camera_model         list[(4,4) torch]  model→camera
  camera_params           list[(3,3)]  ※ 핀홀 K. 원본은 Fisheye624 파라미터라
                                        infer_shape_pinhole.py 의 패치가 필요
  object_point_projections list[(M,2) torch]  객체 포인트의 uv (crop 기준)
  visible_points_model    list[(M,3)]  뷰별 가시 포인트(뷰 선택 점수)
  T_model_world           (4,4) torch  world→model (--do_transform_to_world 용)
  caption / category      str
선택:
  mesh_vertices, mesh_faces  GT (평가용)

  python make_shaper_input.py --gid 1 \
    --recon output/replica_room0_v2/refinegs_full/1/train/ours_7000/fuse_post.ply \
    --colmap data/replica_room0_v2/sparse/0 \
    --images data/replica_room0_v2/images \
    --masks_root data/replica_room0_v2/masks \
    --stems ~/See3D/dataset/stage6/clean_stems/1.txt \
    --caption "a sofa" --out ~/ShapeR/data/refinegs_obj1.pkl
"""
import os
import io
import glob
import pickle
import argparse

import numpy as np
import open3d as o3d
import torch
from PIL import Image

try:
    from warp_gt_to_pose import read_colmap, cam_center
except Exception:                                     # repo 밖에서 실행 시
    read_colmap = None

    def cam_center(R, t):
        return -R.T @ t


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
        return a == 188                               # amodal 규약: 188=visible
    return a > 127


def load_depth_map(depth_dir, stem, scale):
    """stem → depth(meters). 이름 규약: frameNNNN→depthNNNN / 동일이름 / _depth."""
    for c in (stem.replace("frame", "depth"), stem, stem + "_depth"):
        for ext in (".png", ".npy"):
            p = os.path.join(os.path.expanduser(depth_dir), c + ext)
            if not os.path.exists(p):
                continue
            if ext == ".npy":
                return np.load(p).astype(np.float32)
            return np.array(Image.open(p)).astype(np.float32) / scale
    return None


def filter_observed(P_w, stems, cams, args):
    """관측이 확인된 점만 남긴다.

    ShapeR 는 포인트를 geometric anchor 로 '충실히' 따른다. 그런데 recon(fuse_post.ply)
    은 미관측 영역에 부정확한 덩어리·floater 를 갖고 있어, 그대로 주면 생성물이 그
    퍼진 형태를 재현한다(obj6: 다리가 성긴 그물로 생성됨).
    → 각 점이 '어느 뷰에서 실제로 관측된 표면'인지(|z-depth|<margin, 마스크 안) 검사해
      확인된 점만 조건으로 준다. 미관측 영역은 점 없음 = 모델이 자유롭게 추론.
    """
    n_seen = np.zeros(len(P_w), np.int32)
    n_dep = n_msk = 0
    for s in stems:
        D = load_depth_map(args.depth_dir, s, args.depth_scale)
        if D is None:
            continue
        n_dep += 1
        # 마스크도 함께 봐야 한다 — depth 일치만 보면 바닥·인접 객체 표면 위의 점도
        # '관측됨'으로 통과해 버린다(그 점들이 생성물을 오염시킨다)
        M = load_mask(args.masks_root, args.gid, s) if args.masks_root else None
        if M is not None:
            n_msk += 1
        c = cams[s]
        Hd, Wd = D.shape
        sx, sy = Wd / c["W"], Hd / c["H"]
        Xc = P_w @ c["R"].T + c["t"]
        z = Xc[:, 2]
        zz = np.maximum(z, 1e-6)
        u = (c["fx"] * Xc[:, 0] / zz + c["cx"]) * sx
        v = (c["fy"] * Xc[:, 1] / zz + c["cy"]) * sy
        ok = (z > 0.05) & (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd)
        if not ok.any():
            continue
        ui = np.clip(u, 0, Wd - 1).astype(int); vi = np.clip(v, 0, Hd - 1).astype(int)
        d = D[vi, ui]
        hit = ok & (d > 0.01) & (np.abs(z - d) < args.seen_margin)
        if M is not None:                                  # 객체 마스크 안이어야 함
            if M.shape != (Hd, Wd):
                M = np.array(Image.fromarray(M.astype(np.uint8))
                             .resize((Wd, Hd), Image.NEAREST)) > 0
            hit &= M[vi, ui]
        n_seen += hit.astype(np.int32)
    keep = n_seen >= args.seen_min_views
    print(f"[filter] 관측 확인 점 {int(keep.sum())}/{len(P_w)} "
          f"({keep.mean()*100:.1f}%)  depth 뷰 {n_dep}, 마스크 {n_msk}뷰, "
          f"기준 |z-d|<{args.seen_margin*1000:.0f}mm ∧ 마스크 안 이 {args.seen_min_views}뷰 이상")
    if keep.sum() < 200:
        print("  ⚠ 남은 점이 너무 적음 — depth 경로/스케일 확인. 필터 미적용")
        return P_w
    return P_w[keep]


def sample_free_points(stems, cams, center, R_align, bounds, args, n_target=6000):
    """관측된 빈 공간 샘플(오브젝트 프레임).

    카메라~관측 표면 사이 구간은 '비어 있음이 관측된' 곳이다. 이 점들을 pkl 에 실어
    생성 과정의 free-space 구속으로 쓰면, 다리 밑 같은 관측 가능 영역의 할루시네이션이
    원천 차단된다(융합 단계 carve 로 지우는 것보다 앞선 개입).
    """
    rng = np.random.default_rng(0)
    per = max(64, n_target // max(1, len(stems)))
    out = []
    for s in stems:
        D = load_depth_map(args.depth_dir, s, args.depth_scale)
        if D is None:
            continue
        c = cams[s]
        Hd, Wd = D.shape
        sx, sy = Wd / c["W"], Hd / c["H"]
        C = -c["R"].T @ c["t"]
        vs, us = np.nonzero(D > 0.05)
        if len(vs) == 0:
            continue
        k = min(per * 3, len(vs))
        sel = rng.choice(len(vs), k, replace=False)
        v_, u_ = vs[sel], us[sel]
        d = D[v_, u_]
        x = ((u_ / sx) - c["cx"]) / c["fx"]
        y = ((v_ / sy) - c["cy"]) / c["fy"]
        dirs = np.stack([x, y, np.ones_like(x)], 1) @ c["R"]      # world 방향(z성분=1 규약)
        tau = rng.uniform(0.25, 0.95, k) * np.maximum(d - 2 * args.seen_margin, 1e-3)
        P = C[None] + dirs * tau[:, None]
        Pm = (R_align @ (P - center).T).T
        keep = (np.abs(Pm) <= bounds).all(1)                      # 오브젝트 bbox 안만
        if keep.any():
            out.append(Pm[keep])
    if not out:
        print("[free] 샘플 0개 — depth 경로 확인")
        return np.zeros((0, 3), np.float32)
    F = np.concatenate(out)
    if len(F) > n_target:
        F = F[rng.choice(len(F), n_target, replace=False)]
    print(f"[free] 관측된 빈 공간 샘플 {len(F)}점 (오브젝트 bbox 내)")
    return F.astype(np.float32)


def find_image(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="RefineGS 객체 → ShapeR 입력 pkl")
    ap.add_argument("--gid", required=True)
    ap.add_argument("--recon", required=True, help="관측 객체 메쉬(fuse_post.ply)")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks_root", default="")
    ap.add_argument("--stems", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption", default="", help="비우면 'a 3D object'")
    ap.add_argument("--n_points", type=int, default=20000,
                    help="포인트 수. ※ 도메인 갭 A/B: 조밀(20000) vs SLAM 밀도(1500)")
    ap.add_argument("--n_views", type=int, default=32,
                    help="pkl 에 담을 후보 뷰 수(ShapeR 이 여기서 16개 선택)")
    ap.add_argument("--world_up", default="z", choices=["x", "y", "z"])
    ap.add_argument("--bounds_margin", type=float, default=1.15,
                    help="half-extent 여유. 1.0 이면 미관측 확장분이 |p|>1 로 잘릴 수 있음")
    ap.add_argument("--point_std", type=float, default=1e-3,
                    help="포인트 불확실성(inv_dist_std/dist_std). 우리 depth 는 SLAM 반정밀"
                         "보다 정확하므로 작게. 필터가 걸러내지 않도록")
    ap.add_argument("--img_max_side", type=int, default=640,
                    help="이미지 다운스케일 긴 변(ShapeR 추론은 280px 라 큰 해상도 무의미)")
    ap.add_argument("--seed", type=int, default=0,
                    help="포인트 샘플링 시드. 고정해야 같은 명령이 같은 pkl 을 낸다 — "
                         "설정 A/B 를 비교하려면 필수")
    ap.add_argument("--gt_mesh", default="", help="선택: GT 메쉬(평가용)")
    ap.add_argument("--depth_dir", default="",
                    help="[관측 필터] depth 폴더. 지정 시 '관측이 확인된 점'만 조건으로 준다 — "
                         "recon 의 미관측 영역 쓰레기가 생성물을 오염시키는 것을 막는다")
    ap.add_argument("--depth_scale", type=float, default=6553.5)
    ap.add_argument("--seen_margin", type=float, default=0.02,
                    help="관측 판정 허용오차(m): |z - depth| < margin")
    ap.add_argument("--seen_min_views", type=int, default=2,
                    help="관측 판정 최소 뷰 수")
    ap.add_argument("--free_points", type=int, default=6000,
                    help="pkl 에 실을 '관측된 빈 공간' 샘플 수(0=off). "
                         "shaper_field.py --guide_free_w 로 생성 구속에 사용")
    args = ap.parse_args()

    assert read_colmap is not None, "warp_gt_to_pose 임포트 실패 — RefineGS 루트에서 실행하세요"

    # ---- 1) 관측 포인트 (world, metric) ----
    m = o3d.io.read_triangle_mesh(os.path.expanduser(args.recon))
    assert len(m.vertices), f"recon 로드 실패: {args.recon}"
    # ※ 재현성: 포인트 샘플링이 시드 없이 매번 달라지면, 같은 명령이어도 조건 포인트가
    #   바뀌어 생성 결과가 달라진다(ShapeR 는 포인트를 anchor 로 충실히 따름).
    #   실제로 파이프라인의 주된 변동 원인은 ShapeR 샘플링이 아니라 여기다.
    _seeded = False
    try:
        o3d.utility.random.seed(args.seed)             # Open3D >= 0.16
        _seeded = True
    except Exception:
        pass
    try:
        pc = m.sample_points_uniformly(args.n_points, seed=args.seed)
        _seeded = True
    except TypeError:
        pc = m.sample_points_uniformly(args.n_points)
    P_w = np.asarray(pc.points, np.float64)
    print(f"[points] {len(P_w)}점 (world, seed={args.seed}"
          + ("" if _seeded else ", ⚠ 시드 미지원 Open3D — 실행마다 달라짐") + ")")

    # ---- 2) 오브젝트 프레임: 중력 정렬 + AABB 중심 ----
    # centroid 대신 AABB 중심 — 절반 미관측 시 centroid 는 관측 쪽으로 크게 치우침.
    # ※ AABB 는 '샘플 점'이 아니라 '메쉬 정점'에서 계산한다 — 그래야 --n_points 를 바꿔도
    #   오브젝트 프레임(center/bounds/scale)이 동일해져 앙상블 평균이 유효하다.
    V_w = np.asarray(m.vertices, np.float64)
    lo, hi = V_w.min(0), V_w.max(0)
    center = (lo + hi) / 2
    R_align = np.eye(3)
    if args.world_up != "z":                          # ShapeR 은 z-up 오브젝트 프레임 가정
        ax = {"x": 0, "y": 1}[args.world_up]
        perm = [0, 1, 2]; perm[ax], perm[2] = perm[2], perm[ax]
        R_align = np.eye(3)[perm]
    bounds = np.abs((R_align @ (V_w - center).T).T).max(0) * args.bounds_margin
    scale = 0.9 / bounds.max()
    T_model_world = np.eye(4)                          # world → model
    T_model_world[:3, :3] = R_align
    T_model_world[:3, 3] = -R_align @ center

    # ---- 3) 카메라 / 뷰 ----
    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    if args.stems and os.path.exists(os.path.expanduser(args.stems)):
        stems = [l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()]
    else:
        stems = sorted(cams)
    stems = [s for s in stems if s in cams and find_image(args.images, s)]
    assert stems, "사용 가능한 뷰 없음 — --images / --stems 확인"

    # ---- 2b) 관측 필터 (프레임은 메쉬 정점 기준이라 필터와 무관하게 고정) ----
    F_m = np.zeros((0, 3), np.float32)
    if args.depth_dir:
        P_w = filter_observed(P_w, stems, cams, args)
        if args.free_points > 0:
            F_m = sample_free_points(stems, cams, center, R_align, bounds, args,
                                     n_target=args.free_points)
    P_m = (R_align @ (P_w - center).T).T
    clipped = int((np.abs(P_m * scale) > 1.0).any(1).sum())
    print(f"[frame] center={np.round(center,3)}  half-extent={np.round(bounds,3)}m  "
          f"scale={scale:.3f}  (정규화 후 클리핑될 점 {clipped})")
    if len(stems) > args.n_views:                      # 균등 간격 서브샘플
        idx = np.unique(np.linspace(0, len(stems) - 1, args.n_views).round().astype(int))
        stems = [stems[i] for i in idx]

    image_data, Ts_cm, cam_params, obj_uv, vis_pts = [], [], [], [], []
    n_mask = 0
    for s in stems:
        c = cams[s]
        img = Image.open(find_image(args.images, s)).convert("L")
        W0, H0 = img.size
        sc = min(1.0, args.img_max_side / max(W0, H0))
        if sc < 1.0:
            img = img.resize((int(round(W0 * sc)), int(round(H0 * sc))), Image.LANCZOS)
        W, H = img.size
        fx, fy = c["fx"] * sc, c["fy"] * sc
        cx, cy = c["cx"] * sc, c["cy"] * sc

        # model → camera : T_cm = T_cw @ T_wm
        T_cw = np.eye(4); T_cw[:3, :3] = c["R"]; T_cw[:3, 3] = c["t"]
        T_cm = T_cw @ np.linalg.inv(T_model_world)

        # 객체 포인트 투영(핀홀) — crop 기준 + 뷰 선택 점수
        Xc = P_m @ T_cm[:3, :3].T + T_cm[:3, 3]
        z = Xc[:, 2]
        ok = z > 1e-6
        u = np.full(len(P_m), -1.0); v = np.full(len(P_m), -1.0)
        u[ok] = fx * Xc[ok, 0] / z[ok] + cx
        v[ok] = fy * Xc[ok, 1] / z[ok] + cy
        infr = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if args.masks_root:                            # 객체 마스크로 가시성 정제
            mk = load_mask(args.masks_root, args.gid, s)
            if mk is not None:
                n_mask += 1
                if mk.shape != (H, W):
                    mk = np.array(Image.fromarray(mk.astype(np.uint8))
                                  .resize((W, H), Image.NEAREST)) > 0
                ui = np.clip(u, 0, W - 1).astype(int); vi = np.clip(v, 0, H - 1).astype(int)
                infr &= mk[vi, ui]
        if infr.sum() < 20:                            # 객체가 거의 안 보이는 뷰는 제외
            continue

        buf = io.BytesIO(); img.save(buf, format="PNG")
        image_data.append(buf.getvalue())
        Ts_cm.append(torch.tensor(T_cm, dtype=torch.float32))
        cam_params.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32))
        obj_uv.append(torch.tensor(np.stack([u[infr], v[infr]], 1), dtype=torch.float32))
        vis_pts.append(P_m[infr].astype(np.float32))
    assert image_data, "유효 뷰 0개 — 마스크/포즈 확인"
    print(f"[views] {len(image_data)}뷰 (마스크 적용 {n_mask}), "
          f"가시점 중앙값 {int(np.median([len(x) for x in vis_pts]))}")

    # ---- 4) pkl 조립 ----
    N = len(P_m)
    sample = {
        "points_model": torch.tensor(P_m, dtype=torch.float32),
        "bounds": torch.tensor(bounds, dtype=torch.float32),
        "inv_dist_std": torch.full((N,), args.point_std, dtype=torch.float32),
        "dist_std": torch.full((N,), args.point_std, dtype=torch.float32),
        "image_data": image_data,
        "Ts_camera_model": torch.stack(Ts_cm),
        "camera_params": np.stack(cam_params),         # 3x3 K (핀홀 패치 필요)
        "object_point_projections": obj_uv,
        "visible_points_model": vis_pts,
        "T_model_world": torch.tensor(T_model_world, dtype=torch.float32),
        "caption": args.caption or "a 3D object",
        "is_ariagen2": False,
        "pinhole": True,                               # ← 패치가 이 플래그로 rectify 우회
    }
    if len(F_m):                                       # 생성 단계 free-space 구속용
        sample["free_points_model"] = torch.tensor(F_m, dtype=torch.float32)
    if args.gt_mesh:
        gm = o3d.io.read_triangle_mesh(os.path.expanduser(args.gt_mesh))
        if len(gm.vertices):
            gv = (R_align @ (np.asarray(gm.vertices) - center).T).T
            sample["mesh_vertices"] = torch.tensor(gv, dtype=torch.float32)
            sample["mesh_faces"] = torch.tensor(np.asarray(gm.triangles), dtype=torch.int64)
            print(f"[gt] 평가용 GT 포함 (verts {len(gv)})")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(sample, fh)
    print(f"→ {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("  실행: python infer_shape_pinhole.py --input_pkl "
          f"{os.path.basename(out)} --config balance --do_transform_to_world")


if __name__ == "__main__":
    main()