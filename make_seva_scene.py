#!/usr/bin/env python3
"""SEVA(stable-virtual-camera) 씬 빌더 — 객체별 GT 프레임 + unknown 지향 타깃 궤적.

출력(ReconfusionParser 규약):
  <out>/<scene>/
    images/000000.png ...        입력=GT 프레임, 타깃=더미 검정
    transforms.json              전역 intrinsics + frames[].transform_matrix (c2w, **OpenGL**)
    train_test_split_<P>.json    train_ids=입력(GT), test_ids=타깃(궤적)

타깃 궤적: unknown 점군의 평균 법선 방향(그 면이 향하는 쪽)에서 객체를 바라보도록 배치.
입력 프레임: clean_stems 정화 목록에서 균등 P장.

  python make_seva_scene.py --gid 6 \
    --masks_root data/replica_room0_v2/masks \
    --images data/replica_room0_v2/images \
    --colmap data/replica_room0_v2/sparse/0 \
    --stems ~/See3D/dataset/stage6/clean_stems/6.txt \
    --unknown ~/See3D/dataset/obj6/unknown.ply \
    --n_input 16 --n_target 40 \
    --out ~/See3D/dataset/seva --scene obj6

RefineGS 루트에서 실행. Deps: numpy, open3d, PIL, (repo) warp_gt_to_pose.
"""
import os
import json
import shutil
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from warp_gt_to_pose import read_colmap, cam_center

CV2GL = np.diag([1.0, -1.0, -1.0])     # CV(+Y down,+Z fwd) → OpenGL(+Y up,+Z back)


def look_at_c2w(pos, target, up_hint=np.array([0, 0, 1.0])):
    """CV 규약 c2w (x right, y down, z forward)."""
    f = target - pos; f /= np.linalg.norm(f) + 1e-9
    if abs(f @ up_hint) > 0.95:
        up_hint = np.array([0, 1.0, 0])
    r = np.cross(f, up_hint); r /= np.linalg.norm(r) + 1e-9
    d = np.cross(f, r)
    R = np.stack([r, d, f], axis=1)     # 열 = 카메라 축의 world 표현
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = pos
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gid", required=True)
    ap.add_argument("--masks_root", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--stems", default="", help="clean_stems/<gid>.txt (없으면 masks 폴더 전체)")
    ap.add_argument("--unknown", required=True, help="make_unknown_points 출력 ply")
    ap.add_argument("--n_input", type=int, default=16)
    ap.add_argument("--n_target", type=int, default=40)
    ap.add_argument("--radius_scale", type=float, default=2.2, help="객체 반경 대비 카메라 거리")
    ap.add_argument("--spread_deg", type=float, default=50.0,
                    help="unknown 평균 법선 주변으로 카메라를 퍼뜨릴 각도(원뿔 반각)")
    ap.add_argument("--up_axis", type=int, default=2, help="월드 up 축(0=x,1=y,2=z)")
    ap.add_argument("--cam_below", type=float, default=0.15,
                    help="관측 카메라 최저 높이에서 이만큼까지만 내려갈 수 있음(바닥 관통 방지)")
    ap.add_argument("--min_area_frac", type=float, default=0.15,
                    help="입력 후보: 마스크 면적이 최대의 이 비율 이상인 프레임만")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="")
    args = ap.parse_args()

    scene = args.scene or f"obj{args.gid}"
    root = os.path.join(os.path.expanduser(args.out), scene)
    img_dir = os.path.join(root, "images")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(img_dir, exist_ok=True)

    # ── 입력 GT 프레임 선별 ──
    if args.stems and os.path.exists(os.path.expanduser(args.stems)):
        stems = [l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()]
    else:
        import glob as _g
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in _g.glob(os.path.join(args.masks_root, args.gid, "images", "*")))
    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    stems = [s for s in stems if s in cams]
    assert len(stems) >= 2, f"유효 프레임 부족: {len(stems)}"

    # [v2] 마스크 면적 기준 선별 — 객체가 크게 보이는 프레임 우선(참조 품질↑),
    #      그중 시점 다양성을 위해 카메라 위치로 균등 서브샘플
    areas = []
    for s in stems:
        mp = os.path.join(args.masks_root, args.gid, "masks", s + ".png")
        a = 0
        if os.path.exists(mp):
            im = np.array(Image.open(mp))
            al = im[..., 3] if (im.ndim == 3 and im.shape[2] == 4) else (
                np.array(Image.open(mp).convert("L")))
            a = int((al > (0 if al.max() <= 1 else 127)).sum())
        areas.append(a)
    areas = np.array(areas, float)
    if areas.max() > 0:
        cand = [s for s, a in zip(stems, areas) if a >= max(areas.max() * args.min_area_frac, 1)]
        print(f"마스크 면적 필터: {len(cand)}/{len(stems)} (최대 면적의 {args.min_area_frac:.0%} 이상)")
    else:
        cand = stems
    if len(cand) < args.n_input:
        cand = stems
    # 카메라 위치 기준 균등 선별(시점 다양성)
    C = np.stack([cam_center(cams[s]["R"], cams[s]["t"]) for s in cand])
    picked = [int(np.argmax(np.linalg.norm(C - C.mean(0), axis=1)))]
    for _ in range(min(args.n_input, len(cand)) - 1):
        d = np.min(np.linalg.norm(C[:, None] - C[picked][None], axis=-1), axis=1)
        picked.append(int(np.argmax(d)))          # farthest-point sampling
    sel = [cand[i] for i in picked]
    print(f"입력 GT 프레임 {len(sel)}장 (면적 상위 + FPS 다양성) / 후보 {len(stems)}")

    c0 = cams[sel[0]]
    W, H = int(c0["W"]), int(c0["H"])
    fx, fy, cx, cy = c0["fx"], c0["fy"], c0["cx"], c0["cy"]

    frames, train_ids, test_ids = [], [], []
    idx = 0
    for s in sel:
        c = cams[s]
        src = None
        for ext in (".jpg", ".jpeg", ".png", ".JPEG"):
            p = os.path.join(args.images, s + ext)
            if os.path.exists(p):
                src = p; break
        if src is None:
            continue
        name = f"{idx:06d}.png"
        Image.open(src).convert("RGB").save(os.path.join(img_dir, name))
        c2w = np.eye(4); c2w[:3, :3] = c["R"].T; c2w[:3, 3] = cam_center(c["R"], c["t"])
        c2w[:3, :3] = c2w[:3, :3] @ CV2GL                       # → OpenGL
        frames.append(dict(file_path=f"images/{name}", transform_matrix=c2w.tolist()))
        train_ids.append(idx); idx += 1

    # ── 타깃 궤적: unknown 법선 방향에서 객체를 바라보게 ──
    up = o3d.io.read_point_cloud(os.path.expanduser(args.unknown))
    U = np.asarray(up.points); UN = np.asarray(up.normals)
    assert len(U) > 0, "unknown 점군 비어있음"
    ctr = U.mean(0)
    obj_r = float(np.percentile(np.linalg.norm(U - ctr, axis=1), 90))
    nmean = UN.mean(0); nmean /= np.linalg.norm(nmean) + 1e-9
    rad = max(obj_r * args.radius_scale, 0.5)
    print(f"unknown centroid {np.round(ctr,3)}  r90 {obj_r:.2f}  방향 {np.round(nmean,3)}  카메라거리 {rad:.2f}")

    # [v2] 물리 제약: 바닥 위 최소 높이 + 방 bbox 내부 (바닥 관통 카메라 방지)
    cam_pos_all = np.stack([cam_center(cams[s]["R"], cams[s]["t"]) for s in stems])
    up_axis = args.up_axis
    floor = float(np.percentile(cam_pos_all[:, up_axis], 2)) - args.cam_below   # 관측 카메라 최저 부근
    room_lo, room_hi = cam_pos_all.min(0) - 0.5, cam_pos_all.max(0) + 0.5
    print(f"물리 제약: {'xyz'[up_axis]} ≥ {floor:.2f} (바닥), 방 bbox {np.round(room_lo,1)}~{np.round(room_hi,1)}")

    # nmean 주변 원뿔에서 spiral 샘플 (제약 위반 시 up 방향으로 밀어 올림)
    a = nmean
    tmp = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(a, tmp); e1 /= np.linalg.norm(e1) + 1e-9
    e2 = np.cross(a, e1)
    n_fix = 0
    for i in range(args.n_target):
        f = i / max(args.n_target - 1, 1)
        th = np.deg2rad(args.spread_deg) * np.sqrt(f)           # 중심→바깥 나선
        ph = 2 * np.pi * 3.0 * f
        d = (np.cos(th) * a + np.sin(th) * (np.cos(ph) * e1 + np.sin(ph) * e2))
        pos = ctr + rad * d / (np.linalg.norm(d) + 1e-9)
        if pos[up_axis] < floor:                                 # 바닥 아래 → 최소 높이로 승격
            pos[up_axis] = floor
            n_fix += 1
        pos = np.clip(pos, room_lo, room_hi)                     # 방 밖 → 안으로
        c2w = look_at_c2w(pos, ctr)
        c2w[:3, :3] = c2w[:3, :3] @ CV2GL
        name = f"{idx:06d}.png"
        Image.new("RGB", (W, H), (0, 0, 0)).save(os.path.join(img_dir, name))   # 더미
        frames.append(dict(file_path=f"images/{name}", transform_matrix=c2w.tolist()))
        test_ids.append(idx); idx += 1

    with open(os.path.join(root, "transforms.json"), "w") as f:
        json.dump(dict(fl_x=fx, fl_y=fy, cx=cx, cy=cy, w=W, h=H, frames=frames), f, indent=1)
    with open(os.path.join(root, f"train_test_split_{len(train_ids)}.json"), "w") as f:
        json.dump(dict(train_ids=train_ids, test_ids=test_ids), f, indent=1)

    print(f"\nSEVA 씬 생성: {root}")
    print(f"  train {len(train_ids)} (GT) / test {len(test_ids)} (unknown 지향 궤적, 바닥 보정 {n_fix}개)")
    print(f"\n실행 예:\n  cd <seva_repo> && python demo.py --data_path {os.path.dirname(root)} "
          f"--data_items {scene} --task img2trajvid --num_inputs {len(train_ids)} "
          f"--cfg 3.0 --L_short 576 --use_traj_prior True --chunk_strategy interp")


if __name__ == "__main__":
    main()
