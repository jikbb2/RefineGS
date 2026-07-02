#!/usr/bin/env python3
"""
traj(c2w 4x4, 프레임당 1행 16값) → COLMAP images.txt 재생성 (전체 프레임 dense pose).

배경: data/replica_room0_v2/sparse/0 은 stride-10 서브셋(200 pose)만 담고 있어
dense-stride relabel/recon 이 pose 기아로 무의미했음. GT trajectory 로 2000 프레임
전체의 pose 를 같은 world frame 으로 생성한다.

핵심 안전장치: 기존 images.txt 의 200개 pose 와 traj 변환 결과를 **먼저 대조**하여
convention(c2w 방향, world frame, 행 순서)이 일치하는지 검증한다. 불일치하면 쓰지 않고
에러로 종료 → 그 경우 colmap image_registrator 경로로 가야 함.

사용:
    python make_dense_colmap.py \
        --traj ~/room_0/imap/00/traj_w_c.txt \
        --frames data/replica_room0_v2/images --img_ext .jpg \
        --colmap_in data/replica_room0_v2/sparse/0 \
        --out data/replica_room0_v2/sparse_dense/0

검증 통과 후 파이프라인에서 SCENE_COLMAP=data/replica_room0_v2/sparse_dense/0 로 사용
(또는 기존 sparse/0 을 백업하고 교체).
"""
import argparse, glob, os, re, shutil
import numpy as np


def rot2quat(R):
    """3x3 회전행렬 → (w,x,y,z) 쿼터니언 (COLMAP 규약)."""
    K = np.array([
        [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
        [R[0,1]+R[1,0], R[1,1]-R[0,0]-R[2,2], 0, 0],
        [R[0,2]+R[2,0], R[1,2]+R[2,1], R[2,2]-R[0,0]-R[1,1], 0],
        [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]],
    ]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3,0,1,2], np.argmax(w)]      # (w,x,y,z)
    return q if q[0] >= 0 else -q


def read_existing_images_txt(path):
    """images.txt → {name: (qvec, tvec, camera_id)}; 최대 image_id 도 반환."""
    out = {}
    L = [l for l in open(path) if not l.startswith("#") and l.strip()]
    # images.txt 는 2줄/이미지 (2번째 줄은 points2D — 비었을 수 있음). pose 줄만 파싱.
    for ln in L:
        t = ln.split()
        if len(t) >= 10 and t[9].lower().endswith((".jpg",".jpeg",".png")):
            q = np.array(list(map(float, t[1:5])))
            tv = np.array(list(map(float, t[5:8])))
            out[t[9]] = (q, tv, int(t[8]))
    return out


def load_traj(path):
    """N행 × 16값(c2w row-major) 또는 4N행 × 4값 → (N,4,4)."""
    A = np.loadtxt(path)
    if A.ndim == 2 and A.shape[1] == 16:
        return A.reshape(-1, 4, 4)
    if A.ndim == 2 and A.shape[1] == 4 and A.shape[0] % 4 == 0:
        return A.reshape(-1, 4, 4)
    raise ValueError(f"traj 형식 인식 실패: shape={A.shape} (기대: N×16 또는 4N×4)")


def c2w_to_colmap(c2w):
    """c2w → COLMAP w2c (qvec, tvec)."""
    R_wc = c2w[:3,:3].T                 # w2c 회전
    t_wc = -R_wc @ c2w[:3,3]
    return rot2quat(R_wc), t_wc


def quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return np.degrees(2*np.arccos(min(d, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--img_ext", default=".jpg")
    ap.add_argument("--colmap_in", required=True, help="기존 sparse/0 (검증 기준 + cameras/points3D 복사원)")
    ap.add_argument("--out", required=True, help="새 sparse 디렉토리 (예: .../sparse_dense/0)")
    ap.add_argument("--rot_tol_deg", type=float, default=0.5)
    ap.add_argument("--trans_tol", type=float, default=0.01, help="meters")
    ap.add_argument("--force", action="store_true", help="검증 실패해도 쓰기(비추천)")
    args = ap.parse_args()

    traj = load_traj(args.traj)
    frames = sorted(glob.glob(os.path.join(args.frames, f"*{args.img_ext}")))
    names = [os.path.basename(f) for f in frames]
    print(f"traj poses={len(traj)}  frames={len(names)}")

    def frame_idx(name):
        m = re.search(r"(\d+)", os.path.splitext(name)[0])
        return int(m.group(1)) if m else None

    idxs = [frame_idx(n) for n in names]
    assert all(i is not None for i in idxs), "프레임명에서 정수 인덱스 추출 실패"
    assert max(idxs) < len(traj), f"frame idx {max(idxs)} ≥ traj {len(traj)} — traj가 프레임 전체를 커버하지 않음"

    # ── 검증: 기존 200 pose vs traj 변환 ──
    exist = read_existing_images_txt(os.path.join(args.colmap_in, "images.txt"))
    print(f"기존 colmap poses={len(exist)} — traj 변환과 대조...")
    max_rot, max_tr, ncmp = 0.0, 0.0, 0
    cam_id = None
    for name, (q0, t0, cid) in exist.items():
        i = frame_idx(name)
        if i is None or i >= len(traj):
            continue
        q1, t1 = c2w_to_colmap(traj[i])
        max_rot = max(max_rot, quat_angle_deg(q0, q1))
        max_tr = max(max_tr, float(np.linalg.norm(t0 - t1)))
        cam_id = cid
        ncmp += 1
    print(f"대조 {ncmp}개: max 회전차={max_rot:.4f}°  max 병진차={max_tr:.5f}m")
    ok = (max_rot < args.rot_tol_deg) and (max_tr < args.trans_tol) and ncmp > 0
    if not ok and not args.force:
        # 흔한 원인: traj가 w2c 이거나 world frame 상이. w2c 가정으로 재시도 안내.
        raise SystemExit(
            "★검증 실패 — traj convention 이 기존 colmap 과 불일치.\n"
            "  1) traj 가 w2c 일 수 있음: c2w_to_colmap 대신 직접 사용해 재검증 필요\n"
            "  2) world frame 자체가 다르면 이 traj 는 못 씀 → colmap image_registrator 사용:\n"
            "     colmap feature_extractor / vocab_tree_matcher 후\n"
            "     colmap image_registrator --database_path DB --input_path sparse/0 --output_path sparse_dense/0")
    if ok:
        print("✓ convention/world-frame 일치 — dense images.txt 생성")

    os.makedirs(args.out, exist_ok=True)
    for f in ("cameras.txt", "points3D.txt", "points3D.ply"):
        s = os.path.join(args.colmap_in, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out, f))
    with open(os.path.join(args.out, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for k, (name, i) in enumerate(zip(names, idxs), start=1):
            q, t = c2w_to_colmap(traj[i])
            f.write(f"{k} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cam_id or 1} {name}\n\n")
    print(f"✓ {args.out}/images.txt — {len(names)} poses (camera_id={cam_id or 1})")
    print(f"다음: SCENE_COLMAP={args.out} 로 relabel/recon 실행")


if __name__ == "__main__":
    main()
