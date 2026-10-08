#!/usr/bin/env python3
"""
Rebuild COLMAP images.txt from a trajectory (c2w 4x4, one row of 16 values per frame), so
that every frame has a pose.

Why: data/<scene>/sparse/0 holds only the stride-10 subset (200 poses), which starves a
dense-stride relabel or reconstruction of poses and makes it meaningless. This writes poses
for all 2000 frames, in the same world frame, from the GT trajectory.

The safeguard that matters: the 200 poses already in images.txt are compared against the
converted trajectory FIRST, to verify that the convention agrees (c2w direction, world
frame, row order). On a mismatch nothing is written and the script exits with an error --
that case needs colmap image_registrator instead.

Usage:
    python make_dense_colmap.py \
        --traj ~/room_0/imap/00/traj_w_c.txt \
        --frames data/<scene>/images --img_ext .jpg \
        --colmap_in data/<scene>/sparse/0 \
        --out data/<scene>/sparse_dense/0

Once it passes, use SCENE_COLMAP=data/<scene>/sparse_dense/0 in the pipeline (or back up
sparse/0 and replace it).
"""
import argparse, glob, os, re, shutil
import numpy as np


def rot2quat(R):
    """3x3 rotation matrix -> (w, x, y, z) quaternion, in COLMAP's convention."""
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
    """images.txt -> {name: (qvec, tvec, camera_id)}."""
    out = {}
    L = [l for l in open(path) if not l.startswith("#") and l.strip()]
    # images.txt holds two lines per image; the second is points2D and may be empty. Parse
    # only the pose lines.
    for ln in L:
        t = ln.split()
        if len(t) >= 10 and t[9].lower().endswith((".jpg",".jpeg",".png")):
            q = np.array(list(map(float, t[1:5])))
            tv = np.array(list(map(float, t[5:8])))
            out[t[9]] = (q, tv, int(t[8]))
    return out


def load_traj(path):
    """N rows of 16 values (c2w, row-major), or 4N rows of 4 -> (N, 4, 4)."""
    A = np.loadtxt(path)
    if A.ndim == 2 and A.shape[1] == 16:
        return A.reshape(-1, 4, 4)
    if A.ndim == 2 and A.shape[1] == 4 and A.shape[0] % 4 == 0:
        return A.reshape(-1, 4, 4)
    raise ValueError(f"unrecognised traj layout: shape={A.shape} (expected N x 16 or 4N x 4)")


def c2w_to_colmap(c2w):
    """c2w -> COLMAP w2c (qvec, tvec)."""
    R_wc = c2w[:3,:3].T                 # w2c rotation
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
    ap.add_argument("--colmap_in", required=True,
                    help="the existing sparse/0: the reference for the check, and the source "
                         "of cameras/points3D")
    ap.add_argument("--out", required=True, help="the new sparse directory, e.g. .../sparse_dense/0")
    ap.add_argument("--rot_tol_deg", type=float, default=0.5)
    ap.add_argument("--trans_tol", type=float, default=0.01, help="meters")
    ap.add_argument("--force", action="store_true", help="write even if the check fails (not advised)")
    args = ap.parse_args()

    traj = load_traj(args.traj)
    frames = sorted(glob.glob(os.path.join(args.frames, f"*{args.img_ext}")))
    names = [os.path.basename(f) for f in frames]
    print(f"traj poses={len(traj)}  frames={len(names)}")

    def frame_idx(name):
        m = re.search(r"(\d+)", os.path.splitext(name)[0])
        return int(m.group(1)) if m else None

    idxs = [frame_idx(n) for n in names]
    assert all(i is not None for i in idxs), "could not read an integer index out of a frame name"
    assert max(idxs) < len(traj), \
        f"frame idx {max(idxs)} >= traj {len(traj)} -- the trajectory does not cover every frame"

    # -- the check: the poses already in colmap, against the converted trajectory --
    exist = read_existing_images_txt(os.path.join(args.colmap_in, "images.txt"))
    print(f"existing colmap poses={len(exist)} -- comparing against the converted trajectory...")
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
    print(f"compared {ncmp}: max rotation diff={max_rot:.4f} deg  max translation diff={max_tr:.5f} m")
    ok = (max_rot < args.rot_tol_deg) and (max_tr < args.trans_tol) and ncmp > 0
    if not ok and not args.force:
        # Usually one of two things: the trajectory is w2c, or its world frame differs.
        raise SystemExit(
            "[abort] check failed -- the trajectory convention disagrees with the existing colmap.\n"
            "  1) the trajectory may already be w2c: use it directly instead of c2w_to_colmap\n"
            "     and check again\n"
            "  2) if the world frame itself differs, this trajectory is unusable -- run colmap\n"
            "     image_registrator instead: colmap feature_extractor / vocab_tree_matcher, then\n"
            "     colmap image_registrator --database_path DB --input_path sparse/0 --output_path sparse_dense/0")
    if ok:
        print("[ok] convention and world frame agree -- writing the dense images.txt")

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
    print(f"[ok] {args.out}/images.txt -- {len(names)} poses (camera_id={cam_id or 1})")
    print(f"next: run relabel / reconstruction with SCENE_COLMAP={args.out}")


if __name__ == "__main__":
    main()