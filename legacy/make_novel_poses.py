#!/usr/bin/env python3
"""RefineGS — 궤적 근처 full-frame novel pose 생성 (GT-warp 입력용).

문제: 객체 중심 orbit pose 는 화면 상당부가 캡처 장면 밖(검은 frustum) → 2DGS 추가학습 supervision 으로 부적합.
해결: 실제 카메라 궤적 *근처*의 작은 baseline perturbation → 프레임이 장면으로 꽉 참 + 작은 disocclusion 만.
      방향(R)은 유지하고 위치만 카메라 right/up 축으로 소량 이동 → full-frame 보장, 미세 신규각.

출력: poses.npz (warp_gt_to_pose / patch_train_novelview 호환)
  records: idx, world_view_transform(4x4), full_proj_transform(4x4), FoVx, FoVy, width, height

실행:
  python make_novel_poses.py --colmap data/replica_room0_v2/sparse/0 \
    --base_stride 12 --offsets 0.15 --n_dir 4 --out ~/See3D/dataset/refinegs_obj24/poses_mild

Deps: numpy. (COLMAP reader 내장)
"""
import argparse, os, struct
import numpy as np


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]], float)


def read_colmap(d):
    cams, imgs = {}, []
    if os.path.isfile(os.path.join(d, "cameras.bin")):
        with open(os.path.join(d, "cameras.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; mp = {0:3,1:4,2:4,3:5}
            for _ in range(n):
                cid, model, w, h = struct.unpack("<iiQQ", f.read(24)); k = mp[model]
                p = struct.unpack(f"<{k}d", f.read(8*k))
                fx, fy, cx, cy = (p[0],p[1],p[2],p[3]) if model==1 else (p[0],p[0],p[1],p[2])
                cams[cid] = (fx,fy,cx,cy,int(w),int(h))
        with open(os.path.join(d, "images.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                struct.unpack("<I", f.read(4)); q = struct.unpack("<4d", f.read(32))
                t = np.array(struct.unpack("<3d", f.read(24))); cid = struct.unpack("<I", f.read(4))[0]
                nm = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00": break
                    nm += c
                n2 = struct.unpack("<Q", f.read(8))[0]; f.read(24*n2)
                imgs.append((qvec2rot(q), t, cid))
    else:
        for ln in open(os.path.join(d,"cameras.txt")):
            if ln.startswith("#") or not ln.strip(): continue
            tt=ln.split(); cid=int(tt[0]); model=tt[1]; w,h=int(tt[2]),int(tt[3]); p=list(map(float,tt[4:]))
            fx,fy,cx,cy=(p[0],p[1],p[2],p[3]) if model=="PINHOLE" else (p[0],p[0],p[1],p[2])
            cams[cid]=(fx,fy,cx,cy,w,h)
        L=[l for l in open(os.path.join(d,"images.txt")) if not l.startswith("#")]
        for i in range(0,len(L),2):
            tt=L[i].split()
            if len(tt)<10: continue
            q=list(map(float,tt[1:5])); t=np.array(list(map(float,tt[5:8])))
            imgs.append((qvec2rot(q),t,int(tt[8])))
    out=[]
    for R,t,cid in imgs:
        fx,fy,cx,cy,w,h=cams[cid]
        out.append(dict(R=R,t=t,fx=fx,fy=fy,W=w,H=h))
    return out


def proj_matrix(znear, zfar, fovx, fovy):
    tx, ty = np.tan(fovx/2), np.tan(fovy/2)
    P = np.zeros((4,4))
    P[0,0]=1/tx; P[1,1]=1/ty
    P[2,2]=zfar/(zfar-znear); P[2,3]=-(zfar*znear)/(zfar-znear); P[3,2]=1.0
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--base_stride", type=int, default=12, help="base 로 쓸 학습카메라 간격")
    ap.add_argument("--offsets", type=float, default=0.15, help="baseline 이동량(m)")
    ap.add_argument("--n_dir", type=int, default=4, choices=[2,4,8], help="이동 방향 수(right/up 조합)")
    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cams = read_colmap(a.colmap)
    bases = cams[::a.base_stride]
    dirs2 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,1)][:a.n_dir]

    recs = []
    idx = 0
    for c in bases:
        Rw2c, t = c["R"], c["t"]
        C = -Rw2c.T @ t                          # cam center (world)
        right = Rw2c.T[:, 0]; up = Rw2c.T[:, 1]   # 카메라 축(월드)
        fovx = 2*np.arctan(c["W"]/(2*c["fx"])); fovy = 2*np.arctan(c["H"]/(2*c["fy"]))
        proj = proj_matrix(a.znear, a.zfar, fovx, fovy).T
        for (sx, sy) in dirs2:
            Cn = C + a.offsets*(sx*right + sy*up)
            tn = -Rw2c @ Cn                        # R 유지, 위치만 이동
            M = np.eye(4); M[:3,:3] = Rw2c; M[:3,3] = tn   # W2C
            wvt = M.T                              # stored(transpose)
            full = wvt @ proj
            recs.append(dict(idx=idx, world_view_transform=wvt.astype(np.float32),
                             full_proj_transform=full.astype(np.float32),
                             FoVx=float(fovx), FoVy=float(fovy),
                             width=int(c["W"]), height=int(c["H"])))
            idx += 1

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "poses.npz"), records=np.array(recs, dtype=object))
    print(f"bases {len(bases)} × dirs {len(dirs2)} = {len(recs)} novel poses (offset {a.offsets}m)")
    print(f"→ {a.out}/poses.npz  — warp_gt_to_pose --poses 로 사용. full-frame 보장(작은 disocclusion만).")


if __name__ == "__main__":
    main()
