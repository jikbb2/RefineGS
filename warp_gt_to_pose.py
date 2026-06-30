#!/usr/bin/env python3
"""RefineGS — GT depth forward-warp to novel pose (See3D 입력, noisy 렌더 대체).

근본 문제: 2DGS 를 unseen pose 에서 렌더하면 floater/streak 노이즈 → See3D 입력으로 부적합.
해결(See3D 원래 방식): *실제 GT 픽셀*을 GT depth 로 target novel pose 에 forward-projection.
  → 관측면 = 진짜 색(노이즈 없음), hole = 진짜 미관측/disocclusion 만(작음).

입력 pose: render_hole_novel --soft_out 의 poses.npz (reachable novel pose, 우리가 잘 추정).
출력: soft_in 포맷(기존 See3D 파이프라인과 호환)
  view_<i>.jpg   GT-warp (실색, hole 은 검정)
  weight_<i>.png hole map (255=hole/미관측, 0=known/실색) ← generate_novel_views see3d 가 mask 로 사용
  poses.npz      복사

forward-warp:
  src 픽셀 (u,v,d) → Xc = d·K^-1[u,v,1] → Xw = R_s^T(Xc - t_s)
  → target: Xc_t = R_t·Xw + t_t,  uv_t = K_t·Xc_t/z   (z-buffer 로 최근접 색 채택)
  k_nearest GT 프레임을 합쳐 채움. 빈 픽셀 = hole.

실행:
  python warp_gt_to_pose.py \
    --poses ~/See3D/dataset/refinegs_obj24/soft_in_carved/poses.npz \
    --gt_images data/replica_room0_v2/images \
    --gt_depth /home/elicer/nice-slam/Datasets/Replica/room0/results \
    --colmap data/replica_room0_v2/sparse/0 \
    --depth_scale 6553.5 --k_nearest 6 \
    --out ~/See3D/dataset/refinegs_obj24/soft_in_gtwarp

Deps: numpy, PIL, (cv2 선택).
"""
import argparse, os, struct, glob, shutil
import numpy as np
from PIL import Image


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
                imgs.append((qvec2rot(q), t, cid, nm.decode()))
    else:
        for ln in open(os.path.join(d,"cameras.txt")):
            if ln.startswith("#") or not ln.strip(): continue
            tt = ln.split(); cid=int(tt[0]); model=tt[1]; w,h=int(tt[2]),int(tt[3]); p=list(map(float,tt[4:]))
            fx,fy,cx,cy = (p[0],p[1],p[2],p[3]) if model=="PINHOLE" else (p[0],p[0],p[1],p[2])
            cams[cid]=(fx,fy,cx,cy,w,h)
        L=[l for l in open(os.path.join(d,"images.txt")) if not l.startswith("#")]
        for i in range(0,len(L),2):
            tt=L[i].split()
            if len(tt)<10: continue
            q=list(map(float,tt[1:5])); t=np.array(list(map(float,tt[5:8])))
            imgs.append((qvec2rot(q),t,int(tt[8]),tt[9]))
    out=[]
    for R,t,cid,nm in imgs:
        fx,fy,cx,cy,w,h = cams[cid]
        out.append(dict(R=R,t=t,fx=fx,fy=fy,cx=cx,cy=cy,W=w,H=h,
                        stem=os.path.splitext(os.path.basename(nm))[0]))
    return out


def load_depth(path, scale, W, H):
    im = Image.open(path)
    d = np.asarray(im).astype(np.float32)
    if d.ndim == 3: d = d[..., 0]
    d = d / scale
    if d.shape != (H, W):
        d = np.asarray(Image.fromarray(d).resize((W, H), Image.NEAREST))
    return d


def target_from_pose(rec):
    wvt = np.asarray(rec["world_view_transform"], float)   # stored = getWorld2View2(R,T).T
    M = wvt.T                                              # W2C 4x4
    R, t = M[:3, :3], M[:3, 3]
    W, H = int(rec["width"]), int(rec["height"])
    fx = W / (2*np.tan(float(rec["FoVx"])/2)); fy = H / (2*np.tan(float(rec["FoVy"])/2))
    return R, t, fx, fy, W/2.0, H/2.0, W, H


def cam_center(R, t):  # -R^T t
    return -R.T @ t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--gt_images", required=True)
    ap.add_argument("--gt_depth", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--depth_scale", type=float, default=6553.5)
    ap.add_argument("--k_nearest", type=int, default=6)
    ap.add_argument("--src_stride", type=int, default=2, help="src 픽셀 stride(속도). 1=full")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cams = read_colmap(a.colmap)
    # GT 이미지/depth 있는 프레임만
    src = []
    for c in cams:
        ip = os.path.join(a.gt_images, c["stem"] + ".jpg")
        if not os.path.exists(ip): ip = os.path.join(a.gt_images, c["stem"] + ".png")
        dp = os.path.join(a.gt_depth, c["stem"].replace("frame", "depth") + ".png")
        if os.path.exists(ip) and os.path.exists(dp):
            c["img_path"], c["depth_path"] = ip, dp
            c["center"] = cam_center(c["R"], c["t"])
            src.append(c)
    if not src:
        raise SystemExit("GT 이미지/depth 매칭 0 — 경로/이름(frame↔depth) 확인")
    print(f"source GT frames: {len(src)}")
    src_centers = np.stack([c["center"] for c in src])

    recs = list(np.load(a.poses, allow_pickle=True)["records"])
    os.makedirs(a.out, exist_ok=True)
    st = a.src_stride

    for rec in recs:
        i = int(rec["idx"])
        Rt, tt, fxt, fyt, cxt, cyt, W, H = target_from_pose(rec)
        tc = cam_center(Rt, tt)
        order = np.argsort(np.linalg.norm(src_centers - tc, axis=1))[:a.k_nearest]

        zbuf = np.full(H*W, np.inf, np.float32)
        color = np.zeros((H*W, 3), np.float32)
        for si in order:
            c = src[si]
            img = np.asarray(Image.open(c["img_path"]).convert("RGB")).astype(np.float32)
            Hs, Ws = img.shape[:2]
            dep = load_depth(c["depth_path"], a.depth_scale, Ws, Hs)
            vs, us = np.mgrid[0:Hs:st, 0:Ws:st]
            us = us.ravel(); vs = vs.ravel(); d = dep[vs, us]
            ok = d > 1e-3
            us, vs, d = us[ok], vs[ok], d[ok]
            # backproject (src cam) → world
            x = (us - c["cx"]) / c["fx"] * d
            y = (vs - c["cy"]) / c["fy"] * d
            Xc = np.stack([x, y, d], 1)
            Xw = (Xc - c["t"]) @ c["R"]                     # R^T(Xc-t) = (Xc-t)@R
            # project → target
            Xct = Xw @ Rt.T + tt
            z = Xct[:, 2]
            front = z > 1e-3
            Xct, z = Xct[front], z[front]
            col = img[vs[front], us[front]]
            ut = (Xct[:, 0]/z*fxt + cxt).astype(np.int64)
            vt = (Xct[:, 1]/z*fyt + cyt).astype(np.int64)
            inb = (ut >= 0)&(ut < W)&(vt >= 0)&(vt < H)
            flat = vt[inb]*W + ut[inb]; zz = z[inb]; cc = col[inb]
            # z-buffer: 최근접만
            srt = np.argsort(-zz)                            # 먼 것 먼저 → 가까운 것이 나중에 덮음
            flat, zz, cc = flat[srt], zz[srt], cc[srt]
            color[flat] = cc
            zbuf[flat] = np.minimum(zbuf[flat], zz)

        filled = zbuf < np.inf
        view = color.reshape(H, W, 3).astype(np.uint8)
        hole = (~filled).reshape(H, W)
        Image.fromarray(view).save(os.path.join(a.out, f"view_{i:04d}.jpg"), quality=95)
        Image.fromarray((hole*255).astype(np.uint8)).save(os.path.join(a.out, f"weight_{i:04d}.png"))
        print(f"[{i:04d}] filled {filled.mean():.3f}  hole {hole.mean():.3f}")

    shutil.copy(a.poses, os.path.join(a.out, "poses.npz"))
    print(f"\n→ {a.out} (view=GT-warp, weight=hole, poses.npz). generate_novel_views see3d 입력으로 사용.")


if __name__ == "__main__":
    main()
