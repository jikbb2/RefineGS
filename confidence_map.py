#!/usr/bin/env python3
"""RefineGS B(See3D) — per-Gaussian 신뢰도 맵.

각 Gaussian을 good/bad 로 라벨:
  bad = 미관측(관측 마스크 뷰에서 안 보임) OR 저opacity OR 과대 scale(floater).
이후 scene warp 렌더러가 *good Gaussian만 렌더* → 자연히 미관측/floater 영역이 hole → See3D가 채움.

신호:
  - 가시성: Gaussian 중심을 관측 카메라로 투영 → 마스크 안 + front-facing(nx,ny,nz·시선>0) + z>0 가
    >= min_views 프레임이면 observed.
  - opacity: sigmoid(opacity field) > op_thr.
  - scale:   max(exp(scale_0,scale_1)) < scale_thr (floater 제외).

출력: <out_prefix>_conf.npy (입력 ply 정점 순서, 1=good/0=bad)  +  <out_prefix>_qa.ply (green=good,red=bad).

실행:
  python confidence_map.py \
    --gaussians output/replica_room0_v2/refinegs_fix/24/point_cloud/iteration_10000/point_cloud.ply \
    --colmap_dir data/replica_room0_v2/masks/24/sparse/0 \
    --masks_dir ~/relabel_replica_room0_v2/24 \
    --min_views 2 --op_thr 0.3 --scale_thr 0.1 --out_prefix /tmp/obj24

Deps: numpy, plyfile, PIL.
"""
import argparse, os, struct
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def read_cameras(sparse_dir):
    cams = {}; images = []
    if os.path.isfile(os.path.join(sparse_dir, "cameras.bin")):
        with open(os.path.join(sparse_dir, "cameras.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; mp = {0: 3, 1: 4, 2: 4, 3: 5}
            for _ in range(n):
                cid, model, w, h = struct.unpack("<iiQQ", f.read(24)); k = mp[model]
                p = struct.unpack(f"<{k}d", f.read(8*k))
                fx, fy, cx, cy = (p[0], p[1], p[2], p[3]) if model == 1 else (p[0], p[0], p[1], p[2])
                cams[cid] = (fx, fy, cx, cy, int(w), int(h))
        with open(os.path.join(sparse_dir, "images.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                struct.unpack("<I", f.read(4)); q = struct.unpack("<4d", f.read(32))
                t = np.array(struct.unpack("<3d", f.read(24))); cid = struct.unpack("<I", f.read(4))[0]
                name = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00": break
                    name += c
                n2 = struct.unpack("<Q", f.read(8))[0]; f.read(24*n2)
                images.append((qvec2rot(q), t, cid, name.decode()))
    else:
        for ln in open(os.path.join(sparse_dir, "cameras.txt")):
            if ln.startswith("#") or not ln.strip(): continue
            tt = ln.split(); cid = int(tt[0]); model = tt[1]; w, h = int(tt[2]), int(tt[3]); p = list(map(float, tt[4:]))
            fx, fy, cx, cy = (p[0], p[1], p[2], p[3]) if model == "PINHOLE" else (p[0], p[0], p[1], p[2])
            cams[cid] = (fx, fy, cx, cy, w, h)
        L = [l for l in open(os.path.join(sparse_dir, "images.txt")) if not l.startswith("#")]
        for i in range(0, len(L), 2):
            tt = L[i].split()
            if len(tt) < 10: continue
            q = list(map(float, tt[1:5])); t = np.array(list(map(float, tt[5:8])))
            images.append((qvec2rot(q), t, int(tt[8]), tt[9]))
    out = []
    for R, t, cid, name in images:
        fx, fy, cx, cy, w, h = cams[cid]
        out.append(dict(R=R, t=t, fx=fx, fy=fy, cx=cx, cy=cy, W=w, H=h,
                        stem=os.path.splitext(os.path.basename(name))[0]))
    return out


def load_masks(masks_dir, cams):
    files = {os.path.splitext(f)[0]: os.path.join(masks_dir, f)
             for f in os.listdir(masks_dir) if f.lower().endswith((".png", ".jpg"))}
    out = {}
    for c in cams:
        p = files.get(c["stem"])
        out[c["stem"]] = (np.asarray(Image.open(p).convert("L")) > 127) if p else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--min_views", type=int, default=2)
    ap.add_argument("--op_thr", type=float, default=0.3)
    ap.add_argument("--scale_thr", type=float, default=0.1, help="exp(scale) 이상이면 floater")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out_prefix", required=True)
    a = ap.parse_args()

    v = PlyData.read(a.gaussians)["vertex"]; nm = v.data.dtype.names
    P = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    N = (np.column_stack([v["nx"], v["ny"], v["nz"]]).astype(np.float64)
         if "nx" in nm else np.zeros_like(P))
    op = sigmoid(np.asarray(v["opacity"]).astype(np.float64))
    sc = np.exp(np.column_stack([v["scale_0"], v["scale_1"]]).astype(np.float64)).max(1)

    cams = read_cameras(a.colmap_dir); masks = load_masks(a.masks_dir, cams)
    cnt = np.zeros(len(P), np.int32)
    for cam in cams[::a.stride]:
        m = masks.get(cam["stem"])
        if m is None: continue
        Xc = P @ cam["R"].T + cam["t"]; z = Xc[:, 2]; ok = z > 1e-6
        u = np.where(ok, cam["fx"]*Xc[:, 0]/np.where(ok, z, 1)+cam["cx"], -1)
        vv = np.where(ok, cam["fy"]*Xc[:, 1]/np.where(ok, z, 1)+cam["cy"], -1)
        Hm, Wm = m.shape; sx = Wm/cam["W"]; sy = Hm/cam["H"]
        ui = (u*sx).astype(np.int64); vi = (vv*sy).astype(np.int64)
        ok &= (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm)
        inm = np.zeros(len(P), bool); inm[ok] = m[vi[ok], ui[ok]]
        if N.any():
            C = -cam["R"].T @ cam["t"]
            front = np.einsum("ij,ij->i", N, C[None] - P) > 0
            inm &= front
        cnt += inm
    observed = cnt >= a.min_views
    good = observed & (op > a.op_thr) & (sc < a.scale_thr)

    np.save(a.out_prefix + "_conf.npy", good.astype(np.float32))
    # QA colored ply (green=good, red=bad)
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(len(P), dt)
    arr["x"], arr["y"], arr["z"] = P[:, 0], P[:, 1], P[:, 2]
    arr["red"] = np.where(good, 0, 255); arr["green"] = np.where(good, 255, 0); arr["blue"] = 0
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(a.out_prefix + "_qa.ply")

    print(f"gaussians {len(P)}")
    print(f"  observed(>= {a.min_views}뷰): {observed.mean():.3f}")
    print(f"  opacity>{a.op_thr}: {(op>a.op_thr).mean():.3f}   scale<{a.scale_thr}: {(sc<a.scale_thr).mean():.3f}")
    print(f"  GOOD(전부 충족): {good.mean():.3f}   BAD: {(~good).mean():.3f}")
    print(f"→ {a.out_prefix}_conf.npy (1=good/0=bad),  {a.out_prefix}_qa.ply (green=good/red=bad)")


if __name__ == "__main__":
    main()
