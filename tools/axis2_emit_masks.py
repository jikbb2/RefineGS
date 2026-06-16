#!/usr/bin/env python3
"""
축2 Step 4 — 선택된 granularity(selected.npz)를 각 뷰로 역투영 → per-view 마스크 PNG.

각 part(또는 whole)의 GS 포인트를 모든 뷰에 투영 + software z-buffer(가시 포인트만) +
반경 페인팅으로 채운 마스크 생성. per-object recon(prepare_folder/train)이 먹는 형식.

출력: <out_dir>/part{k}/{stem}.png  (whole 결정이면 part0 하나 = 객체 마스크)

규약: COLMAP 투영은 axis2_vote/eval_object_mesh와 동일. 의존: numpy, PIL (SAM3 불필요).

실행 (어느 env든):
    python axis2_emit_masks.py \
        --in_dir ~/axis2_vote_98 \
        --colmap_dir /home/elicer/RefineGS/data/replica_room0/masks/98/sparse/0 \
        --images_dir /home/elicer/RefineGS/data/replica_room0/masks/98/images \
        --radius 4 --out_dir ~/axis2_masks_98
"""
import argparse
import glob
import os
import struct
import numpy as np
from PIL import Image


def _qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def _read_cameras_txt(p):
    cams = {}
    for ln in open(p):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split(); cid = int(t[0]); model = t[1]
        w, h = int(t[2]), int(t[3]); pr = list(map(float, t[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = pr[:4]
        else:
            fx = fy = pr[0]; cx, cy = pr[1], pr[2]
        cams[cid] = (fx, fy, cx, cy, w, h)
    return cams


def _read_images_txt(p):
    out = []; lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        q = list(map(float, t[1:5])); tv = np.array(list(map(float, t[5:8])))
        out.append({"R": _qvec2rot(q), "t": tv, "camera_id": int(t[8]), "name": t[9]})
    return out


def _read_bin(d):
    cams = {}
    with open(os.path.join(d, "cameras.bin"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; mp = {0: 3, 1: 4, 2: 4, 3: 5}
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            k = mp[model]; pr = struct.unpack(f"<{k}d", f.read(8*k))
            if model == 1:
                fx, fy, cx, cy = pr[:4]
            else:
                fx = fy = pr[0]; cx, cy = pr[1], pr[2]
            cams[cid] = (fx, fy, cx, cy, int(w), int(h))
    imgs = []
    with open(os.path.join(d, "images.bin"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I", f.read(4))
            q = struct.unpack("<4d", f.read(32)); tv = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<I", f.read(4))[0]; name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2 = struct.unpack("<Q", f.read(8))[0]; f.read(24*n2)
            imgs.append({"R": _qvec2rot(q), "t": tv, "camera_id": cid, "name": name.decode()})
    return cams, imgs


def load_cameras(d):
    if os.path.isfile(os.path.join(d, "images.bin")):
        cams, imgs = _read_bin(d)
    else:
        cams = _read_cameras_txt(os.path.join(d, "cameras.txt"))
        imgs = _read_images_txt(os.path.join(d, "images.txt"))
    out = {}
    for im in imgs:
        fx, fy, cx, cy, w, h = cams[im["camera_id"]]
        stem = os.path.splitext(os.path.basename(im["name"]))[0]
        out[stem] = dict(R=im["R"], t=im["t"], fx=fx, fy=fy, cx=cx, cy=cy, W=w, H=h)
    return out


def visible_pixels(xyz, cam):
    Xc = xyz @ cam["R"].T + cam["t"]
    z = Xc[:, 2]; ok = z > 1e-6
    u = cam["fx"] * Xc[:, 0] / np.where(ok, z, 1) + cam["cx"]
    v = cam["fy"] * Xc[:, 1] / np.where(ok, z, 1) + cam["cy"]
    W, H = cam["W"], cam["H"]
    ui = np.round(u).astype(np.int64); vi = np.round(v).astype(np.int64)
    inb = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = np.where(inb)[0]
    if len(idx) == 0:
        return idx, ui, vi
    pid = vi[idx] * W + ui[idx]; order = np.argsort(z[idx])
    _, first = np.unique(pid[order], return_index=True)
    return idx[order[first]], ui, vi


def paint(mask, us, vs, r):
    H, W = mask.shape
    for u, v in zip(us, vs):
        mask[max(0, v-r):min(H, v+r+1), max(0, u-r):min(W, u+r+1)] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="selected.npz 있는 폴더")
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--radius", type=int, default=4, help="포인트 페인팅 반경(px)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    d = np.load(os.path.join(args.in_dir, "selected.npz"))
    xyz = d["xyz"]; sel = d["selected"]               # K x N bool
    K = len(sel)
    print(f"granularity parts: {K}, GS={xyz.shape[0]}")
    cams = load_cameras(args.colmap_dir)
    stems = {os.path.splitext(os.path.basename(f))[0]: f
             for f in glob.glob(os.path.join(args.images_dir, "*"))}
    print(f"cameras={len(cams)} images={len(stems)}")

    for k in range(K):
        os.makedirs(os.path.join(args.out_dir, f"part{k}"), exist_ok=True)
    counts = np.zeros(K, int)
    for stem, cam in cams.items():
        if stem not in stems:
            continue
        vis, ui, vi = visible_pixels(xyz, cam)
        if len(vis) == 0:
            continue
        visset = np.zeros(xyz.shape[0], bool); visset[vis] = True
        for k in range(K):
            pm = sel[k] & visset
            m = np.zeros((cam["H"], cam["W"]), bool)
            if pm.any():
                pts = np.where(pm)[0]
                paint(m, ui[pts], vi[pts], args.radius)
            if m.any():
                counts[k] += 1
            Image.fromarray((m * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"part{k}", f"{stem}.png"))
    print("\npart별 비어있지 않은 마스크 수:")
    for k in range(K):
        print(f"  part{k}: {counts[k]} views")
    print(f"\n저장: {args.out_dir}/part*/<stem>.png")
    print("다음(task9): 이 마스크로 per-object recon(prepare_folder→train) → eval_object_mesh 평가.")


if __name__ == "__main__":
    main()
