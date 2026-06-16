#!/usr/bin/env python3
"""
축2 Step 2 — multi-view 3D part voting (training-free; GARField의 학습 field 대체).

각 뷰: SAM3 concept로 객체 마스크 → point grid multimask로 part 후보(Step 1) →
객체 GS 포인트를 투영+software z-buffer로 *가시* 포인트만 골라 각 part를
GS 포인트 인덱스 집합으로 변환. 모든 뷰의 part가 동일 3D 포인트 공간의 부분집합이
되므로 Jaccard로 뷰 간 part를 클러스터(=multi-view 일관 part) + view-support 집계.

규약: stage3(autocast bf16), predict_inst(multimask_output=True), COLMAP 투영은
eval_object_mesh와 동일.

의존: numpy(<2, sam3 호환), torch, PIL, sam3. (PLY는 인라인 리더 — plyfile 불필요)

실행 (sam3 env):
    conda activate sam3
    LD_LIBRARY_PATH= python axis2_vote.py \
        --gs_ply output/replica_room0/raw_graph_reg/98/point_cloud/iteration_7000/point_cloud.ply \
        --images_dir data/replica_room0/masks/98/images \
        --colmap_dir data/replica_room0/masks/98/sparse/0 \
        --concept table --stride 20 --grid 6 \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --out_dir ~/axis2_vote_98
  (경로는 객체에 맞게 조정. raw_graph_reg/98 없으면 axis3_sweep/reg_strong/98/.../point_cloud.ply 사용)
"""
import argparse
import glob
import json
import os
import struct
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ── COLMAP (eval_object_mesh와 동일 규약) ────────────────────────────────────
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
        n = struct.unpack("<Q", f.read(8))[0]
        mp = {0: 3, 1: 4, 2: 4, 3: 5}
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
            cid = struct.unpack("<I", f.read(4))[0]
            name = b""
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


def load_gs_xyz(path):
    """외부 의존 없이 binary/ascii PLY의 vertex x,y,z만 읽음 (plyfile 불필요).
    GS point_cloud.ply / mesh fuse_post.ply 모두 vertex에 x,y,z를 가짐."""
    TYPE = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
            "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        fmt = f.readline().split()[1].decode()        # ascii / binary_little_endian
        props, n_vert, in_v = [], 0, False
        while True:
            ln = f.readline()
            if ln.strip() == b"end_header":
                break
            p = ln.split()
            if p[0] == b"element":
                in_v = (p[1] == b"vertex")
                if in_v:
                    n_vert = int(p[2])
            elif p[0] == b"property" and in_v:
                props.append((p[2].decode(), p[1].decode()))   # (name, type)
        names = [n for n, _ in props]
        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        if fmt.startswith("binary_little_endian"):
            dt = np.dtype([(n, TYPE[t]) for n, t in props])
            data = np.frombuffer(f.read(n_vert * dt.itemsize), dtype=dt, count=n_vert)
            return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        else:  # ascii
            arr = np.array([f.readline().split()[:len(props)] for _ in range(n_vert)],
                           dtype=np.float64)
            return arr[:, [ix, iy, iz]]


# ── SAM3 helpers ─────────────────────────────────────────────────────────────
def to_bool(m):
    if hasattr(m, "detach"):
        m = m.detach().float().cpu().numpy()
    m = np.squeeze(np.asarray(m))
    if m.ndim == 2:
        m = m[None]
    return [x > 0.5 if x.dtype != bool else x for x in m]


def concept_mask(proc, state, concept):
    out = proc.set_text_prompt(state=state, prompt=concept)
    m = out.get("masks") if isinstance(out, dict) else None
    if m is None:
        return None
    bm = to_bool(m)
    return bm[int(np.argmax([x.sum() for x in bm]))] if bm else None


def grid_in_mask(mask, grid):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), int)
    pts = []
    for gy in np.linspace(ys.min(), ys.max(), grid):
        for gx in np.linspace(xs.min(), xs.max(), grid):
            xi, yi = int(round(gx)), int(round(gy))
            if mask[yi, xi]:
                pts.append((xi, yi))
    return np.array(pts, int)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def cov(small, big):
    s = small.sum()
    return np.logical_and(small, big).sum() / s if s else 0.0


def view_parts(model, proc, state, obj, grid, keep_cov, min_part, max_part):
    """Step 1: 객체 마스크 안 part 후보(2D bool masks) 리스트."""
    oa = obj.sum(); cands = []
    for pt in grid_in_mask(obj, grid):
        masks, scores, _ = model.predict_inst(
            state, point_coords=np.array([pt]), point_labels=np.array([1]),
            multimask_output=True)
        for mm in to_bool(masks):
            a = mm.sum()
            if a == 0:
                continue
            if cov(mm, obj) >= keep_cov and min_part <= a/oa <= max_part:
                cands.append(mm)
    # NMS
    order = sorted(range(len(cands)), key=lambda i: -cands[i].sum())
    kept = []
    for i in order:
        if all(iou(cands[i], cands[k]) < 0.7 for k in kept):
            kept.append(i)
    return [cands[i] for i in kept]


def visible_pixel_of_points(xyz, cam):
    """투영 + software z-buffer. returns (vis_idx, u_int, v_int) for visible points."""
    Xc = xyz @ cam["R"].T + cam["t"]
    z = Xc[:, 2]
    ok = z > 1e-6
    u = cam["fx"] * Xc[:, 0] / np.where(ok, z, 1) + cam["cx"]
    v = cam["fy"] * Xc[:, 1] / np.where(ok, z, 1) + cam["cy"]
    W, H = cam["W"], cam["H"]
    ui = np.round(u).astype(np.int64); vi = np.round(v).astype(np.int64)
    inb = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = np.where(inb)[0]
    if len(idx) == 0:
        return idx, ui, vi
    pid = vi[idx] * W + ui[idx]
    zz = z[idx]
    order = np.argsort(zz)              # near→far
    pid_s = pid[order]
    _, first = np.unique(pid_s, return_index=True)   # nearest per pixel
    vis_idx = idx[order[first]]
    return vis_idx, ui, vi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_ply", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max_views", type=int, default=20)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--keep_cov", type=float, default=0.8)
    ap.add_argument("--min_part", type=float, default=0.05)
    ap.add_argument("--max_part", type=float, default=0.7)
    ap.add_argument("--jac_th", type=float, default=0.3, help="3D Jaccard 클러스터 임계")
    ap.add_argument("--min_views", type=int, default=2, help="일관 part로 인정할 최소 뷰 수")
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--out_dir", default=os.path.expanduser("~/axis2_vote_out"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    xyz = load_gs_xyz(args.gs_ply)
    N = len(xyz)
    print(f"GS points: {N}")
    cams = load_cameras(args.colmap_dir)
    print(f"cameras: {len(cams)}")

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*")))[::args.stride][:args.max_views]
    print(f"views to process: {len(imgs)}")

    mk = dict(enable_inst_interactivity=True)
    if args.bpe:
        mk["bpe_path"] = args.bpe
    model = build_sam3_image_model(**mk)
    proc = Sam3Processor(model)

    observations = []   # list of boolean index arrays (len N) over GS points
    obs_view = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ip in imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            cam = cams.get(stem)
            if cam is None:
                continue
            image = Image.open(ip).convert("RGB")
            state = proc.set_image(image)
            obj = concept_mask(proc, state, args.concept)
            if obj is None or obj.sum() == 0:
                continue
            proc.reset_all_prompts(state); state = proc.set_image(image)
            parts = view_parts(model, proc, state, obj, args.grid,
                               args.keep_cov, args.min_part, args.max_part)
            if not parts:
                continue
            vis_idx, ui, vi = visible_pixel_of_points(xyz, cam)
            if len(vis_idx) == 0:
                continue
            Hm, Wm = parts[0].shape
            for pm in parts:
                if pm.shape != (cam["H"], cam["W"]):
                    pm = np.array(Image.fromarray((pm*255).astype(np.uint8))
                                  .resize((cam["W"], cam["H"]), Image.NEAREST)) > 127
                sel = pm[vi[vis_idx], ui[vis_idx]]   # 가시 포인트 중 part 안
                member = np.zeros(N, bool)
                member[vis_idx[sel]] = True
                if member.sum() >= max(20, 0.01*N*0.1):  # 너무 작은 건 버림
                    observations.append(member)
                    obs_view.append(stem)
            print(f"  {stem}: obj+{len(parts)}parts, vis={len(vis_idx)}")

    print(f"\n총 part 관측: {len(observations)} (across views)")
    # ── 3D Jaccard 클러스터링 (greedy) ──
    def jac(a, b):
        u = np.logical_or(a, b).sum()
        return np.logical_and(a, b).sum() / u if u else 0.0

    clusters = []  # each: dict(rep=boolidx, members=[idx], views=set)
    order = sorted(range(len(observations)), key=lambda i: -observations[i].sum())
    for i in order:
        placed = False
        for c in clusters:
            if jac(observations[i], c["rep"]) > args.jac_th:
                c["members"].append(i); c["views"].add(obs_view[i])
                c["rep"] = np.logical_or(c["rep"], observations[i])  # union 갱신
                placed = True
                break
        if not placed:
            clusters.append(dict(rep=observations[i].copy(),
                                 members=[i], views={obs_view[i]}))

    consistent = [c for c in clusters if len(c["views"]) >= args.min_views]
    consistent.sort(key=lambda c: -c["rep"].sum())
    print(f"클러스터: {len(clusters)}, 일관 part(views>={args.min_views}): {len(consistent)}\n")
    for k, c in enumerate(consistent):
        print(f"  part{k}: GS {c['rep'].sum()}/{N} ({c['rep'].sum()/N*100:4.1f}%) "
              f"views={len(c['views'])}")

    # 저장: 일관 part의 GS 인덱스 + 메타
    np.savez_compressed(os.path.join(args.out_dir, "parts3d.npz"),
                        xyz=xyz,
                        parts=np.stack([c["rep"] for c in consistent]) if consistent
                        else np.empty((0, N), bool))
    json.dump([{"id": k, "n_gs": int(c["rep"].sum()),
                "frac": round(c["rep"].sum()/N, 3), "views": len(c["views"])}
               for k, c in enumerate(consistent)],
              open(os.path.join(args.out_dir, "parts3d.json"), "w"), indent=2)
    print(f"\n저장: {args.out_dir} (parts3d.npz/json)")
    print("판정: 일관 part가 객체를 의미있게 분할(예 상판/다리)하고 views가 충분하면 Step 2 OK.")
    print("다음 Step 3: 각 part의 view-support(C1)+GS수/가시뷰(C2)+기하 distinctness(C3)로 granularity 선택.")


if __name__ == "__main__":
    main()
