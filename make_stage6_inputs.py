#!/usr/bin/env python3
"""STAGE 6/6.5 입력 v2 — 객체 실루엣 교차 weight + 줌 크롭(비대칭 frustum 재계산).

soft_in(warp_gt_to_pose 출력)에서:
  nv_out    : gen=view(실픽셀), weight = known ∧ 객체실루엣   ← STAGE 6 감독을 객체에 집중
  see3d_out : view 그대로,      weight = unobs ∧ 객체실루엣   ← 생성/감독 대상을 객체에 한정
--zoom > 0 이면 객체 bbox×zoom 정사각 창으로 view/weight를 크롭하고,
poses.npz record를 크롭 카메라(비대칭 frustum full_proj + 재산출 FoV)로 갱신
→ NV 로더/렌더는 무수정으로 크롭 카메라를 그대로 사용, See3D 512²가 객체에 집중(유효 해상도 ↑).

  python make_stage6_inputs.py --soft_in ~/See3D/dataset/stage6/soft_in_all \
    --recon_root output/replica_room0_v2/refinegs_full \
    --nv_out ~/See3D/dataset/stage6/nv_all --see3d_out ~/See3D/dataset/stage6/see3d_in_all \
    --zoom 1.6

Deps: numpy, PIL, open3d, scipy.
"""
import os
import re
import glob
import shutil
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation


def cam_params(rec):
    W, H = int(rec["width"]), int(rec["height"])
    fx = W / (2 * np.tan(float(rec["FoVx"]) / 2))
    fy = H / (2 * np.tan(float(rec["FoVy"]) / 2))
    return fx, fy, W, H


def target_from_pose(rec):
    wvt = np.asarray(rec["world_view_transform"], float)
    M = wvt.T
    fx, fy, W, H = cam_params(rec)
    return M[:3, :3], M[:3, 3], fx, fy, W / 2.0, H / 2.0, W, H


def obj_silhouette(verts, rec, dil=6):
    R, t, fx, fy, cx, cy, W, H = target_from_pose(rec)
    pc = verts @ R.T + t
    z = pc[:, 2]
    ok = z > 1e-3
    u = np.clip(fx * pc[ok, 0] / z[ok] + cx, 0, W - 1).astype(np.int64)
    v = np.clip(fy * pc[ok, 1] / z[ok] + cy, 0, H - 1).astype(np.int64)
    m = np.zeros((H, W), bool)
    m[v, u] = True
    return binary_dilation(m, iterations=dil)


def crop_window(sil, zoom, min_side, W, H):
    """실루엣 bbox×zoom 정사각 창 (프레임 내 클램프). (x0,x1,y0,y1)"""
    ys, xs = np.nonzero(sil)
    cx_, cy_ = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * zoom
    side = int(np.clip(side, min_side, min(W, H)))
    x0 = int(np.clip(cx_ - side / 2, 0, W - side))
    y0 = int(np.clip(cy_ - side / 2, 0, H - side))
    return x0, x0 + side, y0, y0 + side


def crop_record(rec, x0, x1, y0, y1, znear=0.01, zfar=100.0):
    """크롭 창에 맞는 비대칭 frustum record 재계산 (focal 유지, 주점 이동)."""
    fx, fy, W, H = cam_params(rec)
    Wc, Hc = x1 - x0, y1 - y0
    P = np.zeros((4, 4))
    P[0, 0] = 2 * fx / Wc
    P[1, 1] = 2 * fy / Hc
    P[0, 2] = (W - x0 - x1) / Wc          # 주점 오프셋 (대칭 크롭이면 0)
    P[1, 2] = (H - y0 - y1) / Hc
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    wvt = np.asarray(rec["world_view_transform"], float)
    out = dict(rec)
    out["width"], out["height"] = int(Wc), int(Hc)
    out["FoVx"] = float(2 * np.arctan(Wc / (2 * fx)))
    out["FoVy"] = float(2 * np.arctan(Hc / (2 * fy)))
    out["world_view_transform"] = wvt
    out["full_proj_transform"] = wvt @ P.T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_in", required=True)
    ap.add_argument("--recon_root", required=True)
    ap.add_argument("--nv_out", required=True)
    ap.add_argument("--see3d_out", required=True)
    ap.add_argument("--n_verts", type=int, default=30000)
    ap.add_argument("--zoom", type=float, default=1.6, help="객체 bbox 대비 크롭 배율. 0=크롭 안 함")
    ap.add_argument("--min_side", type=int, default=200, help="크롭 최소 변(px)")
    ap.add_argument("--min_known_obj", type=float, default=0.15)
    ap.add_argument("--min_unobs_obj", type=float, default=0.10)
    args = ap.parse_args()

    soft = os.path.expanduser(args.soft_in)
    nv = os.path.expanduser(args.nv_out)
    sd = os.path.expanduser(args.see3d_out)
    for d in (nv, sd):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

    recs = {int(r["idx"]): (r.item() if hasattr(r, "item") and not isinstance(r, dict) else r)
            for r in np.load(os.path.join(soft, "poses.npz"), allow_pickle=True)["records"]}

    mesh_cache = {}
    def verts_of(gid):
        if gid not in mesh_cache:
            p = sorted(glob.glob(os.path.join(args.recon_root, gid, "train", "ours_*", "fuse_post.ply")))[-1]
            mesh_cache[gid] = np.asarray(
                o3d.io.read_triangle_mesh(p).sample_points_uniformly(args.n_verts).points)
        return mesh_cache[gid]

    nv_recs, sd_recs = [], []
    for wp in sorted(glob.glob(soft + "/weight_*.png")):
        i = int(os.path.basename(wp)[7:11])
        rec = recs.get(i)
        if rec is None:
            continue
        m = re.match(r"g(.+?)_o", str(rec.get("stem", "")))
        if not m:
            continue
        gid = m.group(1)

        w = np.array(Image.open(wp).convert("L"))
        H, W = w.shape
        sil = obj_silhouette(verts_of(gid), rec)
        if sil.shape != w.shape:
            sil = np.array(Image.fromarray(sil).resize((W, H), Image.NEAREST)) > 0
        if sil.sum() < 50:
            continue
        known_o = (w == 255) & sil
        unobs_o = (w == 128) & sil
        s = max(sil.sum(), 1)
        r_known, r_unobs = known_o.sum() / s, unobs_o.sum() / s

        view = Image.open(os.path.join(soft, f"view_{i:04d}.jpg")).convert("RGB")
        if args.zoom > 0:
            x0, x1, y0, y1 = crop_window(sil, args.zoom, args.min_side, W, H)
            rec_out = crop_record(rec, x0, x1, y0, y1)
            view_c = view.crop((x0, y0, x1, y1))
            known_c = known_o[y0:y1, x0:x1]
            unobs_c = unobs_o[y0:y1, x0:x1]
        else:
            rec_out, view_c, known_c, unobs_c = dict(rec), view, known_o, unobs_o

        if r_known >= args.min_known_obj:
            view_c.save(os.path.join(nv, f"gen_{i:04d}.jpg"), quality=95)
            Image.fromarray((known_c * 255).astype(np.uint8)).save(os.path.join(nv, f"weight_{i:04d}.png"))
            nv_recs.append(rec_out)
        if r_unobs >= args.min_unobs_obj:
            view_c.save(os.path.join(sd, f"view_{i:04d}.jpg"), quality=95)
            Image.fromarray((unobs_c * 255).astype(np.uint8)).save(os.path.join(sd, f"weight_{i:04d}.png"))
            sd_recs.append(rec_out)
        print(f"[{i:04d}] g{gid:>3}  obj-known {r_known:.2f}  obj-unobs {r_unobs:.2f}"
              + (f"  crop {rec_out['width']}x{rec_out['height']}" if args.zoom > 0 else ""))

    np.savez(os.path.join(nv, "poses.npz"), records=np.array(nv_recs, dtype=object))
    np.savez(os.path.join(sd, "poses.npz"), records=np.array(sd_recs, dtype=object))
    print(f"\nNV {len(nv_recs)}뷰 → {nv}\nSee3D {len(sd_recs)}뷰 → {sd}")
    if args.zoom > 0:
        print("(줌 크롭 적용 — poses.npz가 크롭 카메라로 갱신됨. NV 로더/렌더 무수정 호환)")


if __name__ == "__main__":
    main()
