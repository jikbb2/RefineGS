#!/usr/bin/env python3
"""STAGE 6/6.5 QA — See3D 입력·출력이 '객체의 미관측'을 제대로 겨냥했는지 검증.

뷰별로: [warp 조건 | weight | gen | gen+객체 윤곽] 몽타주 저장 + 지표:
  focus    = (unobs ∧ 객체실루엣) / unobs   ← 감독이 목표 객체에 집중된 비율 (핵심)
  cover    = (unobs ∧ 객체실루엣) / 객체실루엣 ← 객체 중 미관측(생성 몫) 비율
  known_err= known 픽셀에서 |gen - warp| 평균 (0~255, 생성이 실픽셀을 존중했는지)

  python stage6_qa.py --soft_in ~/See3D/dataset/stage6/soft_in_v2 \
    --gen_dir ~/See3D/dataset/stage6/gen65 \
    --recon_root output/replica_room0_v2/refinegs_full \
    --out ~/stage6_qa

Deps: numpy, PIL, open3d, scipy.
"""
import os
import re
import glob
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation


def target_from_pose(rec):
    wvt = np.asarray(rec["world_view_transform"], float)
    M = wvt.T
    R, t = M[:3, :3], M[:3, 3]
    W, H = int(rec["width"]), int(rec["height"])
    fx = W / (2 * np.tan(float(rec["FoVx"]) / 2))
    fy = H / (2 * np.tan(float(rec["FoVy"]) / 2))
    return R, t, fx, fy, W / 2.0, H / 2.0, W, H


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_in", required=True)
    ap.add_argument("--gen_dir", default="", help="gen_%04d.jpg 폴더(없으면 입력 QA만)")
    ap.add_argument("--recon_root", required=True)
    ap.add_argument("--n_verts", type=int, default=30000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    soft = os.path.expanduser(args.soft_in)
    gen_dir = os.path.expanduser(args.gen_dir) if args.gen_dir else ""
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    recs = {int(r["idx"]): (r.item() if hasattr(r, "item") and not isinstance(r, dict) else r)
            for r in np.load(os.path.join(soft, "poses.npz"), allow_pickle=True)["records"]}

    mesh_cache = {}
    def verts_of(gid):
        if gid not in mesh_cache:
            p = sorted(glob.glob(os.path.join(args.recon_root, gid, "train", "ours_*", "fuse_post.ply")))[-1]
            m = o3d.io.read_triangle_mesh(p)
            v = np.asarray(m.sample_points_uniformly(args.n_verts).points)
            mesh_cache[gid] = v
        return mesh_cache[gid]

    per_gid = {}
    for wp in sorted(glob.glob(soft + "/weight_*.png")):
        i = int(os.path.basename(wp)[7:11])
        rec = recs.get(i)
        if rec is None:
            continue
        stem = str(rec.get("stem", ""))
        m = re.match(r"g(.+?)_o", stem)
        if not m:
            continue
        gid = m.group(1)

        w = np.array(Image.open(wp).convert("L"))
        view = np.array(Image.open(os.path.join(soft, f"view_{i:04d}.jpg")).convert("RGB"))
        unobs = w == 128
        known = w == 255
        sil = obj_silhouette(verts_of(gid), rec)
        if sil.shape != w.shape:
            sil = np.array(Image.fromarray(sil).resize((w.shape[1], w.shape[0]), Image.NEAREST)) > 0

        inter = (unobs & sil).sum()
        focus = inter / max(unobs.sum(), 1)
        cover = inter / max(sil.sum(), 1)

        gp = os.path.join(gen_dir, f"gen_{i:04d}.jpg") if gen_dir else ""
        gen = None
        known_err = np.nan
        if gp and os.path.exists(gp):
            gen = np.array(Image.open(gp).convert("RGB").resize((w.shape[1], w.shape[0])))
            if known.sum() > 100:
                known_err = np.abs(gen.astype(float) - view.astype(float))[known].mean()

        # 몽타주: view | weight(색) | gen | gen+실루엣
        wc = np.zeros_like(view)
        wc[known] = (0, 200, 0); wc[unobs] = (255, 140, 0); wc[w == 0] = (60, 60, 60)
        panels = [view, wc]
        if gen is not None:
            ov = gen.copy()
            edge = binary_dilation(sil, iterations=2) & ~sil
            ov[edge] = (255, 0, 0)
            panels += [gen, ov]
        Image.fromarray(np.concatenate(panels, axis=1)).save(
            os.path.join(out, f"qa_{i:04d}_g{gid}.jpg"), quality=85)

        per_gid.setdefault(gid, []).append((focus, cover, known_err))
        print(f"[{i:04d}] g{gid:>3}  focus {focus:.2f}  cover {cover:.2f}"
              + (f"  known_err {known_err:.1f}" if gen is not None else ""))

    print("\n=== gid별 평균 (focus=감독의 객체 집중도 / cover=객체 중 미관측 비율) ===")
    for gid, rows in sorted(per_gid.items()):
        a = np.array(rows, float)
        print(f"g{gid:>3}: 뷰 {len(rows):2d}  focus {np.nanmean(a[:,0]):.2f}  "
              f"cover {np.nanmean(a[:,1]):.2f}  known_err {np.nanmean(a[:,2]):.1f}")
    print(f"\n→ 몽타주: {out}/qa_*.jpg  (녹=known, 주황=unobs(생성/감독 대상), 빨강 윤곽=목표 객체)")


if __name__ == "__main__":
    main()
