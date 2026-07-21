#!/usr/bin/env python3
"""See3D 생성 뷰 → unseen 표면 점군 (SDF prior 주입용).

각 gen 뷰에 Metric3D(단안 depth+normal)를 돌려 depth 를 얻고, 그 뷰의 weight(생성 영역)
픽셀만 back-project. 단안 depth 는 스케일 모호하므로 **같은 뷰의 GT-warp known 픽셀**과
scale-shift 정렬(RANSAC-free 최소제곱)해 씬 스케일에 앵커한다. known 이 부족한 뷰는 skip.

  conda activate mono   # torch cu121 + Metric3D hub
  python make_gen_points.py \
    --gen_dir ~/See3D/dataset/stage6/gen_adapt7 \
    --soft_in ~/See3D/dataset/stage6/see3d_in_adapt7 \
    --warp_dir ~/See3D/dataset/stage6/soft_in_adapt7 \
    --out ~/See3D/dataset/stage6/gen_points.ply

출력: 법선 포함 ply → sdf_distill_depth.py --extra_points 로 사용.
Deps: torch(cu121), open3d, opencv, numpy.
"""
import os
import glob
import argparse
import numpy as np
import cv2
import torch
import open3d as o3d


def cam_from_rec(rec):
    wvt = np.asarray(rec["world_view_transform"], float)
    M = wvt.T                       # w2c
    R, t = M[:3, :3], M[:3, 3]
    W, H = int(rec["width"]), int(rec["height"])
    fx = W / (2 * np.tan(float(rec["FoVx"]) / 2))
    fy = H / (2 * np.tan(float(rec["FoVy"]) / 2))
    return R, t, fx, fy, W / 2.0, H / 2.0, W, H


def metric3d_infer(model, rgb, fx, fy, cx, cy):
    """Metric3D vit → (depth[H,W], normal[3,H,W] camera-space)."""
    H, W = rgb.shape[:2]
    input_size = (616, 1064)
    mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
    sc = min(input_size[0] / H, input_size[1] / W)
    rs = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_LINEAR)
    h, w = rs.shape[:2]
    ph, pw = input_size[0] - h, input_size[1] - w
    pt, pb, pl, pr = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    canvas = cv2.copyMakeBorder(rs, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
    x = torch.from_numpy(canvas.transpose(2, 0, 1)).float()
    x = ((x - mean) / std)[None].cuda()
    with torch.no_grad():
        pred_depth, _, out = model.inference({"input": x})
    d = pred_depth.squeeze()[pt:input_size[0] - pb, pl:input_size[1] - pr]
    d = torch.nn.functional.interpolate(d[None, None], (H, W), mode="bilinear")[0, 0]
    # Metric3D canonical → 실제 초점거리 보정
    d = d * (1000.0 / (fx / sc)) if False else d
    n = None
    if "prediction_normal" in out:
        n = out["prediction_normal"][0, :3, pt:input_size[0] - pb, pl:input_size[1] - pr]
        n = torch.nn.functional.interpolate(n[None], (H, W), mode="bilinear")[0]
        n = torch.nn.functional.normalize(n, dim=0)
    return d.cpu().numpy(), (n.cpu().numpy() if n is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="gen_%04d.jpg + weight_%04d.png + poses.npz")
    ap.add_argument("--soft_in", required=True, help="see3d 입력(크롭 view/weight/poses) — weight 기준")
    ap.add_argument("--warp_dir", required=True, help="원본 워프 폴더 — known depth(scale 앵커)용 depth_%04d.npy")
    ap.add_argument("--model", default="metric3d_vit_small")
    ap.add_argument("--min_anchor", type=int, default=500, help="scale 정렬에 필요한 최소 known 픽셀")
    ap.add_argument("--max_pts_view", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gen = os.path.expanduser(args.gen_dir)
    sd = os.path.expanduser(args.soft_in)
    wd = os.path.expanduser(args.warp_dir)
    recs = {int((r.item() if hasattr(r, "item") and not isinstance(r, dict) else r)["idx"]):
            (r.item() if hasattr(r, "item") and not isinstance(r, dict) else r)
            for r in np.load(os.path.join(sd, "poses.npz"), allow_pickle=True)["records"]}

    model = torch.hub.load("yvanyin/metric3d", args.model, pretrain=True).cuda().eval()

    P_all, N_all, C_all = [], [], []
    n_ok = n_skip = 0
    for gp in sorted(glob.glob(gen + "/gen_*.jpg")):
        i = int(os.path.basename(gp)[4:8])
        rec = recs.get(i)
        wp = os.path.join(sd, f"weight_{i:04d}.png")
        if rec is None or not os.path.exists(wp):
            continue
        R, t, fx, fy, cx, cy, W, H = cam_from_rec(rec)
        rgb = cv2.cvtColor(cv2.imread(gp), cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H))
        gw = cv2.imread(wp, cv2.IMREAD_GRAYSCALE)
        gw = cv2.resize(gw, (W, H), interpolation=cv2.INTER_NEAREST) > 127   # 생성 영역

        dep, nrm = metric3d_infer(model, rgb, fx, fy, cx, cy)

        # scale-shift 정렬: 같은 뷰의 warp known depth 를 앵커로 (a*d + b ≈ d_warp)
        dnp = os.path.join(wd, f"depth_{i:04d}.npy")
        if not os.path.exists(dnp):
            n_skip += 1; continue
        wdep_full = np.load(dnp).astype(np.float32)
        # soft_in 이 크롭본이면 크기 불일치 → 리사이즈로 근사 정렬
        if wdep_full.shape != (H, W):
            wdep_full = cv2.resize(wdep_full, (W, H), interpolation=cv2.INTER_NEAREST)
        anchor = (wdep_full > 1e-3) & (~gw)
        if anchor.sum() < args.min_anchor:
            n_skip += 1; continue
        A = np.stack([dep[anchor], np.ones(anchor.sum(), np.float32)], 1)
        sol, *_ = np.linalg.lstsq(A, wdep_full[anchor], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if a <= 0:
            n_skip += 1; continue
        dep_al = dep * a + b

        vs, us = np.nonzero(gw & (dep_al > 1e-3))
        if len(vs) == 0:
            n_skip += 1; continue
        if len(vs) > args.max_pts_view:
            sel = np.random.choice(len(vs), args.max_pts_view, replace=False)
            vs, us = vs[sel], us[sel]
        d = dep_al[vs, us]
        x = (us - cx) / fx * d
        y = (vs - cy) / fy * d
        Xc = np.stack([x, y, d], 1)
        Xw = (Xc - t) @ R                                  # world
        if nrm is not None:
            nc = nrm[:, vs, us].T
            nw = nc @ R                                    # camera→world
        else:
            nw = np.tile(np.array([0, 0, 1.0]), (len(Xw), 1))
        cam_c = -R.T @ t
        vdir = cam_c[None] - Xw
        flip = (nw * vdir).sum(-1) < 0
        nw[flip] = -nw[flip]
        nw /= (np.linalg.norm(nw, axis=1, keepdims=True) + 1e-9)

        P_all.append(Xw); N_all.append(nw)
        C_all.append(rgb[vs, us].astype(np.float64) / 255.0)
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  {n_ok}뷰 처리 (scale a={a:.3f} b={b:.3f}, {len(Xw)}점)")

    assert P_all, "생성 점군 0 — weight/depth 경로 확인"
    P = np.concatenate(P_all); N = np.concatenate(N_all); C = np.concatenate(C_all)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.normals = o3d.utility.Vector3dVector(N)
    pc.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_point_cloud(out, pc)
    print(f"\n{n_ok}뷰 사용 / {n_skip}뷰 skip → {len(P)}점 → {out}")


if __name__ == "__main__":
    main()
