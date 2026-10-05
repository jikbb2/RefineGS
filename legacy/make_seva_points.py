#!/usr/bin/env python3
"""SEVA 생성 뷰 → unknown 표면 점군 (prior 주입용).

각 생성 이미지에 Metric3D(단안 depth+normal)를 돌리고, **관측 메쉬(TSDF) raycast depth**를
앵커로 scale-shift 정렬 → 픽셀을 3D로 올린 뒤 **unknown 영역 근처만** 유지(환각/배경 제거).

  conda activate mono
  python make_seva_points.py \
    --scene_dir ~/prior/seva/obj6 \
    --samples <seva_repo>/work_dirs/demo/img2trajvid/obj6/samples-rgb \
    --tsdf output/replica_room0_v2/refinegs_full/6/train/ours_7000/fuse_post.ply \
    --unknown ~/prior/obj6/unknown.ply \
    --out ~/prior/obj6/seva_points.ply

전제: samples-rgb/NNN.png 는 scene 의 test_ids 순서, 정사각 center-crop 후 리사이즈된 출력.
Deps: torch(cu121), open3d, opencv, numpy.
"""
import os
import json
import glob
import argparse
import numpy as np
import cv2
import torch
import open3d as o3d
from scipy.spatial import cKDTree

GL2CV = np.diag([1.0, -1.0, -1.0])


def metric3d_infer(model, rgb):
    H, W = rgb.shape[:2]
    isz = (616, 1064)
    mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
    sc = min(isz[0] / H, isz[1] / W)
    rs = cv2.resize(rgb, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_LINEAR)
    h, w = rs.shape[:2]
    ph, pw = isz[0] - h, isz[1] - w
    pt, pb, pl, pr = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    canvas = cv2.copyMakeBorder(rs, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
    x = torch.from_numpy(canvas.transpose(2, 0, 1)).float()
    x = ((x - mean) / std)[None].cuda()
    with torch.no_grad():
        pred, _, out = model.inference({"input": x})
    d = pred.squeeze()[pt:isz[0] - pb, pl:isz[1] - pr]
    d = torch.nn.functional.interpolate(d[None, None], (H, W), mode="bilinear")[0, 0].cpu().numpy()
    n = None
    if "prediction_normal" in out:
        n = out["prediction_normal"][0, :3, pt:isz[0] - pb, pl:isz[1] - pr]
        n = torch.nn.functional.interpolate(n[None], (H, W), mode="bilinear")[0]
        n = torch.nn.functional.normalize(n, dim=0).cpu().numpy()
    return d, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="make_seva_scene 출력(transforms.json + split)")
    ap.add_argument("--samples", required=True, help="SEVA samples-rgb 폴더")
    ap.add_argument("--tsdf", required=True, help="관측 메쉬 — scale 앵커 + '새 표면' 판정")
    ap.add_argument("--unknown", required=True, help="unknown 껍질 점군 — 공간 필터")
    ap.add_argument("--model", default="metric3d_vit_small")
    ap.add_argument("--near_unknown", type=float, default=0.12,
                    help="unknown 점에서 이 거리(m) 이내 생성점만 유지(환각·배경 제거)")
    ap.add_argument("--surf_band", type=float, default=0.03,
                    help="TSDF 표면에서 이 거리(m) 이내는 이미 관측 → 제외")
    ap.add_argument("--min_anchor", type=int, default=800, help="scale 정렬 최소 앵커 픽셀")
    ap.add_argument("--max_pts_view", type=int, default=20000)
    ap.add_argument("--save_depth", action="store_true", help="정렬된 depth npy 저장(NV depth 감독용)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sdir = os.path.expanduser(args.scene_dir)
    tr = json.load(open(os.path.join(sdir, "transforms.json")))
    spl = json.load(open(sorted(glob.glob(os.path.join(sdir, "train_test_split_*.json")))[0]))
    test_ids = spl["test_ids"]
    W0, H0 = int(tr["w"]), int(tr["h"])
    fx0, fy0, cx0, cy0 = tr["fl_x"], tr["fl_y"], tr["cx"], tr["cy"]

    imgs = sorted(glob.glob(os.path.join(os.path.expanduser(args.samples), "*.png")))
    assert imgs, f"생성 이미지 없음: {args.samples}"
    assert len(imgs) <= len(test_ids), f"이미지 {len(imgs)} > test_ids {len(test_ids)}"
    print(f"생성 뷰 {len(imgs)} / test_ids {len(test_ids)}")

    tm = o3d.io.read_triangle_mesh(os.path.expanduser(args.tsdf))
    TV = np.asarray(tm.vertices)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tm))
    tsdf_tree = cKDTree(TV)
    uk = np.asarray(o3d.io.read_point_cloud(os.path.expanduser(args.unknown)).points)
    uk_tree = cKDTree(uk)

    model = torch.hub.load("yvanyin/metric3d", args.model, pretrain=True).cuda().eval()

    P_all, N_all, C_all = [], [], []
    n_ok = n_skip = 0
    for k, ip in enumerate(imgs):
        fi = test_ids[k]
        fr = tr["frames"][fi]
        c2w_gl = np.asarray(fr["transform_matrix"], float)
        c2w = c2w_gl.copy()
        c2w[:3, :3] = c2w_gl[:3, :3] @ GL2CV            # OpenGL → CV
        R_c2w, cam = c2w[:3, :3], c2w[:3, 3]

        rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        gh, gw = rgb.shape[:2]
        # 정사각 center-crop + 리사이즈에 맞춘 intrinsic 재계산
        s = min(W0, H0)
        x0, y0 = (W0 - s) / 2, (H0 - s) / 2
        sc = gw / s
        fx, fy = fx0 * sc, fy0 * sc
        cx, cy = (cx0 - x0) * sc, (cy0 - y0) * sc

        # 앵커: 관측 메쉬 raycast depth
        uu, vv = np.meshgrid(np.arange(gw), np.arange(gh))
        dc = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        dw = dc @ R_c2w.T
        dwn = dw / (np.linalg.norm(dw, axis=1, keepdims=True) + 1e-9)
        rays = np.concatenate([np.broadcast_to(cam.astype(np.float32), dwn.shape),
                               dwn.astype(np.float32)], 1)
        th = rc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(th)
        fwd = R_c2w[:, 2]
        zmesh = np.where(hit, (dwn * th[:, None]) @ fwd, 0).reshape(gh, gw)

        dep, nrm = metric3d_infer(model, rgb)
        anchor = (zmesh > 1e-3)
        if anchor.sum() < args.min_anchor:
            n_skip += 1; continue
        A = np.stack([dep[anchor], np.ones(anchor.sum(), np.float32)], 1)
        sol, *_ = np.linalg.lstsq(A, zmesh[anchor], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if a <= 0:
            n_skip += 1; continue
        dz = dep * a + b

        # 3D 로 올리고 unknown 근처 & 새 표면만 유지
        vs, us = np.nonzero(dz > 1e-3)
        if len(vs) > args.max_pts_view * 4:
            sel = np.random.choice(len(vs), args.max_pts_view * 4, replace=False)
            vs, us = vs[sel], us[sel]
        d = dz[vs, us]
        x = (us - cx) / fx * d
        y = (vs - cy) / fy * d
        Xw = np.stack([x, y, d], 1) @ R_c2w.T + cam
        d_uk, _ = uk_tree.query(Xw, workers=-1)
        d_ts, _ = tsdf_tree.query(Xw, workers=-1)
        keep = (d_uk < args.near_unknown) & (d_ts > args.surf_band)
        if keep.sum() == 0:
            n_skip += 1; continue
        Xw, vs2, us2 = Xw[keep], vs[keep], us[keep]
        if len(Xw) > args.max_pts_view:
            sel = np.random.choice(len(Xw), args.max_pts_view, replace=False)
            Xw, vs2, us2 = Xw[sel], vs2[sel], us2[sel]

        if nrm is not None:
            nw = nrm[:, vs2, us2].T @ R_c2w.T
        else:
            nw = np.tile([0, 0, 1.0], (len(Xw), 1))
        vdir = cam[None] - Xw
        flip = (nw * vdir).sum(-1) < 0
        nw[flip] = -nw[flip]
        nw /= (np.linalg.norm(nw, axis=1, keepdims=True) + 1e-9)

        P_all.append(Xw); N_all.append(nw)
        C_all.append(rgb[vs2, us2].astype(np.float64) / 255.0)
        if args.save_depth:
            np.save(os.path.join(os.path.expanduser(args.samples), f"depth_{k:03d}.npy"),
                    dz.astype(np.float16))
        n_ok += 1
        print(f"[{k:03d}] frame {fi}  a={a:.3f} b={b:.3f}  앵커 {anchor.sum():6d}  유지 {len(Xw):6d}")

    assert P_all, "생성 점군 0 — near_unknown/surf_band 완화 또는 앵커 확인"
    P = np.concatenate(P_all); N = np.concatenate(N_all); C = np.concatenate(C_all)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.normals = o3d.utility.Vector3dVector(N)
    pc.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_point_cloud(out, pc)
    print(f"\n{n_ok}뷰 사용 / {n_skip} skip → {len(P)}점 → {out}")
    print(f"  centroid {P.mean(0).round(3).tolist()}  extent {(P.max(0)-P.min(0)).round(3).tolist()}")


if __name__ == "__main__":
    main()
