#!/usr/bin/env python3
"""prior 점군(TRELLIS dense 등)을 2DGS gaussian 으로 변환해 init ply 에 병합.

미관측 영역에 gaussian 을 실제로 심어 관측 뷰 손실이 '밀 대상'을 갖게 한다.
prior gaussian 은 낮은 opacity 로 시작 → 관측 뷰 손실이 살릴지/죽일지 검증하게 둔다.

[신규] 공간 가변 opacity(--spatial_opacity):
  관측 표면(recon gaussian) 근처 prior 는 아주 낮은 opacity → 실측이 지배(할루시네이션 억제),
  미관측(다리 등) 먼 prior 는 보통 opacity → 유일한 신호로서 형상을 채운다.
  near_band 안에서 near_opacity→far_opacity 로 매끄럽게 램프.

  python augment_init_ply.py \
    --recon_ply output/.../6/point_cloud/iteration_7000/point_cloud.ply \
    --prior_ply ~/obj6_dense.ply \
    --out output/.../6/point_cloud/iteration_7000/point_cloud_aug.ply \
    --spatial_opacity --near_opacity 0.02 --far_opacity 0.12 --near_band 0.03 --init_scale 0.01

recon_ply 의 속성 스키마(2DGS)를 그대로 따라 prior gaussian 을 채운다.
Deps: numpy, plyfile, scipy.
"""
import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


C0 = 0.28209479177387814


def logit(o):
    o = np.clip(o, 1e-4, 1 - 1e-4)
    return np.log(o / (1 - o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_ply", required=True)
    ap.add_argument("--prior_ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init_opacity", type=float, default=0.1, help="prior 시작 opacity(균일 모드)")
    ap.add_argument("--init_scale", type=float, default=0.01, help="prior gaussian 시작 크기(m, log 전)")
    ap.add_argument("--voxel", type=float, default=0.0, help=">0 이면 prior 점 voxel 다운샘플(m)")
    # 공간 가변 opacity (할루시네이션 억제)
    ap.add_argument("--spatial_opacity", action="store_true",
                    help="관측 근처 prior 는 낮은 opacity, 미관측은 높은 opacity")
    ap.add_argument("--near_opacity", type=float, default=0.02, help="관측 근접 prior opacity")
    ap.add_argument("--far_opacity", type=float, default=0.12, help="미관측 prior opacity")
    ap.add_argument("--near_band", type=float, default=0.03, help="관측 근접 판정 band(m)")
    args = ap.parse_args()

    rec = PlyData.read(os.path.expanduser(args.recon_ply))
    rv = rec["vertex"]
    props = [p.name for p in rv.properties]
    n_rec = len(rv[props[0]])
    print(f"recon gaussian {n_rec}, 속성 {len(props)}개: {props[:12]}...")
    R_xyz = np.stack([rv["x"], rv["y"], rv["z"]], -1).astype(np.float32)

    pp = PlyData.read(os.path.expanduser(args.prior_ply))["vertex"]
    P = np.stack([pp["x"], pp["y"], pp["z"]], -1).astype(np.float32)
    has_n = all(k in pp.data.dtype.names for k in ("nx", "ny", "nz"))
    N = np.stack([pp["nx"], pp["ny"], pp["nz"]], -1).astype(np.float32) if has_n else \
        np.tile([0, 0, 1.0], (len(P), 1)).astype(np.float32)
    has_c = all(k in pp.data.dtype.names for k in ("red", "green", "blue"))
    if has_c:
        C = np.stack([pp["red"], pp["green"], pp["blue"]], -1).astype(np.float32) / 255.0
    else:
        C = np.full((len(P), 3), 0.5, np.float32)

    if args.voxel > 0:
        key = np.floor(P / args.voxel).astype(np.int64)
        _, uidx = np.unique(key, axis=0, return_index=True)
        P, N, C = P[uidx], N[uidx], C[uidx]
        print(f"voxel {args.voxel}m 다운샘플 → prior {len(P)}점")
    n_pri = len(P)

    # --- opacity(logit) 결정: 균일 or 공간 가변 ---
    if args.spatial_opacity:
        d = cKDTree(R_xyz).query(P)[0]                       # prior→관측 최근접 거리
        w = np.clip(d / max(args.near_band, 1e-6), 0, 1)     # 0(관측근접)~1(미관측)
        op = args.near_opacity + w * (args.far_opacity - args.near_opacity)
        op_logit = logit(op).astype(np.float32)
        near_n = int((d < args.near_band).sum())
        print(f"공간 가변 opacity: 관측근접 {near_n}점→{args.near_opacity}, "
              f"미관측 {n_pri - near_n}점→최대 {args.far_opacity} (band {args.near_band}m)")
    else:
        op_logit = np.full(n_pri, logit(args.init_opacity), np.float32)

    # rotation: 법선을 3번째 축으로 하는 quaternion (2DGS surfel)
    def normal_to_quat(nrm):
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        z = np.array([0, 0, 1.0])
        v = np.cross(np.tile(z, (len(nrm), 1)), nrm)
        w = 1.0 + (nrm @ z)
        q = np.concatenate([w[:, None], v], 1)
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        return q.astype(np.float32)
    Q = normal_to_quat(N)

    vals = {}
    for name in props:
        if name in ("x", "y", "z"):
            vals[name] = P[:, {"x": 0, "y": 1, "z": 2}[name]]
        elif name in ("nx", "ny", "nz"):
            vals[name] = N[:, {"nx": 0, "ny": 1, "nz": 2}[name]]
        elif name == "f_dc_0":
            vals[name] = (C[:, 0] - 0.5) / C0
        elif name == "f_dc_1":
            vals[name] = (C[:, 1] - 0.5) / C0
        elif name == "f_dc_2":
            vals[name] = (C[:, 2] - 0.5) / C0
        elif name.startswith("f_rest"):
            vals[name] = np.zeros(n_pri, np.float32)
        elif name == "opacity":
            vals[name] = op_logit                            # 균일 or 공간 가변
        elif name.startswith("scale_"):
            vals[name] = np.full(n_pri, np.log(args.init_scale), np.float32)
        elif name == "rot_0":
            vals[name] = Q[:, 0]
        elif name == "rot_1":
            vals[name] = Q[:, 1]
        elif name == "rot_2":
            vals[name] = Q[:, 2]
        elif name == "rot_3":
            vals[name] = Q[:, 3]
        else:
            vals[name] = np.zeros(n_pri, np.float32)

    out_arr = np.empty(n_rec + n_pri, dtype=rv.data.dtype)
    for name in props:
        out_arr[name][:n_rec] = rv[name]
        out_arr[name][n_rec:] = vals[name]

    el = PlyElement.describe(out_arr, "vertex")
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    PlyData([el]).write(out)
    print(f"\n병합: recon {n_rec} + prior {n_pri} = {n_rec + n_pri} gaussian → {out}")


if __name__ == "__main__":
    main()
