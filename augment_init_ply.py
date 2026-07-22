#!/usr/bin/env python3
"""prior 점군(See3D→Metric3D)을 2DGS gaussian 으로 변환해 init ply 에 병합.

미관측 영역에 gaussian 을 실제로 심어 NV 감독이 '밀 대상'을 갖게 한다(후속 태스크 C).
prior gaussian 은 낮은 opacity 로 시작 → 관측 뷰 손실이 살릴지/죽일지 검증하게 둔다.

  python augment_init_ply.py \
    --recon_ply output/replica_room0_v2/refinegs_full/6/point_cloud/iteration_7000/point_cloud.ply \
    --prior_ply ~/See3D/dataset/stage6/gen_points_g6.ply \
    --out output/replica_room0_v2/refinegs_full/6/point_cloud/iteration_7000/point_cloud_aug.ply \
    --init_opacity 0.1 --init_scale 0.01

recon_ply 의 속성 스키마(2DGS: xyz, normals, f_dc, f_rest, opacity, scale_0/1, rot_0..3)를
그대로 따라 prior gaussian 을 채운다. 스키마 불일치 시 recon 헤더에 맞춰 자동 정렬.
Deps: numpy, plyfile.
"""
import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_ply", required=True)
    ap.add_argument("--prior_ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init_opacity", type=float, default=0.1, help="prior gaussian 시작 opacity(낮게)")
    ap.add_argument("--init_scale", type=float, default=0.01, help="prior gaussian 시작 크기(m, log 전)")
    ap.add_argument("--voxel", type=float, default=0.0, help=">0 이면 prior 점 voxel 다운샘플(m)")
    args = ap.parse_args()

    rec = PlyData.read(os.path.expanduser(args.recon_ply))
    rv = rec["vertex"]
    props = [p.name for p in rv.properties]
    n_rec = len(rv[props[0]])
    print(f"recon gaussian {n_rec}, 속성 {len(props)}개: {props[:12]}...")

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

    # rotation: 법선을 z축으로 하는 quaternion (2DGS surfel: 3번째 축 = 법선)
    def normal_to_quat(nrm):
        nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        z = np.array([0, 0, 1.0])
        v = np.cross(np.tile(z, (len(nrm), 1)), nrm)
        w = 1.0 + (nrm @ z)
        q = np.concatenate([w[:, None], v], 1)         # (w,x,y,z)
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        return q.astype(np.float32)
    Q = normal_to_quat(N)

    # 새 gaussian 속성 딕셔너리 (recon 스키마에 맞춰 채움)
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
            # 저장은 logit (inverse sigmoid)
            o = np.clip(args.init_opacity, 1e-4, 1 - 1e-4)
            vals[name] = np.full(n_pri, np.log(o / (1 - o)), np.float32)
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
            vals[name] = np.zeros(n_pri, np.float32)     # 기타(id_*, desc_* 등)는 0

    # recon + prior concat
    out_arr = np.empty(n_rec + n_pri, dtype=rv.data.dtype)
    for name in props:
        out_arr[name][:n_rec] = rv[name]
        out_arr[name][n_rec:] = vals[name]

    el = PlyElement.describe(out_arr, "vertex")
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    PlyData([el]).write(out)
    print(f"\n병합: recon {n_rec} + prior {n_pri} = {n_rec + n_pri} gaussian → {out}")
    print(f"prior 초기값: opacity {args.init_opacity}, scale {args.init_scale}m "
          f"(관측 뷰 손실이 검증·정제)")


if __name__ == "__main__":
    main()
