#!/usr/bin/env python3
"""객체 메쉬 일대일(Hungarian) 매칭 평가 — auto_match 다대일 붕괴 수정판.

기존 eval_object_mesh --auto_match 는 recon 별 greedy 매칭이라 쿠션 여러 개가
같은 GT(gt77 등)에 붙어 comp 가 아티팩트로 폭발. 이 스크립트는:
  1) GT semantic mesh 에서 object_id 별 서브메쉬 추출
  2) recon×GT 전체 chamfer 비용행렬 → scipy Hungarian 으로 전역 1:1 배정
  3) 배정 쌍에 표준 지표(acc/comp/chamfer/NC/P·R·F@τ) + 요약 통계

  python eval_objects_1to1.py \
    --gt_mesh ~/room_0/habitat/mesh_semantic.ply \
    --recon_glob "output/replica_room0_v2/refinegs_full/*/train/ours_*/fuse_post.ply" \
    --exclude merged_bak --out ~/tmp/eval_1to1.csv

Deps: numpy, scipy, open3d, plyfile.
"""
import os
import re
import glob
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def load_gt_objects(gt_path, min_faces=50):
    """semantic ply → {object_id: o3d mesh}. quad face 자동 삼각화."""
    ply = PlyData.read(os.path.expanduser(gt_path))
    v = ply["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    f = ply["face"]
    fname = [n for n in f.data.dtype.names if "vertex" in n][0]
    oid_name = [n for n in f.data.dtype.names if "object" in n.lower()]
    assert oid_name, f"GT face에 object_id 속성 없음: {f.data.dtype.names}"
    oids = np.asarray(f[oid_name[0]])

    faces = np.vstack([np.asarray(x) for x in f[fname]])   # (N,3) 또는 (N,4)
    k = faces.shape[1]
    if k == 3:
        tris, toids = faces, oids
    elif k == 4:                                            # quad → 삼각형 2개
        tris = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
        toids = np.concatenate([oids, oids])
    else:
        raise SystemExit(f"지원 안 되는 face 크기: {k}")

    out = {}
    for oid in np.unique(toids):
        fi = tris[toids == oid]
        if len(fi) < min_faces:
            continue
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(np.ascontiguousarray(fi.astype(np.int32))))
        m.remove_unreferenced_vertices()
        out[int(oid)] = m
    return out


def sample(mesh, n):
    pc = mesh.sample_points_uniformly(n, use_triangle_normal=True)
    return np.asarray(pc.points), np.asarray(pc.normals)


def chamfer_quick(a, b):
    da, _ = cKDTree(b).query(a, workers=-1)
    db, _ = cKDTree(a).query(b, workers=-1)
    return (da.mean() + db.mean()) / 2


def full_metrics(rp, rn, gp, gn, taus):
    tg, tr = cKDTree(gp), cKDTree(rp)
    d_rg, i_rg = tg.query(rp, workers=-1)   # recon→gt (acc/precision)
    d_gr, i_gr = tr.query(gp, workers=-1)   # gt→recon (comp/recall)
    acc, comp = d_rg.mean(), d_gr.mean()
    nc = np.abs((rn * gn[i_rg]).sum(-1)).mean()
    row = dict(accuracy=acc, completion=comp, chamfer_l1=(acc + comp) / 2, normal_consistency=nc)
    for t in taus:
        p, r = (d_rg < t).mean(), (d_gr < t).mean()
        row[f"precision@{t}"] = p; row[f"recall@{t}"] = r
        row[f"f@{t}"] = 2 * p * r / max(p + r, 1e-9)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--recon_glob", required=True)
    ap.add_argument("--exclude", nargs="*", default=["merged_bak"])
    ap.add_argument("--n_pts", type=int, default=100000)
    ap.add_argument("--match_pts", type=int, default=8000)
    ap.add_argument("--taus", nargs="*", type=float, default=[0.005, 0.01, 0.02])
    ap.add_argument("--fail_thr", type=float, default=0.15, help="chamfer(m) 초과 시 실패로 분류(요약 통계 별도)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recons = []
    for p in sorted(glob.glob(args.recon_glob)):
        if any(x in p for x in args.exclude):
            print(f"[skip] {p}")
            continue
        m = re.search(r"refinegs_full/([^/]+)/", p)
        label = m.group(1) if m else os.path.basename(os.path.dirname(p))
        recons.append((label, p))
    assert recons, "recon 없음"
    print(f"recon {len(recons)}개")

    gt = load_gt_objects(args.gt_mesh)
    gids = sorted(gt.keys())
    print(f"GT 객체 {len(gids)}개")

    # 매칭용 서브샘플
    r_sub, g_sub = [], []
    r_mesh = []
    for label, p in recons:
        m = o3d.io.read_triangle_mesh(p)
        r_mesh.append(m)
        r_sub.append(sample(m, args.match_pts)[0] if len(m.triangles) else None)
    for gid in gids:
        g_sub.append(sample(gt[gid], args.match_pts)[0])

    # 비용행렬 (chamfer) → Hungarian 1:1
    C = np.full((len(recons), len(gids)), 1e6)
    for i, rs in enumerate(r_sub):
        if rs is None:
            continue
        # 후보 프루닝: centroid 1.5m 이내만 정밀 계산
        rc = rs.mean(0)
        for j, gs in enumerate(g_sub):
            if np.linalg.norm(gs.mean(0) - rc) < 1.5:
                C[i, j] = chamfer_quick(rs, gs)
    ri, gj = linear_sum_assignment(C)

    import csv
    rows = []
    for i, j in zip(ri, gj):
        label, path = recons[i]
        if C[i, j] >= 1e6:
            rows.append(dict(label=label, gt_id=-1, match_cost=np.nan, recon=path)); continue
        rp, rn = sample(r_mesh[i], args.n_pts)
        gp, gn = sample(gt[gids[j]], args.n_pts)
        row = full_metrics(rp, rn, gp, gn, args.taus)
        row.update(label=label, gt_id=gids[j], match_cost=C[i, j], recon=path)
        rows.append(row)
        print(f"{label:>14} → gt{gids[j]:<3} chamfer {row['chamfer_l1']*1000:6.1f}mm  "
              f"F@1cm {row['f@0.01']:.3f}  NC {row['normal_consistency']:.3f}")

    keys = ["label", "gt_id", "match_cost", "accuracy", "completion", "chamfer_l1",
            "normal_consistency"] + sum([[f"precision@{t}", f"recall@{t}", f"f@{t}"] for t in args.taus], []) + ["recon"]
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    ok = [r for r in rows if r.get("chamfer_l1", 1e9) < args.fail_thr]
    bad = [r for r in rows if r not in ok]
    if ok:
        ch = np.array([r["chamfer_l1"] for r in ok]) * 1000
        f1 = np.array([r["f@0.01"] for r in ok])
        print(f"\n=== 요약 (성공 {len(ok)} / 실패 {len(bad)}, 실패기준 chamfer>{args.fail_thr*1000:.0f}mm) ===")
        print(f"chamfer  mean {ch.mean():.1f}mm  median {np.median(ch):.1f}mm")
        print(f"F@1cm    mean {f1.mean():.3f}  median {np.median(f1):.3f}")
    if bad:
        print("실패 목록:", [r["label"] for r in bad])
    print(f"→ {out}")


if __name__ == "__main__":
    main()
