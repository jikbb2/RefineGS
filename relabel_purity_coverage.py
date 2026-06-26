#!/usr/bin/env python3
"""RefineGS Axis A — re-labeling purity / coverage vs GT semantic mesh.

목적: SAM3 re-labeling 인스턴스가 (1) 한 GT 객체에만 대응하는가(purity, 섞임 진단),
      (2) 모든 GT 객체를 덮는가(coverage, 누락 진단)를 정량화.

원리: data/<scene> 는 replica_to_refinegs 로 GT 포즈+metric 스케일로 구성되어
      COLMAP 좌표계 == habitat GT 메시 좌표계 (eval_object_mesh.py 전제와 동일).
      따라서 relabel 의 points3d.ply(COLMAP 점)를 GT 메시 표면 샘플에 바로
      nearest-match → 각 점의 GT object_id 를 얻어 인스턴스별 분포를 계산.

정책(합의): over-segmentation 허용, under-segmentation(mixing) 금지.
  - purity 낮음  = 한 인스턴스가 여러 GT 객체에 걸침 = mixing(해로움).
  - coverage 빠짐 = 어떤 GT 객체도 dominant 로 못 가짐 = 누락.
  - 한 GT 객체가 여러 인스턴스의 dominant = over-seg(무해, 정보용으로만 표시).

실행 (split_and_splat env, 서버):
  python relabel_purity_coverage.py \
    --gt_mesh /home/elicer/room_0/habitat/mesh_semantic.ply \
    --info    /home/elicer/room_0/habitat/info_semantic.json \
    --relabel_root ~/relabel_replica_room0_v2 \
    --out output/relabel_eval.csv

옵션:
  --gt_samples 500000   GT 표면 샘플 수(많을수록 정밀)
  --icp                 relabel 전체 점을 GT scene 에 글로벌 ICP 정렬(잔차 클 때만)
  --min_pts 20          이보다 점이 적은 인스턴스는 'fragment?' 로 표시
  --struct "wall,floor,ceiling,window,blinds,door,vent,wall-plug,switch,pillar,undefined,beam,column"

Deps: numpy, scipy, trimesh, plyfile  (eval_object_mesh.py 와 동일).
"""
import argparse, csv, glob, json, os
import numpy as np, trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree


# ── GT semantic mesh 로딩 (eval_object_mesh.py 와 동일 로직) ──
def load_semantic_mesh(path):
    ply = PlyData.read(path); v = ply["vertex"]
    vertices = np.column_stack([np.asarray(v["x"]), np.asarray(v["y"]),
                                np.asarray(v["z"])]).astype(np.float64)
    fe = ply["face"]; names = fe.data.dtype.names
    idx_key = "vertex_indices" if "vertex_indices" in names else "vertex_index"
    if "object_id" not in names:
        raise ValueError(f"{path}: no per-face object_id (face props: {names})")
    raw = fe.data[idx_key]; oid = np.asarray(fe.data["object_id"]).astype(np.int64)
    lens = np.fromiter((len(f) for f in raw), dtype=np.int64, count=len(raw))
    if (lens == 3).all():
        tris = np.vstack(raw).astype(np.int64); tri_ids = oid
    elif (lens == 4).all():
        q = np.vstack(raw).astype(np.int64)
        tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], 0)
        tri_ids = np.concatenate([oid, oid], 0)
    else:
        tl, il = [], []
        for f, o in zip(raw, oid):
            f = np.asarray(f, np.int64)
            for k in range(1, len(f) - 1):
                tl.append((f[0], f[k], f[k + 1])); il.append(o)
        tris = np.asarray(tl, np.int64); tri_ids = np.asarray(il, np.int64)
    return vertices, tris, tri_ids


def load_class_names(info_json):
    if not info_json or not os.path.isfile(info_json):
        return {}
    info = json.load(open(info_json)); mp = {}
    for obj in info.get("objects", []):
        oid = obj.get("id"); cls = obj.get("class_name", obj.get("class", ""))
        if oid is not None: mp[int(oid)] = str(cls)
    return mp


def read_ply_xyz(path):
    if not os.path.isfile(path): return np.zeros((0, 3), np.float64)
    p = PlyData.read(path); v = p["vertex"]
    return np.column_stack([np.asarray(v["x"]), np.asarray(v["y"]),
                            np.asarray(v["z"])]).astype(np.float64)


def icp_align(src, tree, dst, iters=40, tol=1e-7):
    T = np.eye(4); s = src.copy(); prev = np.inf
    for _ in range(iters):
        d, idx = tree.query(s, k=1); err = float(d.mean())
        if abs(prev - err) < tol: break
        prev = err; t_ = dst[idx]
        ms, mt = s.mean(0), t_.mean(0)
        H = (s - ms).T @ (t_ - mt); U, _, Vt = np.linalg.svd(H); R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
        tt = mt - R @ ms; s = s @ R.T + tt
        Ti = np.eye(4); Ti[:3, :3], Ti[:3, 3] = R, tt; T = Ti @ T
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--info", default=None)
    ap.add_argument("--relabel_root", required=True)
    ap.add_argument("--gt_samples", type=int, default=500_000)
    ap.add_argument("--purity_th", type=float, default=0.7,
                    help="이하면 mixing(섞임)으로 판정")
    ap.add_argument("--cover_min_pts", type=int, default=30,
                    help="GT 객체를 'covered'로 인정할 최소 dominant 점 수")
    ap.add_argument("--min_pts", type=int, default=20)
    ap.add_argument("--icp", action="store_true")
    ap.add_argument("--struct", default="wall,floor,ceiling,window,blinds,door,"
                    "vent,wall-plug,switch,pillar,undefined,beam,column,curtain")
    ap.add_argument("--out", default="relabel_eval.csv")
    args = ap.parse_args()
    STRUCT = {s.strip() for s in args.struct.split(",") if s.strip()}

    # GT 표면을 object_id 라벨과 함께 샘플
    V, T, TID = load_semantic_mesh(args.gt_mesh)
    names = load_class_names(args.info)
    scene = trimesh.Trimesh(vertices=V, faces=T, process=False)
    gpts, fidx = trimesh.sample.sample_surface(scene, args.gt_samples)
    gpts = np.asarray(gpts, np.float64); gids = TID[fidx]
    gtree = cKDTree(gpts)
    print(f"GT samples={len(gpts)}  GT objects={len(np.unique(TID))}")

    # 인스턴스 로딩
    dirs = sorted(glob.glob(os.path.join(args.relabel_root, "*/")),
                  key=lambda p: int(os.path.basename(p.rstrip("/")))
                  if os.path.basename(p.rstrip("/")).isdigit() else 1e9)
    inst = []
    for d in dirs:
        gid = os.path.basename(d.rstrip("/"))
        P = read_ply_xyz(os.path.join(d, "points3d.ply"))
        if len(P): inst.append((gid, P))

    # (옵션) 전체 점 글로벌 ICP — 잔차 점검
    allP = np.vstack([P for _, P in inst]) if inst else np.zeros((0, 3))
    d0, _ = gtree.query(allP, k=1)
    print(f"relabel→GT nearest: median {np.median(d0)*1000:.1f}mm  "
          f"mean {np.mean(d0)*1000:.1f}mm  (50mm 이하면 정렬 OK)")
    if args.icp:
        Tg = icp_align(allP[:: max(1, len(allP)//20000)], gtree, gpts)
        R, t = Tg[:3, :3], Tg[:3, 3]
        inst = [(g, P @ R.T + t) for g, P in inst]
        print(f"  ICP 적용: |t|={np.linalg.norm(t)*1000:.1f}mm")

    # 인스턴스별 purity
    rows = []; cover = {}  # gt_id -> list of (gid, n_dom, purity)
    for gid, P in inst:
        _, idx = gtree.query(P, k=1); oids = gids[idx]
        u, c = np.unique(oids, return_counts=True); order = np.argsort(-c)
        dom = int(u[order[0]]); dom_n = int(c[order[0]]); tot = int(c.sum())
        purity = dom_n / tot
        sec = int(u[order[1]]) if len(order) > 1 else -1
        sec_n = int(c[order[1]]) if len(order) > 1 else 0
        dom_cls = names.get(dom, "?")
        sec_cls = names.get(sec, "?") if sec >= 0 else "-"
        flags = []
        if len(P) < args.min_pts: flags.append("fragment?")
        if purity < args.purity_th: flags.append("MIXED")
        if dom_cls in STRUCT: flags.append("STRUCT/spurious")
        rows.append(dict(inst=gid, n_pts=len(P), purity=round(purity, 3),
                         dom_id=dom, dom_class=dom_cls,
                         sec_id=sec, sec_class=sec_cls,
                         sec_frac=round(sec_n / tot, 3), flags="|".join(flags)))
        cover.setdefault(dom, []).append((gid, dom_n, purity))

    # coverage: 비구조물 GT 객체별
    gt_ids, gt_counts = np.unique(TID, return_counts=True)
    covered, dropped, overseg = [], [], []
    for oid, fc in zip(gt_ids, gt_counts):
        cls = names.get(int(oid), "?")
        if cls in STRUCT or fc < 10: continue
        claims = [x for x in cover.get(int(oid), []) if x[1] >= args.cover_min_pts]
        if not claims:
            dropped.append((int(oid), cls))
        else:
            covered.append((int(oid), cls, len(claims)))
            if len(claims) > 1: overseg.append((int(oid), cls, len(claims)))

    # 출력
    rows.sort(key=lambda r: r["purity"])
    print("\n=== 인스턴스 purity (낮은 순) ===")
    print(f"{'inst':>5} {'pts':>6} {'purity':>7} {'dom_class':<14} "
          f"{'sec_class':<14} {'sec%':>6}  flags")
    for r in rows:
        print(f"{r['inst']:>5} {r['n_pts']:>6} {r['purity']:>7.3f} "
              f"{r['dom_class']:<14} {r['sec_class']:<14} "
              f"{r['sec_frac']:>6.3f}  {r['flags']}")

    n_mixed = sum(1 for r in rows if "MIXED" in r["flags"])
    n_spur = sum(1 for r in rows if "STRUCT/spurious" in r["flags"])
    n_frag = sum(1 for r in rows if "fragment?" in r["flags"])
    n_gt = len(covered) + len(dropped)
    print("\n=== 요약 ===")
    print(f"인스턴스 {len(rows)}개:  MIXED(섞임) {n_mixed},  "
          f"STRUCT/spurious {n_spur},  fragment {n_frag}")
    print(f"평균 purity: {np.mean([r['purity'] for r in rows]):.3f}")
    print(f"GT 비구조물 객체 {n_gt}개 중:  covered {len(covered)},  "
          f"DROPPED {len(dropped)},  over-seg(>1 인스턴스) {len(overseg)}")
    print(f"  coverage = {len(covered)}/{n_gt} = {len(covered)/max(n_gt,1):.1%}")
    if dropped:
        print("  DROPPED 객체:", ", ".join(f"{c}(id{i})" for i, c in dropped))
    if overseg:
        print("  over-seg(무해):", ", ".join(f"{c}×{n}" for _, c, n in overseg))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV → {args.out}")
    print("판정: MIXED↑ → reid 억제 + depth-dense sig.  DROPPED↑ → vocab/concept 보강.")


if __name__ == "__main__":
    main()
