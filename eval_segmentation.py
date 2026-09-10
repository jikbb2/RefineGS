#!/usr/bin/env python3
"""3D 인스턴스 분할 평가 — 객체별 재구성 메쉬 vs GT object_id.

왜 필요한가:
  이 프로젝트는 3D segmentation 을 기여로 내세우는데 지표가 하나도 없다.
  GT mesh_semantic.ply 에 object_id 가 있으므로 지금 데이터로 바로 계산된다.

방법 (점 기반):
  1) GT 씬 메쉬를 면적 가중으로 균등 샘플 → 점마다 GT object_id
  2) 각 GT 점에서 가장 가까운 '재구성 객체'를 찾는다(임계 --tau 안에 있을 때만)
     → 점마다 예측 인스턴스 id. 임계 밖이면 미할당.
  3) (GT id × 예측 id) 교차표로 인스턴스별 IoU 계산
  4) IoU 임계(0.25 / 0.5)에서 precision / recall / F1

  ⚠ 신뢰도 점수가 없으므로 정식 AP 는 계산하지 않는다. 모든 예측의 점수를 1 로 둔
    AP 는 그 임계에서의 precision 과 같으므로, 오해를 부르지 않게 P/R/F1 로 보고한다.

  ⚠ 한 GT 객체가 여러 재구성 인스턴스로 쪼개지는 경우(SAM3 파편)와 그 반대(병합)를
    모두 세어 출력한다 — 이 프로젝트의 주요 실패 모드라 IoU 평균만으로는 안 보인다.

사용:
  python eval_segmentation.py --gt_mesh ~/room_0/habitat/mesh_semantic.ply \
      --root ~/RefineGS/output/replica_room0_v2/refinegs_full --mesh fused_field_post.ply
  python eval_segmentation.py ... --mesh fuse_post.ply        # baseline 과 비교
"""
import argparse
import collections
import glob
import os

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

DEFAULT_EXCLUDE = "0,9,20,21,25,26,28,29,31,33"


def load_gt(path):
    """GT 메쉬 + 면별 object_id. quad 는 fan 삼각분할."""
    p = PlyData.read(os.path.expanduser(path))
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)
    fe = p["face"]
    idx = fe["vertex_indices"]
    oid = np.asarray(fe["object_id"]) if "object_id" in fe.data.dtype.names else None
    assert oid is not None, "GT 메쉬에 object_id 가 없습니다"
    T, L = [], []
    for f, o in zip(idx, oid):
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1])); L.append(o)
    return V, np.asarray(T, np.int64), np.asarray(L)


def sample_tris(V, T, L, n, seed=0):
    rng = np.random.default_rng(seed)
    e1 = V[T[:, 1]] - V[T[:, 0]]; e2 = V[T[:, 2]] - V[T[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    i = rng.choice(len(T), n, p=area / area.sum())
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    P = ((1 - r1)[:, None] * V[T[i, 0]] + (r1 * (1 - r2))[:, None] * V[T[i, 1]]
         + (r1 * r2)[:, None] * V[T[i, 2]])
    return P, L[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--root", required=True, help="객체별 모델 폴더의 상위")
    ap.add_argument("--mesh", default="fused_field_post.ply")
    ap.add_argument("--fallback", default="fuse_post.ply",
                    help="--mesh 가 없을 때 대신 쓸 파일(게이트 차단 객체 등). 빈값이면 건너뜀")
    ap.add_argument("--iter", default=7000, type=int)
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="'none' 이면 전부 포함")
    ap.add_argument("--tau", type=float, default=0.02,
                    help="GT 점을 재구성 인스턴스에 붙일 거리 임계(m). 이보다 멀면 미할당")
    ap.add_argument("--n_sample", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_gt_pts", type=int, default=200,
                    help="GT 점이 이보다 적은 인스턴스는 평가에서 제외(잡음)")
    args = ap.parse_args()

    V, T, L = load_gt(args.gt_mesh)
    G, GL = sample_tris(V, T, L, args.n_sample, args.seed)
    print(f"[GT] tri {len(T)}  object_id {len(np.unique(L))}종  → 샘플 {len(G)}점")

    ex = set() if args.exclude.strip().lower() in ("none", "") else \
        {int(x) for x in args.exclude.split(",") if x.strip()}

    # 재구성 객체들을 하나의 점군 + 인스턴스 id 로 모은다
    P, PI, used, fell = [], [], [], []
    for d in sorted(glob.glob(os.path.join(os.path.expanduser(args.root), "*"))):
        g = os.path.basename(d)
        if not g.isdigit() or int(g) in ex:
            continue
        od = os.path.join(d, "train", f"ours_{args.iter}")
        p = os.path.join(od, args.mesh)
        if not os.path.isfile(p) and args.fallback:
            p2 = os.path.join(od, args.fallback)
            if os.path.isfile(p2):
                p = p2; fell.append(g)
        if not os.path.isfile(p):
            continue
        m = o3d.io.read_triangle_mesh(p)
        if len(m.vertices) < 100:
            continue
        try:
            pc = m.sample_points_uniformly(number_of_points=20000, seed=args.seed)
        except TypeError:
            pc = m.sample_points_uniformly(number_of_points=20000)
        q = np.asarray(pc.points)
        P.append(q); PI.append(np.full(len(q), int(g))); used.append(int(g))
    assert P, f"재구성 메쉬를 찾지 못했습니다 — --root/--mesh 확인 ({args.mesh})"
    P = np.concatenate(P); PI = np.concatenate(PI)
    print(f"[recon] 인스턴스 {len(used)}개, 점 {len(P):,}"
          + (f"  (fallback: {', '.join(fell)})" if fell else ""))

    # GT 점 → 가장 가까운 재구성 인스턴스 (tau 안에서만)
    d, j = cKDTree(P).query(G, workers=-1)
    pred = np.where(d < args.tau, PI[j], -1)
    print(f"[할당] GT 점의 {100*(pred>=0).mean():.1f}% 가 {args.tau*1000:.0f}mm 안에 "
          f"재구성 인스턴스를 가짐")

    # 교차표 → IoU
    gt_ids = [g for g in np.unique(GL) if (GL == g).sum() >= args.min_gt_pts]
    inter = collections.Counter(zip(GL[pred >= 0], pred[pred >= 0]))
    gt_cnt = collections.Counter(GL.tolist())
    pr_cnt = collections.Counter(pred[pred >= 0].tolist())

    best = {}                                  # gt_id → (iou, pred_id)
    for g in gt_ids:
        cand = [(k[1], c) for k, c in inter.items() if k[0] == g]
        if not cand:
            best[g] = (0.0, None); continue
        pid, ci = max(cand, key=lambda x: x[1])
        iou = ci / (gt_cnt[g] + pr_cnt[pid] - ci)
        best[g] = (iou, pid)

    # 쪼개짐/병합 — 이 프로젝트의 주요 실패 모드
    split = collections.defaultdict(set)       # gt → 여러 pred
    merge = collections.defaultdict(set)       # pred → 여러 gt
    for (g, p_), c in inter.items():
        if c >= args.min_gt_pts:
            split[g].add(p_); merge[p_].add(g)
    n_split = sum(1 for v in split.values() if len(v) > 1)
    n_merge = sum(1 for v in merge.values() if len(v) > 1)

    print(f"\n{'GT id':>7}{'GT점':>9}{'IoU':>8}  매칭 인스턴스")
    for g in sorted(gt_ids, key=lambda x: -best[x][0]):
        iou, pid = best[g]
        extra = f"  (쪼개짐 {len(split[g])}개)" if len(split.get(g, ())) > 1 else ""
        print(f"{g:>7}{gt_cnt[g]:>9}{iou:>8.3f}  {pid if pid is not None else '-'}{extra}")

    ious = np.array([best[g][0] for g in gt_ids])
    print(f"\n=== 요약 (인스턴스 {len(gt_ids)}개, tau={args.tau*1000:.0f}mm) ===")
    print(f"  mIoU                 {ious.mean():.4f}")
    print(f"  mIoU (매칭된 것만)     {ious[ious > 0].mean() if (ious > 0).any() else 0:.4f}"
          f"  ({int((ious > 0).sum())}개)")
    for t in (0.25, 0.50):
        tp = int((ious >= t).sum())
        prec = tp / max(len(used), 1)          # 예측 인스턴스 대비
        rec = tp / max(len(gt_ids), 1)         # GT 인스턴스 대비
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  IoU≥{t:.2f}   TP {tp:>3}   P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}")
    print(f"  쪼개짐(GT 1개 → 예측 여러개)  {n_split}건")
    print(f"  병합  (예측 1개 → GT 여러개)  {n_merge}건")
    print(f"\n※ 신뢰도 점수가 없어 정식 AP 대신 P/R/F1 로 보고한다.")
    print(f"  recall 은 GT {len(gt_ids)}개 기준이므로, 재구성하지 않은 객체가 있으면 낮게 나온다.")


if __name__ == "__main__":
    main()
