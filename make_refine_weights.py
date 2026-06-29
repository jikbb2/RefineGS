#!/usr/bin/env python3
"""RefineGS — 품질 인식 refinement routing + weight (교정된 3-routing).

신호 결합 (observed≠good 교정):
  - 관측여부 seen.npy        (render_quality_map: 학습뷰에서 보였나)
  - 복원품질 quality.npy     (render-vs-GT 오차, 0좋음~1깨짐)
  - gen-origin  id_0 태그    (assemble_gaussians)
  - opacity/scale (ply)      (floater/junk)

per-Gaussian route:
  PRUNE(3)    = 과대 scale OR 극저 opacity            → 학습 전 제거(floater/junk)
  RETRAIN(1)  = seen ∧ quality>q_thr ∧ ¬prune          → 실측뷰 joint 재학습으로 교정(생성 불요, 최대 레버)
  GENERATE(2) = ¬seen ∧ gen ∧ ¬prune                   → 미관측 gen → See3D/prior (reachable만 See3D)
  GOOD(0)     = 나머지                                  → 보존

출력:
  <out>/routes.npy(int8), <out>/refine_weight.npy(float32, 학습 loss 가중),
  <out>/point_cloud_pruned.ply(PRUNE 제거), <out>/routes_qa.ply(green/yellow/blue/red)

실행:
  python make_refine_weights.py \
    --gaussians output/replica_room0_v2/scene_b1_obj24/point_cloud/iteration_1/point_cloud.ply \
    --quality   output/replica_room0_v2/scene_b1_obj24/quality/quality.npy \
    --seen      output/replica_room0_v2/scene_b1_obj24/quality/seen.npy \
    --gen_tags 1 --q_thr 0.3 --op_low 0.1 --scale_thr 0.1 \
    --out output/replica_room0_v2/scene_b1_obj24/refine

Deps: numpy, plyfile.
"""
import argparse, os
import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", required=True)
    ap.add_argument("--quality", required=True)
    ap.add_argument("--seen", required=True)
    ap.add_argument("--gen_tags", default="2", help="gen surfel 의 id_0 태그(base⊕recon⊕gen이면 2)")
    ap.add_argument("--base_tags", default="0",
                    help="context(보존) 태그 — prune/retrain/generate 제외, 강제 GOOD. base=0")
    ap.add_argument("--q_thr", type=float, default=0.3, help="broken 판정 품질 임계")
    ap.add_argument("--op_low", type=float, default=0.1, help="이하 opacity = junk prune")
    ap.add_argument("--scale_thr", type=float, default=0.1, help="이상 scale = floater prune")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ply = PlyData.read(a.gaussians)["vertex"]
    n = len(ply.data); nm = ply.data.dtype.names
    op = sigmoid(np.asarray(ply["opacity"]).astype(np.float64))
    sc = np.exp(np.column_stack([ply["scale_0"], ply["scale_1"]]).astype(np.float64)).max(1)
    tag = np.asarray(ply["id_0"]).astype(np.int64) if "id_0" in nm else np.zeros(n, np.int64)
    gen = np.isin(tag, [int(t) for t in a.gen_tags.split(",")])
    base = np.isin(tag, [int(t) for t in a.base_tags.split(",")]) if a.base_tags.strip() else np.zeros(n, bool)

    q = np.load(a.quality).astype(np.float32)
    seen = np.load(a.seen).astype(bool)
    for arr, nmn in [(q, "quality"), (seen, "seen")]:
        if len(arr) != n:
            raise SystemExit(f"{nmn} 길이 {len(arr)} != gaussians {n} (동일 ply인지 확인)")

    # base(context)는 prune/retrain/generate 제외 — 큰 벽·바닥 surfel 오프루닝 방지
    obj = ~base
    prune = ((sc > a.scale_thr) | (op < a.op_low)) & obj
    retrain = seen & (q > a.q_thr) & (~prune) & obj
    generate = (~seen) & gen & (~prune) & obj

    routes = np.zeros(n, np.int8)         # 0 GOOD
    routes[retrain] = 1
    routes[generate] = 2
    routes[prune] = 3

    w = np.zeros(n, np.float32)
    w[retrain] = q[retrain]               # 깨진 정도만큼 재학습 가중
    w[generate] = 1.0                     # 미관측은 풀 가중(See3D/prior 대상)

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "routes.npy"), routes)
    np.save(os.path.join(a.out, "refine_weight.npy"), w)

    # pruned ply (PRUNE 제거) — 학습 init 으로 사용
    keep = routes != 3
    merged = np.asarray(ply.data)[keep]
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(
        os.path.join(a.out, "point_cloud_pruned.ply"))

    # QA ply
    xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float32)
    col = {0: (0, 180, 0), 1: (230, 200, 0), 2: (0, 120, 255), 3: (230, 0, 0)}
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dt); arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for r, c in col.items():
        sel = routes == r
        arr["red"][sel], arr["green"][sel], arr["blue"][sel] = c
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(os.path.join(a.out, "routes_qa.ply"))

    print(f"gaussians {n}")
    print(f"  GOOD     {(routes==0).mean():.3f}")
    print(f"  RETRAIN  {(routes==1).mean():.3f}  (관측-broken → 실측 재학습)")
    print(f"  GENERATE {(routes==2).mean():.3f}  (미관측 gen → See3D/prior)")
    print(f"  PRUNE    {(routes==3).mean():.3f}  (floater/junk 제거) → pruned {keep.sum()} verts")
    print(f"→ {a.out}/  routes.npy refine_weight.npy point_cloud_pruned.ply routes_qa.ply")


if __name__ == "__main__":
    main()
