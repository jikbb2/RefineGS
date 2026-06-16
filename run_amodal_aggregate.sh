#!/usr/bin/env bash
# 축2+3 aggregate — 모든 per-object recon에 관측-일관 완성 적용 → recon vs completed 전체 비교.
# 단일 객체 일화가 아니라 수십 객체 평균에서 확실한 차이를 본다.
#
# 사전: split_and_splat env, /home/elicer/RefineGS, amodal_complete_general.py(tools/ 또는 루트).
# 사용:
#   bash run_amodal_aggregate.sh
#   RECON_ROOT=output/replica_room0/raw_graph_reg ITERS=7000 bash run_amodal_aggregate.sh
set -uo pipefail
SCENE=replica_room0
RECON_ROOT=${RECON_ROOT:-output/${SCENE}/raw_graph_reg}
ITERS=${ITERS:-7000}
GT_MESH=${GT_MESH:-/home/elicer/room_0/habitat/mesh_semantic.ply}
SUP=${SUP:-0.05}; DTRIM=${DTRIM:-0.1}; DEPTH=${DEPTH:-9}
SCRIPT=${SCRIPT:-amodal_complete_general.py}

echo "objects under ${RECON_ROOT}:"
ls ${RECON_ROOT} | tr '\n' ' '; echo

# 1) 각 객체에 관측-일관 완성
n=0
for D in ${RECON_ROOT}/*/; do
  ID=$(basename "$D")
  REC="${D}train/ours_${ITERS}/fuse_post.ply"
  OUT="${D}train/ours_${ITERS}/fuse_completed.ply"
  [ -f "$REC" ] || { echo "  [skip] $ID: no fuse_post"; continue; }
  [ -f "$OUT" ] && { n=$((n+1)); continue; }   # 이미 있음
  python ${SCRIPT} --recon_ply "$REC" --out_ply "$OUT" \
      --poisson_depth ${DEPTH} --density_trim ${DTRIM} --support_dist ${SUP} \
      >/dev/null 2>&1 && n=$((n+1)) || echo "  [fail] $ID"
done
echo "completed objects: $n"

# 2) batch 평가: recon vs completed (auto_match)
python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
  --recon_glob "${RECON_ROOT}/*/train/ours_${ITERS}/fuse_post.ply" \
  --label_from_path -4 --auto_match --out ${RECON_ROOT}/_eval_recon.csv
python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
  --recon_glob "${RECON_ROOT}/*/train/ours_${ITERS}/fuse_completed.ply" \
  --label_from_path -4 --auto_match --out ${RECON_ROOT}/_eval_completed.csv

# 3) aggregate 비교
python - <<EOF
import csv
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        if r.get("chamfer_l1"): d[r.get("label", r.get("recon",""))]=r
    return d
A=load("${RECON_ROOT}/_eval_recon.csv"); B=load("${RECON_ROOT}/_eval_completed.csv")
keys=sorted(set(A)&set(B))
import statistics as st
def col(d,k,c):
    import numpy as np; return [float(d[x][c]) for x in k if c in d[x]]
mr=lambda c: 1000*sum(float(A[x][c]) for x in keys)/len(keys)
mc=lambda c: 1000*sum(float(B[x][c]) for x in keys)/len(keys)
print(f"\n매칭 객체: {len(keys)}")
for c in ["chamfer_l1","accuracy","completion"]:
    r=mr(c); cc=mc(c); print(f"  {c:>12}: recon={r:7.1f}mm  completed={cc:7.1f}mm  Δ={r-cc:+6.1f}")
imp=sum(1 for x in keys if float(B[x]["completion"])<float(A[x]["completion"]))
print(f"  completion 개선된 객체: {imp}/{len(keys)}")
EOF
echo "기준: completed의 completion/chamfer 평균이 recon보다 작고, 개선 객체 비율이 높으면 확실한 차이."
