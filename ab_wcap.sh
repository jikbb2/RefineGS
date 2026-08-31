#!/usr/bin/env bash
# grid_wcap A/B — 앙상블 prior(obj6_best.npz)를 고정하고 wcap 만 바꾼다.
#
#   wcap 은 '몇 뷰쯤 관측되면 관측 TSDF 를 100% 신뢰할 것인가'다.
#   낮추면 관측 권한이 커져 seen recall↑ / precision↓, free 위반↑.
#   cos 가중을 켜면서 Wo(관측 가중합)가 줄었으므로 재조정이 필요했다.
#
#   기준선: ab_ensemble.sh 의 ens 팔 = 같은 npz + wcap 5.0
#           seen F@1cm 0.9213 (P 0.9521 / R 0.8925), unseen P@2cm 0.6872, free 2.15%
#
# 사용: bash ab_wcap.sh            (3.0 과 4.0 을 본다)
#       WCAPS="2 3 4" bash ab_wcap.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
GID=${GID:-6}
ITER=${ITER:-7000}
PRIOR=${PRIOR:-$HOME/prior}
NPZ=${NPZ:-${PRIOR}/obj${GID}_best.npz}      # ← 앙상블 산출물(field_std 보유)
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS_DIR=${STEMS_DIR:-$HOME/See3D/dataset/stage6/clean_stems}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
MATCH_MIN_SHARE=${MATCH_MIN_SHARE:-0.03}
WCAPS=${WCAPS:-"3 4"}

MDIR=${OUT}/${GID}; OUTD=${MDIR}/train/ours_${ITER}
STEMS=${STEMS_DIR}/${GID}.txt
LOGDIR=${LOGDIR:-${PRIOR}/logs}
CSV=${CSV:-${PRIOR}/_ab_wcap_obj${GID}.csv}
mkdir -p "${LOGDIR}"; rm -f "${CSV}"

cd "${ROOT}" || exit 1
[ -f "${NPZ}" ] || { echo "prior 없음: ${NPZ}"; exit 1; }
python - "${NPZ}" <<'PY'
import numpy as np, sys
z = np.load(sys.argv[1])
print(f"[prior] {sys.argv[1].split('/')[-1]}  키={list(z.files)}")
print("  field_std %s → %s" % ("있음" if "field_std" in z.files else "없음",
      "앙상블 산출물" if "field_std" in z.files else "⚠ 단일 샘플 — 앙상블 A/B 와 조건이 다르다"))
PY
echo "=== grid_wcap A/B  obj${GID}  wcaps=${WCAPS} ==="

for W in ${WCAPS}; do
  NAME="wcap${W}"
  echo "  [${NAME}]  (tail -f ${LOGDIR}/ab_${NAME}.log)"
  python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
    --prior_field "${NPZ}" --gt_depth_dir "${GTD}" \
    --grid_wcap "${W}" \
    --out "${OUTD}/ab_${NAME}.ply" \
    > "${LOGDIR}/ab_${NAME}.log" 2>&1 \
    || { echo "    융합 실패"; tail -20 "${LOGDIR}/ab_${NAME}.log"; continue; }
  grep -h "grid_wcap\|^\[관측신뢰도\]\|^\[grid-fuse\] 관측복셀" \
    "${LOGDIR}/ab_${NAME}.log" | sed 's/^/      /'

  python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
    --recon "${OUTD}/fuse_post.ply" --recon2 "${OUTD}/ab_${NAME}_post.ply" \
    --colmap "${COLMAP}" --gid "${GID}" \
    --masks_root "${MASKS}" --use_mask \
    ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
    --match_min_share "${MATCH_MIN_SHARE}" \
    --tag "${NAME}" --csv_all --csv "${CSV}" \
    > "${LOGDIR}/ab_eval_${NAME}.log" 2>&1 \
    || { echo "    평가 실패"; tail -20 "${LOGDIR}/ab_eval_${NAME}.log"; }
done

echo ""
python - "${CSV}" <<'PY'
import csv, sys, os
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r.get("mesh", "").startswith("ab_")] if os.path.exists(sys.argv[1]) else []
if not rows: sys.exit("CSV 가 비었습니다 — 로그를 확인하세요")
# ab_ensemble.sh 의 ens 팔(wcap 5.0, 같은 npz)을 기준선으로 함께 놓는다
base = {"tag": "wcap5(기준)", "seen_acc": 4.3083, "seen_F1.0": 0.9213,
        "seen_P1.0": 0.9521, "seen_R1.0": 0.8925, "unseen_acc": 23.2374,
        "unseen_P2.0": 0.6872, "unseen_R2.0": 0.6833, "free_pct": 2.1490}
rows = [base] + rows
cols = [("seen acc(mm)", "seen_acc", "↓"), ("seen F@1cm", "seen_F1.0", "↑"),
        ("seen P@1cm", "seen_P1.0", "↑"), ("seen R@1cm", "seen_R1.0", "↑"),
        ("unseen acc(mm)", "unseen_acc", "↓"), ("unseen P@2cm", "unseen_P2.0", "↑"),
        ("unseen R@2cm", "unseen_R2.0", "↑"), ("free 위반(%)", "free_pct", "↓")]
w = max(len(c[0]) for c in cols) + 2
print(" " * w + "".join(f"{r['tag']:>13}" for r in rows))
for nm, k, d in cols:
    print(f"{nm:<{w}}" + "".join(f"{float(r.get(k,'nan')):>13.4g}" for r in rows) + f"  {d}")
PY
