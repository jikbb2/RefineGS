#!/usr/bin/env bash
# 평가 자체의 노이즈 폭 측정 — 동일 메쉬를 시드만 바꿔 반복 평가한다.
#
# 왜 필요한가:
#   eval_seen_unseen.py 의 sample_points_uniformly 가 시드 없이 돌고 있었다.
#   wcap 스윕에서 seen F@1cm 이 3→4→5 에 대해 0.9254→0.9179→0.9213 로 비단조로
#   튀었는데, 물리적으로 단조여야 할 값이었다. 즉 설정 차이가 아니라 샘플링 차이를
#   보고 있었을 수 있다.
#
#   여기서 나오는 표준편차보다 작은 차이는 '차이가 아니다'. 설정을 고를 때,
#   논문 표에 숫자를 적을 때 이 값을 기준선으로 삼는다.
#
# 사용: MESH=.../ab_wcap5_post.ply bash eval_noise.sh
#       SEEDS="0 1 2 3 4 5 6 7" bash eval_noise.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
GID=${GID:-6}
ITER=${ITER:-7000}
PRIOR=${PRIOR:-$HOME/prior}
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
OUTD=${OUT}/${GID}/train/ours_${ITER}
MESH=${MESH:-${OUTD}/ab_ens_post.ply}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS=${STEMS:-$HOME/See3D/dataset/stage6/clean_stems/${GID}.txt}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
MATCH_MIN_SHARE=${MATCH_MIN_SHARE:-0.03}
SEEDS=${SEEDS:-"0 1 2 3 4"}

LOGDIR=${LOGDIR:-${PRIOR}/logs}
CSV=${CSV:-${PRIOR}/_eval_noise_obj${GID}.csv}
mkdir -p "${LOGDIR}"; rm -f "${CSV}"
cd "${ROOT}" || exit 1

[ -f "${MESH}" ] || { echo "메쉬 없음: ${MESH}"; exit 1; }
echo "=== 평가 노이즈 측정  obj${GID} ==="
echo "메쉬: ${MESH}"
echo "시드: ${SEEDS}   (같은 메쉬, 같은 GT, 샘플링만 다름)"
echo ""

for S in ${SEEDS}; do
  echo -n "  seed ${S} ... "
  python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
    --recon "${OUTD}/fuse_post.ply" --recon2 "${MESH}" \
    --colmap "${COLMAP}" --gid "${GID}" \
    --masks_root "${MASKS}" --use_mask \
    ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
    --match_min_share "${MATCH_MIN_SHARE}" --seed "${S}" \
    --tag "s${S}" --csv "${CSV}" \
    > "${LOGDIR}/noise_s${S}.log" 2>&1 && echo "ok" || { echo "실패"; tail -5 "${LOGDIR}/noise_s${S}.log"; }
done

echo ""
python - "${CSV}" <<'PY'
import csv, sys, os, statistics as st
rows = list(csv.DictReader(open(sys.argv[1]))) if os.path.exists(sys.argv[1]) else []
if len(rows) < 2: sys.exit("행이 부족합니다 — 로그를 확인하세요")
keys = [("seen acc(mm)","seen_acc"), ("seen F@1cm","seen_F1.0"),
        ("seen P@1cm","seen_P1.0"), ("seen R@1cm","seen_R1.0"),
        ("unseen acc(mm)","unseen_acc"), ("unseen P@2cm","unseen_P2.0"),
        ("unseen R@2cm","unseen_R2.0"), ("free 위반(%)","free_pct")]
w = max(len(k[0]) for k in keys) + 2
print(f"동일 메쉬 {len(rows)}회 재평가 (샘플링 시드만 다름)\n")
print(f"{'지표':<{w}}{'평균':>11}{'표준편차':>11}{'최소':>11}{'최대':>11}{'폭':>11}")
for nm, k in keys:
    v = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
    if len(v) < 2: continue
    sd = st.stdev(v)
    print(f"{nm:<{w}}{st.mean(v):>11.4g}{sd:>11.3g}{min(v):>11.4g}{max(v):>11.4g}{max(v)-min(v):>11.3g}")
print("\n※ 두 설정의 차이가 위 '폭'보다 작으면 그것은 설정 차이가 아니라 샘플링 차이다.")
PY
