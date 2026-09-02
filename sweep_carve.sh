#!/usr/bin/env bash
# 융합 단계(A′, prior 차단) 단독 스윕 — carve/관측가중이 표면을 깎는 문제를 잡는다.
#
# 왜 prior 를 끄고 도는가:
#   3원 비교(21객체 중앙값)에서 손실은 융합 단계에서 났다.
#     A baseline  seen F@1 0.917 / uns F@2 0.132 / free 3.81%
#     A′ 융합만   seen F@1 0.821 / uns F@2 0.075 / free 0.43%   ← 여기서 무너짐
#     B +prior    seen F@1 0.870 / uns F@2 0.109 / free 1.56%   ← prior 가 절반 회복
#   free 위반을 9배 줄이면서 진짜 표면까지 깎았다. prior 를 켠 채로 튜닝하면
#   두 효과가 또 섞이므로, 여기서는 --min_unknown_frac 1.1 로 prior 를 차단한다.
#
# 왜 여러 객체인가:
#   현재 값(free_min_views 2, obs_cos_min 0.2, obs_erode 2)은 뷰 205장짜리
#   큰 객체 obj6 하나로 정했고, 일반화되지 않았다. 크기·뷰수가 다른 5개의
#   '중앙값'으로 고른다. 한 객체에 최적인 값을 다시 뽑으면 같은 실수의 반복이다.
#
# 목표: A′ 가 baseline 의 seen F@1(0.917) 을 회복하면서 free 위반은 낮게 유지.
#
# 사용: bash sweep_carve.sh                    # carve 축
#       AXIS=cos bash sweep_carve.sh           # 관측가중 축
#       GIDS="6 1 22" bash sweep_carve.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
ITER=${ITER:-7000}
PRIOR=${PRIOR:-$HOME/prior}
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS_DIR=${STEMS_DIR:-$HOME/See3D/dataset/stage6/clean_stems}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
MATCH_MIN_SHARE=${MATCH_MIN_SHARE:-0.03}

# 큰 것(6,1)·중간(22,2)·작고 얇은 것(16) — obj8 은 관측복셀 1.5% 라 별도로 본다
GIDS=${GIDS:-"6 1 22 2 16"}
AXIS=${AXIS:-carve}
case "${AXIS}" in
  carve) VALS=${VALS:-"2 4 8"};  FLAG="--free_min_views" ;;
  cos)   VALS=${VALS:-"0.2 0.1 0.0"}; FLAG="--obs_cos_min" ;;
  erode) VALS=${VALS:-"2 1 0"};  FLAG="--obs_erode" ;;
  *) echo "AXIS 는 carve|cos|erode"; exit 1 ;;
esac

LOGDIR=${LOGDIR:-${PRIOR}/logs}
CSV=${CSV:-${PRIOR}/_sweep_${AXIS}.csv}
mkdir -p "${LOGDIR}"; rm -f "${CSV}"
cd "${ROOT}" || exit 1

echo "=== 융합 단계 스윕 (prior 차단)  축=${AXIS} ${FLAG} ∈ {${VALS}} ==="
echo "객체: ${GIDS}"
echo ""

for V in ${VALS}; do
  for g in ${GIDS}; do
    MDIR=${OUT}/${g}; OUTD=${MDIR}/train/ours_${ITER}
    NPZ=${PRIOR}/obj${g}_field.npz
    STEMS=${STEMS_DIR}/${g}.txt
    [ -f "${NPZ}" ] || { echo "  [skip ${g}] 필드 없음"; continue; }
    NAME="${AXIS}${V}_obj${g}"
    echo -n "  ${NAME} ... "
    python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
      --prior_field "${NPZ}" --gt_depth_dir "${GTD}" \
      --min_unknown_frac 1.1 ${FLAG} "${V}" \
      --out "${OUTD}/sw_${AXIS}${V}.ply" \
      > "${LOGDIR}/sw_${NAME}.log" 2>&1 || { echo "융합 실패"; tail -5 "${LOGDIR}/sw_${NAME}.log"; continue; }
    python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
      --recon "${OUTD}/fuse_post.ply" --recon2 "${OUTD}/sw_${AXIS}${V}_post.ply" \
      --colmap "${COLMAP}" --gid "${g}" --masks_root "${MASKS}" --use_mask \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      --match_min_share "${MATCH_MIN_SHARE}" --seed 0 \
      --tag "${AXIS}=${V}|obj${g}" --csv_all --csv "${CSV}" \
      > "${LOGDIR}/sw_eval_${NAME}.log" 2>&1 && echo "ok" || echo "평가 실패"
  done
done

echo ""
python - "${CSV}" <<'PY'
import csv, sys, os, statistics as st, collections
rows = list(csv.DictReader(open(sys.argv[1]))) if os.path.exists(sys.argv[1]) else []
if not rows: sys.exit("CSV 가 비었습니다")
# tag = "축=값|objN", mesh 가 sw_ 로 시작하는 행이 스윕 결과, fuse_post 행이 baseline
by = collections.defaultdict(list); base = collections.defaultdict(list)
for r in rows:
    v = r["tag"].split("|")[0].split("=")[1]
    (base if r["mesh"].startswith("fuse_post") else by)[v].append(r)
ks = [("seen acc(mm)","seen_acc"), ("seen F@1cm","seen_F1.0"),
      ("uns comp(mm)","unsseen_comp" if "unsseen_comp" in rows[0] else "unseen_comp"),
      ("uns F@2cm","unseen_F2.0"), ("free 위반(%)","free_pct")]
vals = sorted(by, key=float)
w = max(len(k[0]) for k in ks) + 2
bl = next(iter(base.values()), None)
print(f"객체 {len(next(iter(by.values())))}개 중앙값 (prior 차단)\n")
print(" "*w + f"{'baseline':>12}" + "".join(f"{v:>12}" for v in vals))
for nm, k in ks:
    b = st.median(float(r[k]) for r in bl) if bl else float("nan")
    line = f"{nm:<{w}}{b:>12.3f}"
    for v in vals:
        line += f"{st.median(float(r[k]) for r in by[v]):>12.3f}"
    print(line)
print("\n목표: seen F@1cm 이 baseline 을 회복하면서 free 위반은 baseline 보다 낮을 것")
PY
