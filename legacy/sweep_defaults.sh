#!/usr/bin/env bash
# obj6 단독으로 정해진 기본값들을 5객체 프로토콜로 재확인한다.
#
# 왜:
#   관측 신뢰도 가중은 obj6 에서 좋아 보였지만 5객체에서 순손실이었다
#   (seen F@1cm 0.923 → 0.959, 끄는 쪽이 이김). 같은 방식으로 정해진 값이
#   아직 여럿 남아 있고, 배치 한 사이클이 비싸므로 먼저 여기서 거른다.
#
# 현재 기본값의 기준선 (5객체 중앙값, prior ON, 관측가중 off — 이미 측정됨):
#     seen acc 4.979mm / seen F@1cm 0.959 / uns comp 23.18mm
#     uns F@2cm 0.371  / free 위반 5.80%     (baseline: 5.055 / 0.924 / 125.5 / 0.214 / 3.81)
#   각 축에 현재 값을 포함시켰으므로, 그 열이 위 숫자를 재현하지 못하면
#   무언가 달라진 것이다 — 결과를 해석하기 전에 그것부터 확인할 것.
#
# 노이즈 폭(실측, 두 원인):
#   평가 샘플링  seen F@1 ±0.0001, seen acc ±0.022mm, uns F@2 ±0.0033, free ±0.085%p
#   융합 재현성  uns F@2 ±0.0002, free ±0.1%p  (메쉬는 Chamfer 0.05mm 다르지만
#                지표는 20만 점 평균이라 상쇄된다)
#   → 해석 임계는 평가 샘플링이 지배. uns F@2 는 0.007(2σ) 미만 차이를 해석하지 말 것.
#
# 축:
#   wcap        [종결] 8 채택, 기본값 반영 완료.
#               5객체(3/5/8): 5→8 이 seen acc -0.49mm, free -0.97%p 로 크게 이김
#               3객체(8/16/32): 8 에서 평탄. 32 까지 가면 seen acc 만 0.12mm 얻고
#                 uns comp +2.6mm, free +0.33%p 를 잃는다.
#               alpha=clip(Wo/wcap,0,1) 이고 뷰가 200장이라 잘 관측된 복셀의 Wo 는
#               수십~수백이다. wcap 을 올리면 '몇 뷰만 관측된 경계 밴드'만 prior 로
#               넘어간다. 관측가중은 그 밴드를 삭제했고(손해), wcap 은 위임한다(이득).
#   gate        prior 적용 게이트.       obj16/obj8 을 막았는데 그 판단이 옳았는지 미확인
#   carveviews  prior 할루시네이션 carve. free 위반(유일한 약축)의 직접 레버
#   connect     연결성분 필터.           분리 부품이 있는 객체에서 손해일 수 있음
#   obs         [종결] 관측가중 — 기록용. 결론: 전부 off
#
# 비용 (실측, 융합 1회):
#   obj2 240s / obj6 550s / obj22 720s / obj16 1395s / obj1 1700s
#   5객체 = 값당 약 77분. 3값이면 230분.
#
#   ⚠ prior 관련 축에서는 obj16 을 빼는 것이 낫다 — 게이트가 0.0% 로 prior 를 항상
#     차단해 모든 팔에서 같은 값이 나온다(차이에 기여 0인데 값당 23분을 쓴다).
#     GIDS="6 22 2" 면 값당 25분으로 3배 빠르고 신호는 유지된다.
#
# 사용: AXIS=wcap VALS="8 16 32" GIDS="6 22 2" bash sweep_defaults.sh
#       AXIS=gate bash sweep_defaults.sh      # 게이트 축은 obj16 을 포함해야 의미 있음
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

# 큰 것(6,1) · 중간(22,2) · 작고 평평한 것(16)
GIDS=${GIDS:-"6 1 22 2 16"}
AXIS=${AXIS:-wcap}
case "${AXIS}" in
  wcap)       VALS=${VALS:-"8 16 32"} ;;        # 현재 8 (종결)
  gate)       VALS=${VALS:-"0 0.2 0.4"} ;;      # 현재 0.2
  carveviews) VALS=${VALS:-"40 150 300"} ;;     # 현재 150
  connect)    VALS=${VALS:-"on off"} ;;         # 현재 on
  obs)        VALS=${VALS:-"full noerode none"} ;;  # 종결: none
  *) echo "AXIS 는 wcap|gate|carveviews|connect|obs"; exit 1 ;;
esac

arm_flags () {
  case "${AXIS}" in
    wcap)       echo "--grid_wcap $1" ;;
    gate)       echo "--min_unknown_frac $1" ;;
    carveviews) echo "--prior_carve_views $1" ;;
    connect)    [ "$1" = "on" ] && echo "" || echo "--no_keep_connected" ;;
    obs)        case "$1" in
                  full)    echo "--obs_erode 2 --obs_cos_min 0.2 --obs_cos_weight" ;;
                  noerode) echo "--obs_cos_min 0.2 --obs_cos_weight" ;;
                  none)    echo "" ;;
                esac ;;
  esac
}

LOGDIR=${LOGDIR:-${PRIOR}/logs}
CSV=${CSV:-${PRIOR}/_sweep_${AXIS}.csv}
mkdir -p "${LOGDIR}"; rm -f "${CSV}"
cd "${ROOT}" || exit 1

# 실측 융합 시간(초)으로 예상 소요를 낸다 — 5분/회 추정은 3배 빗나갔다(75 vs 230분)
est=0
for g in ${GIDS}; do
  case "${g}" in 2) t=240;; 6) t=550;; 22) t=720;; 16) t=1395;; 1) t=1700;; *) t=700;; esac
  est=$((est+t))
done
nv=$(echo ${VALS} | wc -w); ng=$(echo ${GIDS} | wc -w)
echo "=== 기본값 재확인  축=${AXIS}  값: ${VALS} ==="
echo "객체 ${ng}개: ${GIDS}   융합 $((nv*ng))회, 예상 $((nv*est/60))분"
[ "${AXIS}" != "gate" ] && case " ${GIDS} " in *" 16 "*)
  echo "  ⚠ obj16 은 게이트가 prior 를 항상 차단해 모든 팔에서 같은 값이 나온다"
  echo "    (값당 23분을 쓰면서 차이에는 기여하지 않음). GIDS=\"6 22 2\" 권장" ;;
esac
echo ""
T0=$(date +%s)

for V in ${VALS}; do
  for g in ${GIDS}; do
    MDIR=${OUT}/${g}; OUTD=${MDIR}/train/ours_${ITER}
    NPZ=${PRIOR}/obj${g}_field.npz
    STEMS=${STEMS_DIR}/${g}.txt
    [ -f "${NPZ}" ] || { echo "  [skip ${g}] 필드 없음"; continue; }
    NAME="${AXIS}${V}_obj${g}"; t0=$(date +%s)
    echo -n "  ${NAME} ... "
    python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
      --prior_field "${NPZ}" --gt_depth_dir "${GTD}" \
      $(arm_flags "${V}") \
      --out "${OUTD}/swd_${AXIS}${V}.ply" \
      > "${LOGDIR}/swd_${NAME}.log" 2>&1 \
      || { echo "융합 실패"; tail -5 "${LOGDIR}/swd_${NAME}.log"; continue; }
    # 설정이 실제로 먹었는지 + 게이트가 prior 를 막았는지
    grep -h "^\[gate\]" "${LOGDIR}/swd_${NAME}.log" | tr '\n' ' '
    python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
      --recon "${OUTD}/fuse_post.ply" --recon2 "${OUTD}/swd_${AXIS}${V}_post.ply" \
      --colmap "${COLMAP}" --gid "${g}" --masks_root "${MASKS}" --use_mask \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      --match_min_share "${MATCH_MIN_SHARE}" --seed 0 \
      --tag "${AXIS}=${V}|obj${g}" --csv_all --csv "${CSV}" \
      > "${LOGDIR}/swd_eval_${NAME}.log" 2>&1 \
      && echo "ok ($(( $(date +%s)-t0 ))s)" || echo "평가 실패"
  done
done
echo ""
echo "총 $(( ($(date +%s)-T0)/60 ))분"

python - "${CSV}" "${AXIS}" <<'PY'
import csv, sys, os, statistics as st, collections
rows = list(csv.DictReader(open(sys.argv[1]))) if os.path.exists(sys.argv[1]) else []
if not rows: sys.exit("CSV 가 비었습니다 — 로그를 확인하세요")
by, base = collections.defaultdict(list), []
for r in rows:
    v = r["tag"].split("|")[0].split("=")[1]
    (base if r["mesh"].startswith("fuse_post") else by[v]).append(r)
ks = [("seen acc(mm)", "seen_acc", -1), ("seen F@1cm", "seen_F1.0", +1),
      ("uns comp(mm)", "unseen_comp", -1), ("uns F@2cm", "unseen_F2.0", +1),
      ("free 위반(%)", "free_pct", -1)]
# 노이즈 폭(eval_noise.sh 실측) — 이보다 작은 차이는 표시하지 않는다
NOISE = {"seen_acc": 0.022, "seen_F1.0": 0.0001, "unseen_comp": 0.26,
         "unseen_F2.0": 0.0033, "free_pct": 0.085}
vals = list(by)
w = max(len(k[0]) for k in ks) + 2
print(f"\n객체 {len(by[vals[0]])}개 중앙값  (축={sys.argv[2]}, prior ON)\n")
print(" " * w + f"{'baseline':>12}" + "".join(f"{v:>12}" for v in vals))
best = {}
for nm, k, sgn in ks:
    b = st.median(float(r[k]) for r in base)
    m = {v: st.median(float(r[k]) for r in by[v]) for v in vals}
    bv = max(m, key=lambda v: sgn * m[v]); best[nm] = bv
    spread = max(m.values()) - min(m.values())
    mark = "" if spread > NOISE[k] else "   (차이가 노이즈 이하)"
    print(f"{nm:<{w}}{b:>12.3f}" + "".join(f"{m[v]:>12.3f}" for v in vals) + mark)
print("\n축별 최선: " + ", ".join(f"{nm}→{v}" for nm, v in best.items()))
print("⚠ baseline 열은 GIDS 가 바뀌면 함께 바뀐다(다른 객체 집합의 중앙값).")
print("  실행끼리 절대값을 가로질러 비교하지 말 것 — 같은 표 안에서만 비교한다.")
print("  참고 기준선  5객체(6/1/22/2/16): baseline uns F@2 0.214 / free 3.81")
print("               3객체(6/22/2)     : baseline uns F@2 0.174 / free 4.93")
PY
