#!/usr/bin/env bash
# 융합 단계(prior 차단) 스윕 — seen recall 손실의 원인을 귀속시킨다.
#
# 지금까지의 사실:
#   3원 비교(21객체 중앙값)
#     A baseline  seen F@1 0.917 / uns F@2 0.132 / free 3.81%
#     A′ 융합만   seen F@1 0.821 / uns F@2 0.075 / free 0.43%
#     B +prior    seen F@1 0.870 / uns F@2 0.109 / free 1.56%
#   손실의 성격: A→A′ 에서 seen accuracy 는 좋아지고(5.055→4.857) F@1 은
#   떨어진다(0.924→0.856). 정확도↑ + F↓ = recall 손실. 표면을 놓치고 있다.
#
#   [기각] free_min_views 2/4/8 은 소수 셋째 자리까지 동일 — carve 임계는 원인이 아니다.
#     뷰가 200장이라 진짜 빈 공간은 어차피 다수가 동의한다
#     (로그 증거: '합의 2뷰 24.7% / 1뷰라도 24.7%').
#
#   [검증 중] 관측 신뢰도 가중. cos/erode 로 관측을 버리면 그 복셀은 Wo=0 이 되고,
#     prior 가 차단된 상태에서는 채울 것이 없어 구멍이 된다. B 에서 prior 가
#     seen F@1 을 0.821→0.870 으로 되돌린 것도 이 구멍을 메운 것으로 설명된다.
#     ⇒ 그렇다면 A′ 는 공정한 ablation 팔이 아니다(관측 폐기는 prior 가 메운다는
#       전제 위의 설계). 그 경우 논문에는 A vs B 만 싣고 A′ 는 진단용으로만 쓴다.
#
# 왜 여러 객체인가:
#   현재 값은 뷰 205장짜리 큰 객체 obj6 하나로 정했고 일반화되지 않았다.
#   크기·뷰수가 다른 5개의 중앙값으로 고른다.
#
# 사용: bash sweep_carve.sh                    # 관측가중 구성요소별 해제(기본)
#       AXIS=carve bash sweep_carve.sh         # [기각됨] 재현용
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
AXIS=${AXIS:-obs}
case "${AXIS}" in
  # [기각됨] free_min_views 2/4/8 은 소수 셋째 자리까지 동일했다. 뷰가 200장이라
  #   진짜 빈 공간은 어차피 다수가 동의한다(로그: '합의 2뷰 24.7% / 1뷰라도 24.7%').
  carve) VALS=${VALS:-"2 4 8"} ;;
  # 관측 신뢰도 가중을 구성요소별로 끈다. recall 손실의 귀속이 목적.
  #   full = 현재 설정 / nocos = cos 게이트만 해제 / noerode = 침식만 해제 / none = 전부 해제
  #   none 에서 seen F@1 이 baseline 으로 돌아오면 원인이 확정된다.
  obs)   VALS=${VALS:-"full nocos noerode none"} ;;
  *) echo "AXIS 는 obs|carve"; exit 1 ;;
esac

arm_flags () {            # 팔 이름 → sdf_distill_depth.py 플래그
  case "$1" in
    full)    echo "" ;;                                   # 기본값 = erode 2, cos_min 0.2, cos 가중
    nocos)   echo "--obs_cos_min 0" ;;
    noerode) echo "--obs_erode 0" ;;
    none)    echo "--obs_cos_min 0 --obs_erode 0 --no_obs_cos_weight" ;;
    *)       echo "--free_min_views $1" ;;                # carve 축
  esac
}

LOGDIR=${LOGDIR:-${PRIOR}/logs}
CSV=${CSV:-${PRIOR}/_sweep_${AXIS}.csv}
mkdir -p "${LOGDIR}"; rm -f "${CSV}"
cd "${ROOT}" || exit 1

echo "=== 융합 단계 스윕 (prior 차단)  축=${AXIS}  팔: ${VALS} ==="
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
      --min_unknown_frac 1.1 $(arm_flags "${V}") \
      --out "${OUTD}/sw_${AXIS}${V}.ply" \
      > "${LOGDIR}/sw_${NAME}.log" 2>&1 || { echo "융합 실패"; tail -5 "${LOGDIR}/sw_${NAME}.log"; continue; }
    # 관측복셀 비율이 팔마다 얼마나 달라지는지 — recall 손실의 직접 증거
    grep -h "^\[관측신뢰도\]\|^\[grid-fuse\] 관측복셀" "${LOGDIR}/sw_${NAME}.log" \
      | tr '\n' ' ' | sed 's/^/        /'; echo
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
vals = sorted(by, key=lambda x: (0,float(x)) if x.replace(".","").isdigit() else (1,0))
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
