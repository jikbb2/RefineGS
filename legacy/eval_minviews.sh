#!/usr/bin/env bash
# free 위반 지표가 평가 프로토콜(--min_views)에 얼마나 좌우되는지 잰다.
#
# 배경:
#   융합 내부 진단(obj22)에서 '우리가 빈 공간으로 판정한 곳으로 15mm 이상 들어간
#   표면'은 0.2% 뿐이었다. 그런데 평가는 free 위반 ~11% 라고 한다.
#   → 융합 결함이 아니라 두 가시성 기준의 불일치다.
#     우리 carve : GT depth PNG, 2뷰 합의, depth 경계 픽셀 제외
#     평가       : GT 메쉬 레이캐스팅, min_views=1 (단 1뷰의 판정으로 오류 확정)
#
# ⚠ --min_views 는 free 만 바꾸지 않는다. classify() 가 n_seen >= min_views 를
#   먼저 보므로 1뷰만 관측된 점이 seen 에서 탈락해 free/unseen 으로 재분류된다.
#   실측: baseline 의 free 가 1.20% → 4.25% 로 '올라갔다'. A 와 B 가 반대로 움직인다.
#   따라서 이건 깔끔한 격리가 아니라 '지표의 프로토콜 민감도' 측정이다.
#
# 논문에 쓸 때: 평가 기준을 우리에게 유리하게 바꾸는 것은 리뷰어가 가장 먼저
#   의심할 지점이다. 바꾼다면 (1) baseline 에도 같은 코드가 적용된다는 점,
#   (2) 200뷰 중 1뷰 판정은 포즈·depth 노이즈에 취약하다는 근거를 명시하고,
#   (3) 두 값을 모두 보고하는 것이 안전하다.
#
# 사용: bash eval_minviews.sh                          # obj22, 현재 기본값 메쉬
#       GID=6 RECON2=/tmp/diag6_post.ply bash eval_minviews.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
GID=${GID:-22}
ITER=${ITER:-7000}
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
OUTD=${OUTD:-${OUT}/${GID}/train/ours_${ITER}}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS=${STEMS:-$HOME/See3D/dataset/stage6/clean_stems/${GID}.txt}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
MATCH_MIN_SHARE=${MATCH_MIN_SHARE:-0.03}

RECON=${RECON:-${OUTD}/fuse_post.ply}
# ⚠ 기본값을 fused_field_post.ply 로 두지 않는다 — 마지막 배치가 prior 를 차단한
#   A′ 실행이라 그 파일은 stale 이고 wcap 8 / 관측가중 off 도 반영 전이다.
RECON2=${RECON2:-/tmp/diag${GID}_post.ply}
MVS=${MVS:-"1 2 3"}

cd "${ROOT}" || exit 1
for f in "${RECON}" "${RECON2}"; do
  [ -f "$f" ] || { echo "메쉬 없음: $f"; echo "  현재 기본값으로 먼저 만드세요:"; \
    echo "  python sdf_distill_depth.py -m ${OUT}/${GID} --iteration ${ITER} \\"; \
    echo "    --prior_field ~/prior/obj${GID}_field.npz --out /tmp/diag${GID}.ply"; exit 1; }
done

echo "=== free 위반의 프로토콜 민감도  obj${GID} ==="
echo "A: ${RECON}"
echo "B: ${RECON2}"
echo "   (B 의 mtime: $(date -r "${RECON2}" '+%m-%d %H:%M') — 현재 기본값으로 만든 것인지 확인)"
echo ""
printf "%-11s %10s %10s %10s %10s %10s\n" "min_views" "A free%" "B free%" "A seen%" "B seen%" "B unsF2"
for MV in ${MVS}; do
  LOG=/tmp/_mv${MV}_obj${GID}.log
  python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
    --recon "${RECON}" --recon2 "${RECON2}" \
    --colmap "${COLMAP}" --gid "${GID}" \
    --masks_root "${MASKS}" --use_mask \
    ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
    --match_min_share "${MATCH_MIN_SHARE}" --min_views "${MV}" --seed 0 \
    > "${LOG}" 2>&1 || { echo "  min_views=${MV} 평가 실패"; tail -5 "${LOG}"; continue; }
  python - "${LOG}" "${MV}" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
# '점 구성  recon: seen  89.3%  free위반   4.2%  unseen   6.4%' 를 A, B 순서로 파싱
comp = re.findall(r"seen\s+([\d.]+)%\s+free위반\s+([\d.]+)%", t)
f2 = re.findall(r"\[UNSEEN\].*?F@2\.0cm ([\d.]+)", t, re.S)
if len(comp) >= 2:
    (as_, af), (bs, bf) = comp[0], comp[1]
    print(f"{sys.argv[2]:<11} {af:>10} {bf:>10} {as_:>10} {bs:>10} {f2[1] if len(f2)>1 else '-':>10}")
PY
done
echo ""
echo "읽는 법:"
echo "  · A 와 B 의 free% 가 반대로 움직이면 min_views 는 free 만이 아니라 분류 전체를 바꾼다"
echo "  · B free% 가 min_views 1→2 에서 급감하면, 차이는 '1뷰 판정'의 노이즈 때문이다"
echo "  · 거의 안 변하면 원인은 두 depth 소스(nice-slam PNG vs habitat 메쉬)의 차이다"
