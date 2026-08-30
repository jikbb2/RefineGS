#!/usr/bin/env bash
# 앙상블 A/B — prior npz 만 바꾸고 나머지는 전부 sdf_distill_depth.py 기본값.
#
# 왜 이렇게까지 하나:
#   이전에 "앙상블이 unseen precision 을 8%p 올린다"고 결론냈다가 철회했다.
#   그 비교는 npz 생성 방식과 GRID_WCAP 이 동시에 달라 교란돼 있었다.
#   여기서는 --prior_field 와 --prior_sigma_w 만 다르고 나머지는 손대지 않는다
#   (= 기본값이므로 세 팔이 자동으로 동일해진다).
#
#   ⚠ --prior_sigma_w 가 왜 팔마다 다른가: 기본값 0 이지만 field_std 가 있는
#     npz(앙상블 산출물)에서만 의미가 있다. obj6_best.npz 에만 field_std 가 있어,
#     이걸 명시하지 않으면 '앙상블 효과'와 'σ 가중 효과'가 섞인다.
#
# 세 팔:
#   single    obj6_field.npz  σ off    ← 현재 배치 설정 (기준)
#   ens       obj6_best.npz   σ off    ← 앙상블 순효과
#   ens_sig   obj6_best.npz   σ on     ← σ 가중 순효과
#
# 사용:  bash ab_ensemble.sh            (obj6)
#        GID=1 bash ab_ensemble.sh      (다른 객체)
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
GID=${GID:-6}
ITER=${ITER:-7000}
PRIOR=${PRIOR:-$HOME/prior}
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS_DIR=${STEMS_DIR:-$HOME/See3D/dataset/stage6/clean_stems}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
MATCH_MIN_SHARE=${MATCH_MIN_SHARE:-0.03}

MDIR=${OUT}/${GID}
OUTD=${MDIR}/train/ours_${ITER}
STEMS=${STEMS_DIR}/${GID}.txt
LOGDIR=${LOGDIR:-${PRIOR}/logs}
CSV=${CSV:-${PRIOR}/_ab_ensemble_obj${GID}.csv}
mkdir -p "${LOGDIR}"; rm -f "${CSV}"

cd "${ROOT}" || exit 1
echo "=== 앙상블 A/B  obj${GID} ==="
echo "융합 설정은 sdf_distill_depth.py 기본값 — 각 팔의 [config] 표로 동일함을 확인할 것"
echo ""

run_arm () {                       # $1=이름  $2=npz stem  $3=prior_sigma_w
  local NAME=$1 STEM=$2 SIGW=$3
  local NPZ=${PRIOR}/${STEM}.npz
  [ -f "${NPZ}" ] || { echo "  [skip ${NAME}] ${NPZ} 없음"; return 1; }
  echo "  [${NAME}] npz=${STEM} sigma_w=${SIGW}  (tail -f ${LOGDIR}/ab_${NAME}.log)"

  python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
    --prior_field "${NPZ}" --prior_sigma_w "${SIGW}" \
    --gt_depth_dir "${GTD}" \
    --out "${OUTD}/ab_${NAME}.ply" \
    > "${LOGDIR}/ab_${NAME}.log" 2>&1 \
    || { echo "    융합 실패"; tail -20 "${LOGDIR}/ab_${NAME}.log"; return 1; }

  # [config] 표를 그대로 뽑아 세 팔이 정말 같은 설정이었는지 눈으로 대조한다
  sed -n '/^┌─ \[config\]/,/^└/p' "${LOGDIR}/ab_${NAME}.log" | sed 's/^/      /'
  grep -h "^\[관측신뢰도\]\|^\[prior-field\] σ\|^\[gate\]" \
    "${LOGDIR}/ab_${NAME}.log" | sed 's/^/      /'

  python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
    --recon "${OUTD}/fuse_post.ply" --recon2 "${OUTD}/ab_${NAME}_post.ply" \
    --colmap "${COLMAP}" --gid "${GID}" \
    --masks_root "${MASKS}" --use_mask \
    ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
    --match_min_share "${MATCH_MIN_SHARE}" \
    --tag "${NAME}" --csv_all --csv "${CSV}" \
    > "${LOGDIR}/ab_eval_${NAME}.log" 2>&1 \
    || { echo "    평가 실패"; tail -20 "${LOGDIR}/ab_eval_${NAME}.log"; return 1; }
}

run_arm single  obj6_field 0
run_arm ens     obj6_best  0
run_arm ens_sig obj6_best  1.0

echo ""
echo "=== 결과 (${CSV}) ==="
python - "${CSV}" <<'PY'
import csv, sys, os
rows = list(csv.DictReader(open(sys.argv[1]))) if os.path.exists(sys.argv[1]) else []
if not rows:
    sys.exit("CSV 가 비었습니다 — 모든 팔이 실패했는지 로그를 확인하세요")

def g(r, *ks):
    for k in ks:
        v = r.get(k, "")
        if v not in ("", None):
            try: return float(v)
            except ValueError: pass
    return float("nan")

cols = [("seen acc(mm)",   ("seen_accuracy", "seen_acc"),     "낮을수록"),
        ("seen F@1cm",     ("seen_f1cm", "seen_f_1.0"),       "높을수록"),
        ("unseen acc(mm)", ("unseen_accuracy", "unseen_acc"), "낮을수록"),
        ("unseen P@2cm",   ("unseen_p2cm", "unseen_p_2.0"),   "높을수록"),
        ("unseen R@2cm",   ("unseen_r2cm", "unseen_r_2.0"),   "높을수록")]

tags, out = [], []
for r in rows:
    t = r.get("tag", "?")
    if t not in tags:
        tags.append(t); out.append(r)

w = max(len(c[0]) for c in cols) + 2
print(" " * w + "".join(f"{t:>12}" for t in tags))
for name, keys, dirn in cols:
    print(f"{name:<{w}}" + "".join(f"{g(r,*keys):>12.4g}" for r in out) + f"   ({dirn} 좋음)")
print("\nsingle↔ens = 앙상블 순효과 / ens↔ens_sig = σ 가중 순효과")
print("nan 이 뜨면 CSV 헤더 확인:", ", ".join(rows[0].keys()))
PY
