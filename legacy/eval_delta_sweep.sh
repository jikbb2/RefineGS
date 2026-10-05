#!/usr/bin/env bash
# shell_delta 스윕 결과를 한 표로 비교 (seen/unseen 분리 지표 → CSV)
#   bash eval_delta_sweep.sh
# 기대: δ↓ → unseen P@2cm ↑ (과생성 감소), 단 얇은 구조가 사라지면 unseen R@2cm ↓
#       → unseen F@2cm 이 최대인 δ 를 채택하되 seen 지표가 흔들리지 않는지 확인.
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
GID=${GID:-1}
ITER=${ITER:-7000}
SCENE=${SCENE:-replica_room0_v2}
OUTD=${ROOT}/output/${SCENE}/refinegs_full/${GID}/train/ours_${ITER}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
STEMS=${STEMS:-$HOME/See3D/dataset/stage6/clean_stems/${GID}.txt}
LABELS=${LABELS:-9,70,18,71,8}          # ※ obj1 기준. 다른 객체는 auto-match 로그 보고 수정
CSV=${CSV:-${OUTD}/_sweep_delta.csv}
DELTAS=${DELTAS:-"0.006 0.010 0.016 0.024"}

cd "${ROOT}"
rm -f "${CSV}"

common=(--gt_mesh "${GT_MESH}" --gt_labels "${LABELS}"
        --recon "${OUTD}/fuse_post.ply"
        --colmap data/${SCENE}/sparse/0 --gid ${GID}
        --masks_root data/${SCENE}/masks --use_mask
        --csv "${CSV}")
[ -f "${STEMS}" ] && common+=(--stems "${STEMS}")

# baseline(A) 1회만 기록
echo "=== baseline (fuse_post) ==="
python eval_seen_unseen.py "${common[@]}" --tag baseline --csv_all \
  > "${OUTD}/_sweep_baseline.log" 2>&1 || { echo "baseline 평가 실패"; exit 1; }
head -1 "${CSV}" >/dev/null

for D in ${DELTAS}; do
  M=${OUTD}/fused_d${D}.ply
  [ -f "${M}" ] || { echo "[skip] ${M} 없음"; continue; }
  echo "=== delta ${D} ==="
  python eval_seen_unseen.py "${common[@]}" --recon2 "${M}" --tag "d${D}" \
    > "${OUTD}/_sweep_d${D}.log" 2>&1 \
    || { echo "  평가 실패 ${D}"; continue; }
  tail -6 "${OUTD}/_sweep_d${D}.log" | sed 's/^/    /'
done

echo ""
echo "=== 요약 (CSV: ${CSV}) ==="
python - "${CSV}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cols = [("tag","tag"), ("seen_acc","seenAcc"), ("seen_F1.0","seenF1"),
        ("unseen_comp_med","unsCompMed"), ("unseen_P2.0","unsP2"),
        ("unseen_R2.0","unsR2"), ("unseen_F2.0","unsF2"), ("free_pct","free%")]
print("  " + "  ".join(f"{h:>10}" for _, h in cols))
for r in rows:
    vals = []
    for k, _ in cols:
        v = r.get(k, "")
        try: vals.append(f"{float(v):10.4f}")
        except ValueError: vals.append(f"{v:>10}")
    print("  " + "  ".join(vals))
best = max((r for r in rows if r["tag"] != "baseline"),
           key=lambda r: float(r["unseen_F2.0"]), default=None)
if best:
    print(f"\n  → unseen F@2cm 최대: {best['tag']} "
          f"(F {float(best['unseen_F2.0']):.4f}, P {float(best['unseen_P2.0']):.3f}, "
          f"R {float(best['unseen_R2.0']):.3f}, seen acc {float(best['seen_acc']):.2f}mm)")
PY
