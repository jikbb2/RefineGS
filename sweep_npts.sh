#!/usr/bin/env bash
# 조건 포인트 밀도(n_points) 다객체 스윕.
#
# 왜:
#   obj10 에서 20000 → 1500 으로 줄이자 ShapeR 가 '재현' 대신 '완성'을 했다.
#   조밀한 점군(간격 3.2mm)이 강한 앵커가 되어 입력을 그대로 베끼게 만든 것으로 보인다
#   (ShapeR 는 cm 단위 SLAM 포인트로 학습됐다 — dump_shaper_points.py 참조).
#   그런데 obj6 은 20000 에서 튜닝됐고 얇은 다리가 걸려 있다. 낮추면 무너질 수 있으므로
#   한 객체로 전역 기본값을 정하지 않는다.
#
# 핵심 가설:
#   최적은 '점 개수'가 아니라 '점 간격'일 것이다. 물체 크기가 다르면 같은 n_points 가
#   다른 간격을 낳는다. 그래서 결과 표에 간격(mm)을 함께 찍고, 객체별 최적이 일정한
#   간격에 모이는지 본다. 모이면 n_points 를 크기에 비례시키는 규칙으로 바꾼다.
#
# 주의: 생성이 불안정하다. obj10 에서 [ensemble] 부호 불일치가 '물체 부피 대비 98~100%'
#   였다 — 세 샘플이 물체 전체만큼 서로 다르다. 그래서 표에 불일치율도 같이 낸다.
#   객체별 차이가 그 폭 안이면 결론을 내리지 말 것.
#
# 사용: bash sweep_npts.sh
#       GIDS="6 22" NPTS_LIST="20000 1500" bash sweep_npts.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
PRIOR=${PRIOR:-$HOME/prior}
SHAPER_DIR=${SHAPER_DIR:-$HOME/ShapeR}
LOGDIR=${LOGDIR:-${PRIOR}/logs}
GIDS=${GIDS:-"6 22 2 10"}
NPTS_LIST=${NPTS_LIST:-"20000 5000 1500"}

cd "${ROOT}" || exit 1
mkdir -p "${LOGDIR}"
STAT=${PRIOR}/_npts_stats.tsv; : > "${STAT}"

nn=$(echo ${NPTS_LIST} | wc -w); ng=$(echo ${GIDS} | wc -w)
echo "=== n_points 스윕  객체 {${GIDS}} × 값 {${NPTS_LIST}} = $((nn*ng))회 ==="
echo "각 회차: pkl → 생성(앙상블 3) → 융합 → 평가. 1~2시간 예상"
echo ""

for N in ${NPTS_LIST}; do
  echo "───────── n_points=${N} ─────────"
  NPTS="${N}" ONLY="${GIDS}" PHASE=all \
    CSV="${PRIOR}/_npts${N}.csv" FAILCSV="${PRIOR}/_npts${N}_fail.csv" \
    bash run_field_fusion_batch.sh > "${LOGDIR}/npts${N}.log" 2>&1
  # 회차별 부가 정보: 점 간격, 부호 불일치, 게이트
  for g in ${GIDS}; do
    sp=$(python dump_shaper_points.py "${SHAPER_DIR}/data/obj${g}.pkl" 2>/dev/null \
         | sed -n 's/.*중앙값 \([0-9.]*\)mm.*90%.*/\1/p' | head -1)
    dis=$(grep -h '^\s*\[ensemble\]' "${LOGDIR}/field_${g}.log" 2>/dev/null \
          | sed -n 's/.*부피 대비 \([0-9]*\)%.*/\1/p' | tail -1)
    gate=$(grep -h '생성 표면 중 unknown' "${LOGDIR}/fuse_${g}.log" 2>/dev/null \
           | sed -n 's/.*비율 \([0-9.]*\)%.*/\1/p' | tail -1)
    printf "%s\t%s\t%s\t%s\t%s\n" "${N}" "${g}" "${sp:-nan}" "${dis:-nan}" "${gate:-nan}" >> "${STAT}"
  done
  grep -h "^  obj\|성공\|실패" "${LOGDIR}/npts${N}.log" | tail -8
  echo ""
done

python - "${PRIOR}" "${STAT}" "${NPTS_LIST}" <<'PY'
import csv, os, re, sys
prior, statp, nlist = sys.argv[1], sys.argv[2], sys.argv[3].split()
stat = {}
for l in open(statp):
    n, g, sp, dis, gate = l.rstrip("\n").split("\t")
    stat[(n, g)] = (sp, dis, gate)

data = {}
for n in nlist:
    p = os.path.join(prior, f"_npts{n}.csv")
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p)):
        g = re.sub(r"\D", "", r.get("tag", ""))
        k = "A" if r["mesh"].startswith("fuse_post") else "B"
        data.setdefault((n, g), {})[k] = r

gids = sorted({g for _, g in data}, key=int)
NOISE = 0.0066      # unseen F@2 의 2σ (평가 샘플링 + 융합)
print("\n=== 객체별 결과 (A=baseline, B=우리) ===")
print(f"{'obj':>5}{'n_pts':>8}{'간격mm':>8}{'불일치%':>8}{'gate%':>7}"
      f"{'ΔunsF2':>9}{'ΔseenF1':>9}{'Δfree%':>9}")
for g in gids:
    for n in nlist:
        d = data.get((n, g))
        if not d or "A" not in d or "B" not in d:
            continue
        sp, dis, gate = stat.get((n, g), ("nan",) * 3)
        f = lambda k, s: float(d[s][k])
        print(f"{g:>5}{n:>8}{sp:>8}{dis:>8}{gate:>7}"
              f"{f('unseen_F2.0','B')-f('unseen_F2.0','A'):>+9.3f}"
              f"{f('seen_F1.0','B')-f('seen_F1.0','A'):>+9.3f}"
              f"{f('free_pct','B')-f('free_pct','A'):>+9.2f}")
    print()

print("읽는 법")
print(f"  · ΔunsF2 는 {NOISE:.4f}(2σ) 보다 큰 차이만 해석한다")
print("  · gate% 가 임계(10%) 미만이면 prior 가 차단되어 Δ=0 (passthrough)")
print("  · 불일치% 는 앙상블 세 샘플의 부호가 갈리는 부피(물체 부피 대비).")
print("    100% 에 가까우면 생성이 불안정하므로 그 행의 차이는 시드 운일 수 있다")
print("  · 객체별 최적이 비슷한 '간격'에 모이면 n_points 를 크기 비례로 바꾼다")
PY
