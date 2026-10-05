#!/usr/bin/env bash
# RefineGS 레포 정리 — git mv 기반, 비파괴적.
#   미리보기(기본):  bash reorg.sh
#   실제 적용:       DRY_RUN=0 bash reorg.sh
#
# - 추적 파일은 git mv(히스토리 보존), 미추적 파일은 일반 mv.
# - 없는 파일은 건너뜀(skip). 폐기/대체된 코드는 삭제하지 않고 archive/로 이동.
# - 마지막에 run_full_pipeline.sh 안의 스크립트 경로(SR/AM/AC)를 새 위치로 갱신.
set -uo pipefail
DRY_RUN=${DRY_RUN:-1}

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "git 레포 루트에서 실행하세요."; exit 1; }
cd "$ROOT"
echo "repo: $ROOT   DRY_RUN=$DRY_RUN"; echo

run(){ if [ "$DRY_RUN" = "1" ]; then printf 'DRY:'; printf ' %q' "$@"; echo; else "$@"; fi; }

mv1(){  # mv1 <src> <dest_dir>
  local src="$1" dst="$2"
  [ -e "$src" ] || { echo "  skip(없음): $src"; return; }
  run mkdir -p "$dst"
  if git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
    run git mv -k "$src" "$dst/"
  else
    run mv "$src" "$dst/"
  fi
}

echo "[1] 디렉토리 생성"
for d in refinegs/relabel refinegs/amodal refinegs/refine refinegs/assemble \
         scripts docs archive envs configs third_party/patches; do run mkdir -p "$d"; done

echo "[2] 최종 파이프라인 코드 이동 (keep)"
mv1 sam3_relabel_video.py        refinegs/relabel
mv1 amodal_mask.py               refinegs/amodal
mv1 amodal_complete_general.py   refinegs/amodal
mv1 register_generated_to_recon.py refinegs/refine
mv1 fuse_carve.py                refinegs/refine
mv1 obs_consistency_report.py    refinegs/refine
mv1 scene_assemble.py            refinegs/assemble
mv1 run_full_pipeline.sh         scripts

echo "[3] 대체/폐기된 실험 코드 → archive/ (삭제 아님)"
for f in sam3_relabel.py amodal_complete.py fuse_generated_recon.py \
         diag_registration.py sam3_video_probe.py axis2_select_and_eval.py; do
  mv1 "$f" archive
done

echo "[4] 설계 문서 → docs/"
for f in RefineGS_Problem_Redefinition.md novelty_landscape_comparison.md \
         axis2_part_aware_design.md per_object_recon_quality_plan.md \
         RefineGS_status_and_testbed_redesign.md RefineGS_stage4_handoff.md; do
  mv1 "$f" docs
done

echo "[5] 드라이버 내부 스크립트 경로 갱신 (SR/AM/AC)"
DRV=scripts/run_full_pipeline.sh
if [ "$DRY_RUN" = "1" ]; then DRV=run_full_pipeline.sh; fi   # dry-run 시엔 아직 원위치
if [ -f "$DRV" ]; then
  run sed -i 's#SR=sam3_relabel_video.py#SR=refinegs/relabel/sam3_relabel_video.py#'        "$DRV"
  run sed -i 's#AM=amodal_mask.py#AM=refinegs/amodal/amodal_mask.py#'                         "$DRV"
  run sed -i 's#AC=amodal_complete_general.py#AC=refinegs/amodal/amodal_complete_general.py#' "$DRV"
else
  echo "  (드라이버 파일을 찾지 못함 — 경로 갱신 생략)"
fi

echo
echo "완료 (DRY_RUN=$DRY_RUN)."
if [ "$DRY_RUN" = "1" ]; then
  echo "→ 위 계획 확인 후 실제 적용:  DRY_RUN=0 bash reorg.sh"
else
  cat <<'EOF'
→ 다음 단계:
   1) cp .gitignore <repo>/.gitignore   (이미 추적 중인 대용량은: git rm -r --cached data output)
   2) 외부 의존성 패치 저장 (third_party/patches/ 참고 — README 의 'External dependencies')
   3) 실행 확인:  conda activate split_and_splat && SCENE=replica_room0_v2 bash scripts/run_full_pipeline.sh recon
   4) git add -A && git commit -m "Reorganize repo: axis-based layout, archive superseded code, docs/"
EOF
fi
