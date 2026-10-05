#!/usr/bin/env bash
# 중간생성물 정리 — 단계별. 기본은 미리보기(DRY=1), 실제 삭제는 DRY=0.
#
#   bash clean_outputs.sh recon                  # recon 단계 산출물 (relabel 유지)
#   bash clean_outputs.sh amodal3r               # Amodal3R(gen/register) 산출물
#   bash clean_outputs.sh all                    # recon + amodal3r (relabel 유지)
#   bash clean_outputs.sh all --with-relabel     # relabel 까지 (sam3 재실행 필요!)
#
#   DRY=0 bash clean_outputs.sh all              # 실제 삭제
set -uo pipefail
SCOPE=${1:-help}
WITH_RELABEL=0; [ "${2:-}" = "--with-relabel" ] && WITH_RELABEL=1
SCENE=${SCENE:-replica_room0_v2}
DRY=${DRY:-1}
ROOT=/home/elicer/RefineGS; A3R=${A3R:-$HOME/Amodal3R}
cd "$ROOT"

rmp () {   # rmp <path-or-glob> ...
  for p in "$@"; do
    for x in $p; do
      [ -e "$x" ] || continue
      if [ "$DRY" = 1 ]; then echo "  DRY  rm -rf $x"; else rm -rf "$x"; echo "  removed $x"; fi
    done
  done
}
rmfind () {  # rmfind <dir> <find-args...>
  local d="$1"; shift
  [ -d "$d" ] || return 0
  while IFS= read -r x; do
    if [ "$DRY" = 1 ]; then echo "  DRY  rm $x"; else rm -rf "$x"; echo "  removed $x"; fi
  done < <(find "$d" "$@" 2>/dev/null)
}

clean_recon () {
  echo "[recon 단계] per-object 폴더 / 학습출력 / amodal / 단일테스트"
  rmp "data/${SCENE}/masks" "data/${SCENE}/discard"
  rmp "output/${SCENE}/refinegs_full"
  rmp "${HOME}/amodal_${SCENE}"
  rmp "output/${SCENE}/iso0_depth" "output/${SCENE}/isolation_test" "output/${SCENE}/refinegs_27"
}

clean_amodal3r () {
  echo "[Amodal3R 단계] 입력(G1) / 생성·정합·정리(G2,R1) / scene 합성 / 객체폴더 복사본"
  rmp "${A3R}/input" "${A3R}/poc_output"
  rmp "output/${SCENE}/scene_genclean_mesh.ply" "output/${SCENE}/scene_*.ply"
  rmfind "output/${SCENE}/refinegs_full" -name "fuse_genclean.ply"
  rmfind "data/${SCENE}/masks" -type d -name "sil_amodal"
}

clean_relabel () {
  echo "[relabel] sam3 출력 (재생성하려면 sam3 env로 relabel 다시 실행해야 함!)"
  rmp "${HOME}/relabel_${SCENE}"
}

case "$SCOPE" in
  recon)     clean_recon ;;
  amodal3r)  clean_amodal3r ;;
  all)       clean_recon; clean_amodal3r; [ "$WITH_RELABEL" = 1 ] && clean_relabel ;;
  *) echo "Usage: [DRY=0] bash clean_outputs.sh {recon|amodal3r|all} [--with-relabel]"; exit 0 ;;
esac

echo
[ "$DRY" = 1 ] && echo "※ 미리보기였습니다. 실제 삭제: DRY=0 bash clean_outputs.sh $SCOPE ${2:-}" \
              || echo "삭제 완료 (SCENE=${SCENE})."
echo "보존: data/${SCENE}/{images,sparse}, ${HOME}/relabel_${SCENE}$([ "$WITH_RELABEL" = 1 ] && echo ' (이번엔 삭제됨)'), 코드 패치"
