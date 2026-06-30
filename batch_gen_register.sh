#!/usr/bin/env bash
# RefineGS — 전 객체 Amodal3R gen + register 배치 (전체 씬 조립 준비).
# 객체당: prepare_amodal3r_input(crop fix+hull) → run_amodal3r_poc(3뷰) → register_robust → clean_components
# 환경 2개: amodal3r(prepare/gen), split_and_splat(register). conda activate per-step.
#
# 사용:
#   bash batch_gen_register.sh            # refinegs_fix 의 모든 gid 자동
#   bash batch_gen_register.sh "0 1 8 27" # 특정 gid만
#   FORCE=1 bash batch_gen_register.sh    # 이미 등록된 것도 재실행
#
# 결과: ~/Amodal3R/poc_output/<gid>/seed_1/mesh_registered_clean.ply (성공한 객체만)
#   → register 실패(평면/대칭 등)는 건너뜀(그 객체는 base 만으로 남음 = 정상). 마지막에 성공/실패 요약.
set -u
RG=/home/elicer/RefineGS
AM=$HOME/Amodal3R
SCENE=replica_room0_v2
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

run_in() {  # $1=env, $2=command-string
  bash -c "source $CONDA_SH && conda activate $1 && unset PYTHONPATH && export LD_LIBRARY_PATH= && $2"
}

if [ "$#" -ge 1 ]; then
  GIDS="$1"
else
  GIDS=$(ls "$RG/output/$SCENE/refinegs_fix" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
fi

ok=""; fail=""; skip=""
for gid in $GIDS; do
  RECON=$(ls $RG/output/$SCENE/refinegs_fix/$gid/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | head -1)
  CLEAN=$AM/poc_output/$gid/seed_1/mesh_registered_clean.ply
  if [ -z "$RECON" ]; then echo "[skip $gid] recon 없음"; skip="$skip $gid"; continue; fi
  if [ -f "$CLEAN" ] && [ "${FORCE:-0}" != "1" ]; then echo "[done $gid] 이미 등록됨"; ok="$ok $gid"; continue; fi

  echo "================ gid $gid ================"
  # 1) prepare (amodal3r)
  if ! run_in amodal3r "cd $RG && python prepare_amodal3r_input.py --scene $SCENE --gid $gid --topk 3 --fill hull \
      --relabel ~/relabel_$SCENE --amodal ~/amodal_$SCENE --scene_img data/$SCENE/images --out $AM/input"; then
    echo "[fail $gid] prepare"; fail="$fail ${gid}:prep"; continue; fi
  # 2) gen (amodal3r)
  rm -rf $AM/poc_output/$gid/seed_1
  if ! run_in amodal3r "cd $AM && python run_amodal3r_poc.py --labels $gid --seeds 1 --n_views 3 --skip_glb"; then
    echo "[fail $gid] gen"; fail="$fail ${gid}:gen"; continue; fi
  GEN=$AM/poc_output/$gid/seed_1/mesh.ply
  if [ ! -f "$GEN" ]; then echo "[fail $gid] mesh 없음"; fail="$fail ${gid}:nomesh"; continue; fi
  # 3) register + clean (split_and_splat)
  REG=$AM/poc_output/$gid/seed_1/mesh_registered.ply
  if ! run_in split_and_splat "cd $RG && python register_robust.py --gen '$GEN' --recon '$RECON' --out '$REG' \
      && python clean_components.py --in '$REG' --out '$CLEAN' --min_frac 0.02"; then
    echo "[fail $gid] register"; fail="$fail ${gid}:reg"; continue; fi
  if [ -f "$CLEAN" ]; then echo "[ok $gid]"; ok="$ok $gid"; else echo "[fail $gid] clean 없음"; fail="$fail ${gid}:clean"; fi
done

echo ""
echo "==================== 요약 ===================="
echo "성공(등록됨):$ok"
echo "실패:$fail"
echo "skip(recon없음):$skip"
echo "→ 성공한 객체로 assemble_whole_scene.py 가 전체 씬을 조립합니다."
