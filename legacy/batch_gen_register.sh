#!/usr/bin/env bash
# RefineGS — 전 객체 Amodal3R gen + register 배치 (전체 씬 조립 준비). 3단계(모델 1회 로딩).
#   Phase1 prepare(객체별, amodal3r) → Phase2 gen(전체 1프로세스, amodal3r) → Phase3 register(객체별, split_and_splat)
#
# 사용:
#   bash batch_gen_register.sh             # refinegs_fix 의 모든 gid 자동
#   bash batch_gen_register.sh "0 1 8 27"  # 특정 gid만
#   백그라운드: nohup bash batch_gen_register.sh > ~/batch_gr.log 2>&1 &
#
# register 는 recon *메쉬*(train/ours_*/fuse_post.ply) 필요. 평면/대칭은 register 실패→skip(그 객체는 base로 남음).
# 결과: ~/Amodal3R/poc_output/<gid>/seed_1/mesh_registered_clean.ply (성공 객체만)
set -u
RG=/home/elicer/RefineGS
AM=$HOME/Amodal3R
SCENE=replica_room0_v2
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

run_in() { bash -c "source $CONDA_SH && conda activate $1 && unset PYTHONPATH && export LD_LIBRARY_PATH= && $2"; }

if [ "$#" -ge 1 ]; then GIDS="$1"
else GIDS=$(ls "$RG/output/$SCENE/refinegs_fix" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); fi

# recon 메쉬 있는 gid만 대상
TARGS=""
for gid in $GIDS; do
  RM=$(ls $RG/output/$SCENE/refinegs_fix/$gid/train/ours_*/fuse_post.ply 2>/dev/null | head -1)
  if [ -z "$RM" ]; then echo "[skip $gid] recon 메쉬 없음"; continue; fi
  if [ -f "$AM/poc_output/$gid/seed_1/mesh_registered_clean.ply" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[done $gid] 이미 등록됨"; continue; fi
  TARGS="$TARGS $gid"
done
TARGS=$(echo $TARGS | xargs)
[ -z "$TARGS" ] && { echo "처리할 객체 없음(전부 등록됐거나 recon 메쉬 없음)"; exit 0; }
echo "대상 gid: $TARGS"

# ── Phase 1: prepare (객체별) ──
echo "===== Phase 1: prepare ====="
PREP=""
for gid in $TARGS; do
  if run_in amodal3r "cd $RG && python prepare_amodal3r_input.py --scene $SCENE --gid $gid --topk 3 --fill hull \
      --relabel ~/relabel_$SCENE --amodal ~/amodal_$SCENE --scene_img data/$SCENE/images --out $AM/input"; then
    PREP="$PREP $gid"; rm -rf $AM/poc_output/$gid/seed_1
  else echo "[fail $gid] prepare"; fi
done
PREP=$(echo $PREP | xargs)
[ -z "$PREP" ] && { echo "prepare 성공 0 — 중단"; exit 1; }

# ── Phase 2: gen (전체 1프로세스 = 모델 1회 로딩) ──
echo "===== Phase 2: gen (labels: $PREP) ====="
run_in amodal3r "cd $AM && python run_amodal3r_poc.py --labels $PREP --seeds 1 --n_views 3 --skip_glb" \
  || echo "[warn] gen 일부 실패 가능 — 아래에서 mesh 존재로 판정"

# ── Phase 3: register (객체별) ──
echo "===== Phase 3: register ====="
ok=""; fail=""
for gid in $PREP; do
  GEN=$AM/poc_output/$gid/seed_1/mesh.ply
  RM=$(ls $RG/output/$SCENE/refinegs_fix/$gid/train/ours_*/fuse_post.ply 2>/dev/null | head -1)
  CLEAN=$AM/poc_output/$gid/seed_1/mesh_registered_clean.ply
  REG=$AM/poc_output/$gid/seed_1/mesh_registered.ply
  if [ ! -f "$GEN" ]; then echo "[fail $gid] gen mesh 없음"; fail="$fail ${gid}:gen"; continue; fi
  if run_in split_and_splat "cd $RG && python register_robust.py --gen '$GEN' --recon '$RM' --out '$REG' \
      && python clean_components.py --in '$REG' --out '$CLEAN' --min_frac 0.02" && [ -f "$CLEAN" ]; then
    echo "[ok $gid]"; ok="$ok $gid"
  else echo "[fail $gid] register"; fail="$fail ${gid}:reg"; fi
done

echo ""
echo "==================== 요약 ===================="
echo "성공(등록됨):$ok"
echo "실패:$fail"
echo "→ assemble_whole_scene.py 가 성공 객체로 전체 씬을 조립합니다."
