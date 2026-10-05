#!/usr/bin/env bash
# gen-centric Amodal3R 완성 — 2 sub-stage (env별 수동 activate)
#
#   STAGE gen      (amodal3r env):        input 준비 + Amodal3R 추론 → mesh.ply (canonical)
#     conda activate amodal3r
#     SCENE=replica_room0_v2 bash run_amodal3r_genstage.sh gen
#
#   STAGE register (split_and_splat env): register(Sim3) + 컴포넌트 정리 + scene 합성
#     conda activate split_and_splat
#     SCENE=replica_room0_v2 bash run_amodal3r_genstage.sh register
#
# 전제: 전체 depth recon 완료 (OUT/<gid>/train/ours_*/fuse_post.ply 존재).
set -uo pipefail
shopt -s nullglob
STAGE=${1:-help}

ROOT=/home/elicer/RefineGS
SCENE=${SCENE:-replica_room0_v2}
OUT=${OUT:-output/${SCENE}/refinegs_full}
A3R=${A3R:-$HOME/Amodal3R}
GENROOT=${GENROOT:-$HOME/Amodal3R/poc_output}
RELABEL=${RELABEL:-$HOME/relabel_${SCENE}}
AMODAL=${AMODAL:-$HOME/amodal_${SCENE}}
SCENE_IMG=${SCENE_IMG:-data/${SCENE}/images}
SEED=${SEED:-1}; NVIEWS=${NVIEWS:-1}; MINFRAC=${MINFRAC:-0.02}; ICP=${ICP:-0.15}
cd ${ROOT}

labels_with_recon () {   # recon(fuse_post)이 있는 객체만
  for d in ${OUT}/*/; do
    g=$(basename "$d")
    [ -n "$(ls ${OUT}/${g}/train/ours_*/fuse_post.ply 2>/dev/null)" ] && echo "$g"
  done
}

if [ "$STAGE" = "gen" ]; then
  LBLS=$(labels_with_recon)
  [ -n "$LBLS" ] || { echo "[ERROR] recon 객체 없음 — 먼저 depth recon 완료"; exit 1; }
  echo "=== [G1] Amodal3R 입력 준비 ==="
  for g in $LBLS; do
    python prepare_amodal3r_input.py --scene ${SCENE} --gid ${g} --topk ${NVIEWS} \
      --relabel ${RELABEL} --amodal ${AMODAL} --scene_img ${SCENE_IMG} --out ${A3R}/input \
      || echo "  prep fail ${g}"
  done
  echo "=== [G2] Amodal3R 추론 (labels: $(echo $LBLS|tr '\n' ' ')) ==="
  ( cd ${A3R} && python run_amodal3r_poc.py --labels ${LBLS} --seeds ${SEED} --n_views ${NVIEWS} --skip_glb )
  echo "=== gen DONE → 다음: conda activate split_and_splat && bash $0 register ==="
  exit 0
fi

if [ "$STAGE" = "register" ]; then
  echo "=== [R1] register(Sim3) + 컴포넌트 정리 + 객체폴더로 복사 ==="
  n=0
  for g in $(labels_with_recon); do
    GEN=$(ls ${GENROOT}/${g}/seed_${SEED}/mesh.ply 2>/dev/null | head -1)
    RECON=$(ls ${OUT}/${g}/train/ours_*/fuse_post.ply 2>/dev/null | head -1)
    [ -n "$GEN" ] && [ -n "$RECON" ] || { echo "  skip ${g} (gen/recon 없음)"; continue; }
    REG=${GENROOT}/${g}/seed_${SEED}/mesh_registered.ply
    CLEAN=${GENROOT}/${g}/seed_${SEED}/mesh_registered_clean.ply
    python register_generated_to_recon.py --gen "$GEN" --recon "$RECON" --out "$REG" --icp_dist ${ICP} \
      || { echo "  register fail ${g}"; continue; }
    python clean_components.py --in "$REG" --out "$CLEAN" --min_frac ${MINFRAC} \
      || { echo "  cleanup fail ${g}"; continue; }
    cp "$CLEAN" "$(dirname "$RECON")/fuse_genclean.ply"
    n=$((n+1))
  done
  echo "  완성 객체: ${n}"
  echo "=== [R2] scene 합성 (gen-centric) ==="
  python scene_assemble.py --root ${OUT} --mesh_name fuse_genclean \
    --out_mesh output/${SCENE}/scene_genclean_mesh.ply
  echo "=== register DONE → output/${SCENE}/scene_genclean_mesh.ply ==="
  exit 0
fi

echo "Usage:"
echo "  conda activate amodal3r        && SCENE=${SCENE} bash $0 gen"
echo "  conda activate split_and_splat && SCENE=${SCENE} bash $0 register"
