#!/usr/bin/env bash
# Three-way comparison on the same objects:
#   A  fuse_post.ply          observed reconstruction
#   B  fused_field_post.ply   grid fusion (the current method)
#   C  prior injected as gaussians, then meshed through render.py
#
# C uses no optimiser and no pruning. If it already moves the unseen metrics, the premise
# of "prior as material for the representation" holds and fine-tuning is refinement.
#
#   GIDS="6 2 5" PRIOR=~/prior_smoke bash run_inject_test.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
OBJ=${OBJ:-${ROOT}/output/${SCENE}/objects_voted}
INJ=${INJ:-${ROOT}/output/${SCENE}/objects_inj}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
PRIOR=${PRIOR:-$HOME/prior_smoke}
ITER=${ITER:-30000}
GIDS=${GIDS:-"6 2 5"}
MAX_NEW=${MAX_NEW:-40000}
NEW_DIST=${NEW_DIST:-0.02}
GSCALE=${GSCALE:-0.006}
VOXEL=${VOXEL:-0.004}

cd "${ROOT}" || exit 1
for g in ${GIDS}; do
  NPZ=${PRIOR}/obj${g}_field.npz
  SRC=${OBJ}/${g}
  DST=${INJ}/${g}
  [ -f "${NPZ}" ] || { echo "[${g}] no field ${NPZ}"; continue; }
  echo ""; echo "######## gid ${g} ########"
  python inject_prior_gaussians.py -m "${SRC}" --iteration "${ITER}" \
    --fields "${NPZ}" --out "${DST}" --max_new "${MAX_NEW}" \
    --new_dist "${NEW_DIST}" --scale "${GSCALE}" || continue
  # same render.py call as mesh_voted_objects.sh, so A and C are comparable
  python render.py -m "${DST}" -s "${MASKS}/${g}" --iteration "${ITER}" --skip_test \
    --depth_ratio 1 --depth_trunc 5.0 --voxel_size "${VOXEL}" --sdf_trunc 0.02 \
    --num_cluster 1 > "${DST}/mesh.log" 2>&1 \
    || { echo "  render FAILED"; tail -5 "${DST}/mesh.log" | sed 's/^/    /'; continue; }

  for pair in "A=${OBJ}/${g}/train/ours_${ITER}/fuse_post.ply" \
              "B=${OBJ}/${g}/train/ours_${ITER}/fused_field_post.ply" \
              "C=${DST}/train/ours_${ITER}/fuse_post.ply"; do
    tag=${pair%%=*}; mesh=${pair#*=}
    [ -f "${mesh}" ] || { echo "  [${tag}] missing"; continue; }
    echo "  ---- ${tag} ----"
    python eval_seen_unseen.py --gt_mesh "${GT_MESH}" --recon "${mesh}" \
      --colmap "${COLMAP}" --gid "${g}" --masks_root "${MASKS}" --use_mask --seed 0 \
      2>&1 | grep -E "^  \[SEEN|^  \[UNSEEN|^  \[FREE|accuracy|completion|F@2.0cm" \
      | sed 's/^/    /' | head -12
  done
done
