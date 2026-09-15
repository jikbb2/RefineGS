#!/usr/bin/env bash
# Scene 1-pass pipeline: one trained scene -> voted instances -> per-object refinement.
#
#   vote     label every gaussian by multi-view voting (GT-depth occlusion test)
#   extract  slice the scene model into per-object 3DGS dirs
#   mesh     render.py TSDF per object -> fuse_post.ply   (the A side)
#   pkl      ShapeR input
#   field    ShapeR signed SDF grid
#   fuse     grid fusion + seen/unseen evaluation         (the B side)
#
# Each stage is skipped when its output already exists, so a failed run can be resumed.
# Use FROM= to force a restart at a stage.
#
#   bash run_scene_pipeline.sh
#   FROM=mesh bash run_scene_pipeline.sh
#   SCENE_MODEL=.../scene_reg PRIOR=~/prior_reg bash run_scene_pipeline.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
DATA=${DATA:-${ROOT}/data/${SCENE}}
SCENE_MODEL=${SCENE_MODEL:-${ROOT}/output/${SCENE}/scene_reg}
OBJ=${OBJ:-${ROOT}/output/${SCENE}/objects_reg}
PRIOR=${PRIOR:-$HOME/prior_reg}
ITER=${ITER:-30000}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
LABEL_DIR=${LABEL_DIR:-${DATA}/labels_scene}
CSV=${CSV:-$HOME/prior/_scene_reg.csv}
FROM=${FROM:-vote}                 # vote | extract | mesh | pkl | field | fuse
# Extraction filters, off by default. Turn them on only as a separate experiment, so the
# change is attributable: --min_margin drops gaussians whose views disagreed (object
# boundaries), --split_below keeps the largest blob of a merged label.
EXTRACT_EXTRA=${EXTRACT_EXTRA:-}

PLY=${SCENE_MODEL}/point_cloud/iteration_${ITER}/point_cloud.ply
stage_at() {                        # is this stage at or after FROM?
  local order="vote extract mesh pkl field fuse" i=0 j=0 k=0
  for s in ${order}; do i=$((i+1)); [ "$s" = "$1" ] && j=$i; [ "$s" = "${FROM}" ] && k=$i; done
  [ "$j" -ge "$k" ]
}

echo "[check] paths"
fail=0
for p in "${PLY}" "${SCENE_MODEL}/cfg_args" "${DATA}/sparse/0" "${LABEL_DIR}/id_map.json" \
         "${DATA}/masks" "${GTD}"; do
  [ -e "${p}" ] || { echo "  MISSING ${p}"; fail=1; }
done
[ "${fail}" -eq 0 ] || { echo "[abort] fix the paths above (override with env vars)"; exit 1; }
echo "  scene=${SCENE_MODEL}"
echo "  objects=${OBJ}   prior=${PRIOR}   from=${FROM}"
cd "${ROOT}" || exit 1

if stage_at vote && { [ "${FROM}" = vote ] || [ ! -f "${SCENE_MODEL}/vote/labels.npy" ]; }; then
  echo ""; echo "=== vote: per-gaussian instance labels ==="
  python vote_labels.py --ply "${PLY}" --colmap "${DATA}/sparse/0" \
    --label_dir "${LABEL_DIR}" --gt_depth_dir "${GTD}" \
    --out "${SCENE_MODEL}/vote" || exit 1
  echo "--- label coherence (previous run: mean compactness 0.754, 17 classes >= 0.8) ---"
  python check_scene_labels.py --ply "${PLY}" --labels "${SCENE_MODEL}/vote/labels.npy" \
    | tail -6
fi

if stage_at extract && { [ "${FROM}" = extract ] || [ ! -f "${OBJ}/objects.json" ]; }; then
  echo ""; echo "=== extract: slice the scene into per-object models ==="
  python extract_objects.py --ply "${PLY}" --labels "${SCENE_MODEL}/vote/labels.npy" \
    --scene_dir "${SCENE_MODEL}" --id_map "${LABEL_DIR}/id_map.json" \
    --source_root "${DATA}/masks" --vote_dir "${SCENE_MODEL}/vote" \
    --out "${OBJ}" --iter "${ITER}" ${EXTRACT_EXTRA} || exit 1
fi

if stage_at mesh; then
  echo ""; echo "=== mesh: TSDF per object (the A side) ==="
  OBJ="${OBJ}" DATA="${DATA}/masks" IT="${ITER}" bash mesh_voted_objects.sh
fi

for ph in pkl field fuse; do
  stage_at "${ph}" || continue
  echo ""; echo "=== ${ph} ==="
  PRIOR="${PRIOR}" ITER="${ITER}" OUT="${OBJ}" CSV="${CSV}" PHASE="${ph}" \
    bash run_field_fusion_batch.sh || exit 1
done

echo ""
echo "[done] summary csv: ${CSV}"
echo "compare against:"
echo "  per-object   seen F@1 0.895 -> 0.915   unseen F@2 0.185 -> 0.250   free 4.40 -> 4.91%"
echo "  scene (old)  seen F@1 0.899 -> 0.888   unseen F@2 0.187 -> 0.243   free 9.88 -> 7.15%"
