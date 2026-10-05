#!/usr/bin/env bash
# Scene 1-pass pipeline: one trained scene -> voted instances -> per-object refinement.
#
#   vote     label every gaussian by multi-view voting (GT-depth occlusion test)
#   extract  slice the scene model into per-object 3DGS dirs
#   mesh     render.py TSDF per object -> fuse_post.ply   (the A side)
#   cond     boundary-filtered TSDF -> tsdf_clean.ply     (conditions the generation only)
#   pkl      ShapeR input
#   field    ShapeR signed SDF grid
#   fuse     grid fusion + seen/unseen evaluation         (the B side)
#
# A stage is skipped when its output already exists, so a failed run resumes where it
# stopped. FROM= restarts at a stage; CLEAN=1 additionally deletes that stage's outputs and
# everything after it, which is what you want when an INPUT changed rather than a crash.
#
#   bash run_scene_pipeline.sh
#   FROM=mesh bash run_scene_pipeline.sh              # resume, keep existing meshes
#   CLEAN=1 FROM=vote bash run_scene_pipeline.sh      # redo everything from scratch
#   EXTRACT_EXTRA="--min_margin 0.3" CLEAN=1 FROM=extract bash run_scene_pipeline.sh
#   CLEAN=1 FROM=cond bash run_scene_pipeline.sh     # redo the conditioning surface onward
#   RUN=0918a FROM=fuse bash run_scene_pipeline.sh   # resume into an earlier run's outputs
#
# CLEAN=1 (delete outputs) and the 'cond' stage are unrelated despite the old name
# tsdf_clean.ply. FROM=cond does not delete anything on its own.
#
# Stage 0 (COLMAP, SAM3 masks, per-object folders, labels_scene/id_map.json) is NOT here.
# Run run_stage0.sh first; the [check] block below is exactly its exit contract.
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
DATA=${DATA:-${ROOT}/data/${SCENE}}
# Defaults are the CONFIRMED configuration. scene_reg and scene_n were regularisation
# experiments that lost (scene-level depth error 5.5mm -> 23.1 / 32.2mm), so `scene` is the
# model to use; override SCENE_MODEL only to reproduce those.
SCENE_MODEL=${SCENE_MODEL:-${ROOT}/output/${SCENE}/scene}
OBJ=${OBJ:-${ROOT}/output/${SCENE}/objects_voted}
PRIOR=${PRIOR:-$HOME/prior_v3}
ITER=${ITER:-30000}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
LABEL_DIR=${LABEL_DIR:-${DATA}/labels_scene}
IMAGES=${IMAGES:-${DATA}/images}
MASKS=${MASKS:-${DATA}/masks}
ONLY=${ONLY:-}                     # restrict to some gids, e.g. ONLY="9 11 63"
POINTS_FROM=${POINTS_FROM:-mesh}   # mesh | depth  (see make_shaper_input.py)
# poses live in sparse/0 or sparse_dense/0 depending on how the room was built
if [ -z "${COLMAP:-}" ]; then
  for sd in sparse/0 sparse_dense/0; do
    [ -d "${DATA}/${sd}" ] && { COLMAP=${DATA}/${sd}; break; }
  done
fi
COLMAP=${COLMAP:-${DATA}/sparse/0}
# One tag for the WHOLE pipeline run. run_field_fusion_batch.sh mints its own when it is
# not given one, and this script calls it three times: without this the logs, the failure
# CSVs and the fused mesh of a single run land under three different timestamps.
RUN=${RUN:-$(date +%m%d_%H%M)}
# Tagged, so a rerun cannot overwrite numbers that were already reported. The batch script
# deletes ${CSV} at the start of its fuse phase, which used to erase the previous run.
CSV=${CSV:-${OBJ}/_scene_${RUN}.csv}
FROM=${FROM:-vote}                 # vote | extract | mesh | cond | pkl | field | fuse
CLEAN=${CLEAN:-0}
# Conditioning surface for the generation (the 'cond' stage). NOT the same thing as
# CLEAN=1, which deletes outputs -- hence the stage is called 'cond', not 'clean'.
# Filtered harder than a mesh meant for viewing: ShapeR anchors to the conditioning
# points, so a ragged seen/unseen boundary is copied straight into the prior. Losing a few
# percent of good surface costs nothing because the prior fills it back in.
COND_NAME=${COND_NAME:-tsdf_clean.ply}
COND_ARGS=${COND_ARGS:-"--min_alpha 0.7 --min_cos 0.35 --max_jump 0.02 --erode 3"}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
GT_INFO=${GT_INFO:-$HOME/room_0/habitat/info_semantic.json}
SHAPER_DIR=${SHAPER_DIR:-$HOME/ShapeR}
PKL_SUBDIR=${PKL_SUBDIR:-$(basename "${OBJ}")}   # keeps pkls apart from other pipelines
# Extraction filters, off by default. Turn them on only as a separate experiment, so the
# change is attributable: --min_margin drops gaussians whose views disagreed (object
# boundaries), --split_below keeps the largest blob of a merged label.
EXTRACT_EXTRA=${EXTRACT_EXTRA:-}

PLY=${SCENE_MODEL}/point_cloud/iteration_${ITER}/point_cloud.ply
# Voting depends on the LABEL SOURCE, not just the scene model. Two label sets over one
# scene (SAM3 and GT-raycast, say) must not share a vote directory: the second run finds
# labels.npy, skips voting, and silently extracts the first run's assignment under the
# second run's names.
VOTE=${VOTE:-${OBJ}/vote}
stage_at() {                        # is this stage at or after FROM?
  local order="vote extract mesh cond pkl field fuse" i=0 j=0 k=0
  for s in ${order}; do i=$((i+1)); [ "$s" = "$1" ] && j=$i; [ "$s" = "${FROM}" ] && k=$i; done
  [ "$j" -ge "$k" ]
}

# Existence alone is not freshness: a conditioning surface built from an older point cloud
# looks identical on disk. Same rule as run_design_c.sh -- reuse only what is newer than
# its generator and its inputs.
fresh() {                           # fresh TARGET DEP...
  local t=$1 d; shift
  [ -f "${t}" ] || return 1
  for d in "$@"; do [ -e "${d}" ] && [ "${d}" -nt "${t}" ] && return 1; done
  return 0
}

echo "[check] paths"
fail=0
for p in "${PLY}" "${SCENE_MODEL}/cfg_args" "${COLMAP}" "${LABEL_DIR}/id_map.json" \
         "${MASKS}" "${IMAGES}" "${GTD}"; do
  [ -e "${p}" ] || { echo "  MISSING ${p}"; fail=1; }
done
[ "${fail}" -eq 0 ] || { echo "[abort] fix the paths above (run_stage0.sh builds them)"; exit 1; }
echo "  scene=${SCENE_MODEL}"
echo "  objects=${OBJ}   prior=${PRIOR}"
echo "  poses=${COLMAP}   gt_depth=${GTD}"
echo "  labels=${LABEL_DIR}   vote=${VOTE}"
echo "  pkl=${SHAPER_DIR}/data/${PKL_SUBDIR}   from=${FROM}   clean=${CLEAN}   run=${RUN}"
cd "${ROOT}" || exit 1

if [ "${CLEAN}" = "1" ]; then
  echo ""; echo "=== clean: removing outputs from '${FROM}' onward ==="
  stage_at vote    && { echo "  ${VOTE}"; rm -rf "${VOTE}"; }
  stage_at extract && { echo "  ${OBJ}  (incl. names.tsv)"; rm -rf "${OBJ}"; }
  # 'mesh' only re-meshes; extract already removed the dirs when it ran
  if stage_at mesh && [ -d "${OBJ}" ]; then
    echo "  ${OBJ}/*/train/ours_${ITER}/fuse*.ply"
    rm -f "${OBJ}"/*/train/ours_"${ITER}"/fuse.ply "${OBJ}"/*/train/ours_"${ITER}"/fuse_post.ply
  fi
  if stage_at cond && [ -d "${OBJ}" ]; then
    echo "  ${OBJ}/*/train/ours_${ITER}/${COND_NAME}"
    rm -f "${OBJ}"/*/train/ours_"${ITER}"/"${COND_NAME}"
  fi
  stage_at pkl   && { echo "  ${SHAPER_DIR}/data/${PKL_SUBDIR}"; rm -rf "${SHAPER_DIR}/data/${PKL_SUBDIR}"; }
  stage_at field && { echo "  ${PRIOR}/obj*_field*.npz";         rm -f "${PRIOR}"/obj*_field*.npz; }
  # The fused meshes and the CSV carry ${RUN}, so an earlier run's outputs are left alone
  # on purpose: CLEAN removes inputs to redo, not evidence for numbers already reported.
  if stage_at fuse && [ -d "${OBJ}" ]; then
    echo "  ${OBJ}/*/train/ours_${ITER}/fused_${RUN}*.ply"
    rm -f "${OBJ}"/*/train/ours_"${ITER}"/fused_"${RUN}"*.ply
  fi
fi

# Skip when the output is already there. FROM only sets where to START; deleting outputs
# is CLEAN's job, so the two do not have to be reasoned about together.
if stage_at vote && [ ! -f "${VOTE}/labels.npy" ]; then
  echo ""; echo "=== vote: per-gaussian instance labels ==="
  python vote_labels.py --ply "${PLY}" --colmap "${COLMAP}" \
    --label_dir "${LABEL_DIR}" --gt_depth_dir "${GTD}" \
    --out "${VOTE}" || exit 1
  echo "--- label coherence (previous run: mean compactness 0.754, 17 classes >= 0.8) ---"
  python check_scene_labels.py --ply "${PLY}" --labels "${VOTE}/labels.npy" \
    | tail -6
fi

if stage_at extract && [ ! -f "${OBJ}/objects.json" ]; then
  echo ""; echo "=== extract: slice the scene into per-object models ==="
  python extract_objects.py --ply "${PLY}" --labels "${VOTE}/labels.npy" \
    --scene_dir "${SCENE_MODEL}" --id_map "${LABEL_DIR}/id_map.json" \
    --source_root "${MASKS}" --vote_dir "${VOTE}" \
    --out "${OBJ}" --iter "${ITER}" ${EXTRACT_EXTRA} || exit 1
fi

# Name the objects against the GT semantic mesh. Runs on the sliced gaussians, so it fits
# between extract and mesh. Gives readable logs and, more usefully, real ShapeR captions:
# without it every object is generated from "a 3D object in a room".
# Not gated on FROM: the pkl stage consumes names.tsv as its captions, so it has to exist
# whenever we start at or before pkl.
if [ ! -f "${OBJ}/names.tsv" ] && [ -f "${GT_MESH}" ] && [ -d "${OBJ}" ]; then
  echo ""; echo "=== name: match each object to a GT class ==="
  python name_objects.py --gt_mesh "${GT_MESH}" --gt_info "${GT_INFO}" \
    --root "${OBJ}" --iter "${ITER}" || true
fi

if stage_at mesh; then
  echo ""; echo "=== mesh: TSDF per object (the A side) ==="
  OBJ="${OBJ}" DATA="${MASKS}" IT="${ITER}" bash mesh_voted_objects.sh
fi

# The surface that conditions the generation, which is NOT the surface we report.
# fuse_post.ply keeps the rough band where observation runs out (grazing angles, silhouette
# depth steps, half-transparent pixels); it sits ON the surface, so the free-space filter in
# make_shaper_input.py passes it, and ShapeR anchors to it. Filtering it out moved obj6
# unseen F@2 0.5942 -> 0.6382. Before this stage existed the whole-pipeline run silently
# conditioned on fuse_post.ply and lost that gain.
# Reported meshes still come from fuse_post.ply (the A side) and the fusion (the B side).
if stage_at cond; then
  echo ""; echo "=== cond: conditioning surface (${COND_NAME}) ==="
  for MDIR in "${OBJ}"/*/; do
    gid=$(basename "${MDIR}")
    [[ "${gid}" =~ ^[0-9]+$ ]] || continue
    [ -z "${ONLY}" ] || [[ " ${ONLY} " == *" ${gid} "* ]] || continue
    [ -f "${MDIR}point_cloud/iteration_${ITER}/point_cloud.ply" ] || continue
    CLN=${MDIR}train/ours_${ITER}/${COND_NAME}
    # The name carries no RUN tag, so a stale file here is invisible: check it against the
    # generator and the reconstruction it was built from rather than against existence.
    fresh "${CLN}" mesh_tsdf_views.py "${MDIR}point_cloud" \
      && { echo "  [${gid}] reuse"; continue; }
    python mesh_tsdf_views.py -m "${MDIR%/}" --load_iteration "${ITER}" \
      --out "${CLN}" ${COND_ARGS} 2>&1 \
      | grep -E "^\[out\]|kept" | tail -2 | sed "s/^/  [${gid}] /"
    [ -f "${CLN}" ] || echo "  [${gid}] FAILED -- pkl will report a missing recon"
  done
fi

for ph in pkl field fuse; do
  stage_at "${ph}" || continue
  echo ""; echo "=== ${ph} ==="
  # RECON_NAME is what makes the cond stage count. Without it the batch script falls back
  # to its own default, fuse_post.ply, and the stage above is dead weight.
  # RUN is passed so all three phases share one log dir and one fused-mesh tag; FAILCSV is
  # per phase because the batch script truncates it on entry.
  PRIOR="${PRIOR}" ITER="${ITER}" OUT="${OBJ}" CSV="${CSV}" PHASE="${ph}" \
    RUN="${RUN}" FAILCSV="${OBJ}/_scene_${RUN}_${ph}_failures.csv" \
    PKL_SUBDIR="${PKL_SUBDIR}" CAPTIONS="${OBJ}/names.tsv" COLMAP="${COLMAP}" \
    MASKS="${MASKS}" IMAGES="${IMAGES}" GT_MESH="${GT_MESH}" ONLY="${ONLY}" \
    POINTS_FROM="${POINTS_FROM}" RECON_NAME="${COND_NAME}" \
    bash run_field_fusion_batch.sh || exit 1
done

echo ""
echo "[done] run=${RUN}  summary csv: ${CSV}"
echo "compare against:"
echo "  per-object   seen F@1 0.895 -> 0.915   unseen F@2 0.185 -> 0.250   free 4.40 -> 4.91%"
echo "  scene (old)  seen F@1 0.899 -> 0.888   unseen F@2 0.187 -> 0.243   free 9.88 -> 7.15%"