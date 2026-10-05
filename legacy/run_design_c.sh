#!/usr/bin/env bash
# Design C over several objects: prior gaussians in the representation, then one TSDF
# over training views AND novel poses.
#
#   1  make_prior_depth      novel poses that see the unobserved side + prior depth
#   2  inject_prior_gaussians prior surface as gaussians, GT-depth carved
#   3  mesh_tsdf_views       rendered-depth TSDF over both view sets, boundary filtered
#   4  eval_seen_unseen      against grid fusion (design B) on the same object
#
# Run run_field_fusion_batch.sh FIRST with the same PRIOR and OBJ: it builds the field
# from THIS base's reconstruction and produces design B's mesh to compare against. A
# field conditioned on a different reconstruction is what invalidated the earlier runs.
#
# STAGE=clean tags its output tsdf_clean_${RUN}.ply, so RECON_NAME below must carry the
# same RUN. Pass RUN explicitly rather than letting each invocation mint its own timestamp.
# For the CONFIRMED pipeline this stage now lives in run_scene_pipeline.sh as the 'cond'
# stage (fixed name tsdf_clean.ply, no RUN tag); use that unless you are A/B-ing filters.
#
#   RUN=0918a GIDS="2 5 6" STAGE=clean bash run_design_c.sh   # conditioning reconstruction
#   ONLY="2 5 6" OUT=~/RefineGS/output/replica_room0_v2/objects_voted ITER=30000 \
#     PRIOR=~/prior_voted PKL_SUBDIR=voted RUN=0918a \
#     RECON_NAME=tsdf_clean_0918a.ply PHASE=all \
#     bash run_field_fusion_batch.sh                        # prior + design B
#   RUN=0918a GIDS="2 5 6" bash run_design_c.sh             # design C
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
OBJ=${OBJ:-${ROOT}/output/${SCENE}/objects_voted}
# One tag per invocation: nothing is written over an earlier result. Pass RUN=<tag> to
# resume, and the freshness check below then skips the stages that already finished.
RUN=${RUN:-$(date +%m%d_%H%M)}
INJ=${INJ:-${ROOT}/output/${SCENE}/objects_inj_${RUN}}
# Same default as the batch script, so one RUN= lines the two up automatically.
FUSE_NAME=${FUSE_NAME:-fused_${RUN}}
PRIOR=${PRIOR:-$HOME/prior_voted}
ITER=${ITER:-30000}
GIDS=${GIDS:-"2 5 6"}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS_DIR=${STEMS_DIR:-$HOME/See3D/dataset/stage6/clean_stems}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
# 0.005 measured better. At 0.02 every prior point within 2cm of an existing gaussian is
# dropped, leaving a band the TSDF cannot fill: on obj6, 0.005 gave unseen F@2 0.6103 and
# 0.02 gave 0.5475 against the same prior generation. 0.02 is run_inject_test.sh's value
# and produces a cleaner-looking point cloud, but the metric prefers 0.005.
NEW_DIST=${NEW_DIST:-0.005}
GSCALE=${GSCALE:-0.006}
CLEAN=${CLEAN:-0}               # 1 = redo every stage
STAGE=${STAGE:-c}               # clean | c | all
CSV=${CSV:-${INJ}/_design_c.csv}
CLN_NAME=tsdf_clean_${RUN}.ply
# The conditioning surface is filtered harder than a mesh meant for viewing. ShapeR
# anchors to these points, so a ragged boundary is copied into the prior; losing a few
# percent of good surface costs nothing because the prior fills it back in.
CLEAN_ARGS=${CLEAN_ARGS:-"--min_alpha 0.7 --min_cos 0.35 --max_jump 0.02 --erode 3"}
NAMES=${NAMES:-${OBJ}/names.tsv}

cd "${ROOT}" || exit 1
mkdir -p "${PRIOR}" "${INJ}"
[ "${CLEAN}" = "1" ] && rm -f "${CSV}"
name_of() { [ -f "${NAMES}" ] && awk -F'\t' -v g="$1" '$1==g{print $2; exit}' "${NAMES}"; }

# Skipping on existence alone silently reuses a file whose provenance is invisible -- a
# stale prior built from another reconstruction is what invalidated two days of runs.
# A result is reusable only if it is newer than its generator and its inputs.
fresh() {                                     # fresh TARGET DEP...
  local t=$1 d; shift
  [ -f "${t}" ] || return 1
  for d in "$@"; do [ -e "${d}" ] && [ "${d}" -nt "${t}" ] && return 1; done
  return 0
}

echo "run=${RUN}  inj=${INJ}  vs ${FUSE_NAME}_post.ply"

# Stage "clean": the observed surface without its rough seen/unseen band, which is what
# should condition the generation. ShapeR anchors to the conditioning points, so a ragged
# boundary is reproduced in the prior. Run this BEFORE run_field_fusion_batch.sh and point
# its RECON_NAME here.
if [ "${STAGE}" = "clean" ] || [ "${STAGE}" = "all" ]; then
  for g in ${GIDS}; do
    CLN=${OBJ}/${g}/train/ours_${ITER}/${CLN_NAME}
    [ "${CLEAN}" = "1" ] && rm -f "${CLN}"
    fresh "${CLN}" mesh_tsdf_views.py "${OBJ}/${g}/point_cloud" \
      && { echo "  [${g}] clean: reuse"; continue; }
    python mesh_tsdf_views.py -m "${OBJ}/${g}" --out "${CLN}" ${CLEAN_ARGS} 2>&1 \
      | grep -E "^\[out\]|kept" | tail -2 | sed "s/^/  [${g}] /"
  done
  [ "${STAGE}" = "clean" ] && exit 0
fi

for g in ${GIDS}; do
  NPZ=${PRIOR}/obj${g}_field.npz
  BASE=${OBJ}/${g}
  OUTD=${OBJ}/${g}/train/ours_${ITER}
  DST=${INJ}/${g}
  PD=${PRIOR}/pd${g}_${RUN}.npz
  TSDF=${DST}/train/ours_${ITER}/tsdf.ply
  echo ""; echo "######## ${g} $(name_of "${g}") ########"
  [ -f "${NPZ}" ]  || { echo "  no field ${NPZ} -- run run_field_fusion_batch.sh"; continue; }
  [ -f "${OUTD}/${FUSE_NAME}_post.ply" ] || echo "  WARN no design-B mesh; eval will skip"
  STEMS=${STEMS_DIR}/${g}.txt

  # prior_mesh.py exists for exactly this: see whether a gap is the generation's fault
  # or the fusion's. Look at it next to fuse_post.ply before trusting anything downstream.
  [ "${CLEAN}" = "1" ] && rm -f "${PRIOR}/obj${g}_prior_${RUN}.ply"
  fresh "${PRIOR}/obj${g}_prior_${RUN}.ply" prior_mesh.py "${NPZ}" \
    || python prior_mesh.py "${NPZ}" \
      --out "${PRIOR}/obj${g}_prior_${RUN}.ply" 2>&1 | tail -3 | sed 's/^/  /'
  [ -f "${PRIOR}/obj${g}_prior_${RUN}.ply" ] || { echo "  prior mesh FAILED"; continue; }

  [ "${CLEAN}" = "1" ] && rm -f "${PD}"
  fresh "${PD}" make_prior_depth.py "${NPZ}" \
    || python make_prior_depth.py --npz "${NPZ}" --colmap "${COLMAP}" \
      --gt_depth_dir "${GTD}" --occluder_mesh "${GT_MESH}" --out "${PD}" \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      2>&1 | tail -4 | sed 's/^/  /'
  [ -f "${PD}" ] || { echo "  prior depth FAILED"; continue; }

  [ "${CLEAN}" = "1" ] && rm -rf "${DST}"
  fresh "${DST}/point_cloud/iteration_${ITER}/point_cloud.ply" \
        inject_prior_gaussians.py "${NPZ}" "${BASE}/point_cloud" \
    || python inject_prior_gaussians.py -m "${BASE}" --iteration "${ITER}" \
      --fields "${NPZ}" --out "${DST}" --max_new 0 --new_dist "${NEW_DIST}" \
      --scale "${GSCALE}" --colmap "${COLMAP}" --carve_depth_dir "${GTD}" \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      2>&1 | tail -2 | sed 's/^/  /'
  [ -f "${DST}/point_cloud/iteration_${ITER}/point_cloud.ply" ] || { echo "  inject FAILED"; continue; }

  [ "${CLEAN}" = "1" ] && rm -f "${TSDF}"
  fresh "${TSDF}" mesh_tsdf_views.py "${PD}" "${DST}/point_cloud" \
    || python mesh_tsdf_views.py -m "${DST}" --prior_depth "${PD}" \
      --out "${TSDF}" 2>&1 | grep -E "^\[cfg\]|^\[views\]|^\[out\]|kept" | tail -4 | sed 's/^/  /'
  [ -f "${TSDF}" ] || { echo "  tsdf FAILED"; continue; }

  [ -f "${OUTD}/${FUSE_NAME}_post.ply" ] || continue
  python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
    --recon "${OUTD}/${FUSE_NAME}_post.ply" --recon2 "${TSDF}" \
    --colmap "${COLMAP}" --gid "${g}" --masks_root "${MASKS}" --use_mask \
    --csv "${CSV}" --tag "c${g}" 2>&1 | sed -n '/^===== A ->/,$p' | sed 's/^/  /'
done

echo ""; echo "csv: ${CSV}   (A = grid fusion, B = injection + novel-pose TSDF)"
