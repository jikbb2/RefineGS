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
#   ONLY="2 5 6" OUT=~/RefineGS/output/replica_room0_v2/objects_voted ITER=30000 \
#     PRIOR=~/prior_voted PKL_SUBDIR=voted PHASE=all bash run_field_fusion_batch.sh
#   GIDS="2 5 6" bash run_design_c.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
OBJ=${OBJ:-${ROOT}/output/${SCENE}/objects_voted}
INJ=${INJ:-${ROOT}/output/${SCENE}/objects_inj3}
PRIOR=${PRIOR:-$HOME/prior_voted}
ITER=${ITER:-30000}
GIDS=${GIDS:-"2 5 6"}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS_DIR=${STEMS_DIR:-$HOME/See3D/dataset/stage6/clean_stems}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
NEW_DIST=${NEW_DIST:-0.005}     # the boundary band the extraction filter empties
GSCALE=${GSCALE:-0.006}
CLEAN=${CLEAN:-0}               # 1 = redo every stage
CSV=${CSV:-${INJ}/_design_c.csv}
NAMES=${NAMES:-${OBJ}/names.tsv}

cd "${ROOT}" || exit 1
mkdir -p "${PRIOR}" "${INJ}"
[ "${CLEAN}" = "1" ] && rm -f "${CSV}"
name_of() { [ -f "${NAMES}" ] && awk -F'\t' -v g="$1" '$1==g{print $2; exit}' "${NAMES}"; }

for g in ${GIDS}; do
  NPZ=${PRIOR}/obj${g}_field.npz
  BASE=${OBJ}/${g}
  OUTD=${OBJ}/${g}/train/ours_${ITER}
  DST=${INJ}/${g}
  PD=${PRIOR}/pd${g}.npz
  TSDF=${DST}/train/ours_${ITER}/tsdf.ply
  echo ""; echo "######## ${g} $(name_of "${g}") ########"
  [ -f "${NPZ}" ]  || { echo "  no field ${NPZ} -- run run_field_fusion_batch.sh"; continue; }
  [ -f "${OUTD}/fused_field_post.ply" ] || echo "  WARN no design-B mesh; eval will skip"
  STEMS=${STEMS_DIR}/${g}.txt

  # The prior surface as a mesh, so it can be opened next to the reconstruction. A prior
  # built from the wrong reconstruction looks plausible in the metrics and wrong here.
  [ "${CLEAN}" = "1" ] && rm -f "${PRIOR}/obj${g}_prior.ply"
  [ -f "${PRIOR}/obj${g}_prior.ply" ] || python -c "
import numpy as np, os, sys, open3d as o3d
from skimage.measure import marching_cubes
z = np.load(sys.argv[1]); F = z['field'].astype(np.float32); G = F.shape[0]
v, f, _, _ = marching_cubes(F, level=0.0, spacing=(2.0/(G-1),)*3)
v = ((v - 1.0) / float(z['scale'])) @ z['R_align'] + z['center']
m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
m.compute_vertex_normals(); o3d.io.write_triangle_mesh(sys.argv[2], m)
print('  [prior] %d verts  inside %.1f%%  bbox %s %s' % (
    len(m.vertices), (F < 0).mean()*100, np.round(v.min(0), 2), np.round(v.max(0), 2)))
" "${NPZ}" "${PRIOR}/obj${g}_prior.ply" || { echo "  prior mesh FAILED"; continue; }

  [ "${CLEAN}" = "1" ] && rm -f "${PD}"
  [ -f "${PD}" ] || python make_prior_depth.py --npz "${NPZ}" --colmap "${COLMAP}" \
      --gt_depth_dir "${GTD}" --occluder_mesh "${GT_MESH}" --out "${PD}" \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      2>&1 | tail -12 | sed 's/^/  /'
  [ -f "${PD}" ] || { echo "  prior depth FAILED"; continue; }

  [ "${CLEAN}" = "1" ] && rm -rf "${DST}"
  [ -f "${DST}/point_cloud/iteration_${ITER}/point_cloud.ply" ] || \
    python inject_prior_gaussians.py -m "${BASE}" --iteration "${ITER}" \
      --fields "${NPZ}" --out "${DST}" --max_new 0 --new_dist "${NEW_DIST}" \
      --scale "${GSCALE}" --colmap "${COLMAP}" --carve_depth_dir "${GTD}" \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      2>&1 | tail -6 | sed 's/^/  /'
  [ -f "${DST}/point_cloud/iteration_${ITER}/point_cloud.ply" ] || { echo "  inject FAILED"; continue; }

  [ "${CLEAN}" = "1" ] && rm -f "${TSDF}"
  [ -f "${TSDF}" ] || python mesh_tsdf_views.py -m "${DST}" --prior_depth "${PD}" \
      --out "${TSDF}" 2>&1 | grep -E "^\[cfg\]|^\[views\]|^\[out\]|kept" | tail -4 | sed 's/^/  /'
  [ -f "${TSDF}" ] || { echo "  tsdf FAILED"; continue; }

  [ -f "${OUTD}/fused_field_post.ply" ] || continue
  python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
    --recon "${OUTD}/fused_field_post.ply" --recon2 "${TSDF}" \
    --colmap "${COLMAP}" --gid "${g}" --masks_root "${MASKS}" --use_mask \
    --csv "${CSV}" --tag "c${g}" 2>&1 | sed -n '/A ->\|A →/,$p' | sed 's/^/  /'
done

echo ""; echo "csv: ${CSV}   (A = grid fusion, B = injection + novel-pose TSDF)"
