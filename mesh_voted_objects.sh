#!/usr/bin/env bash
# TSDF-extract fuse_post.ply for every object split out of the scene model.
#
# Same render.py call and parameters as run_full_pipeline.sh, so the meshes are
# comparable to the per-object baseline. The only differences are the model dir
# (a label slice of the scene model instead of a separately trained object) and
# the iteration (the scene was trained to 30000, the per-object runs to 7000).
#
#   python render.py -m MDIR -s DATA --iteration IT --skip_test \
#     --depth_ratio 1 --depth_trunc 5.0 --voxel_size 0.004 --sdf_trunc 0.02 --num_cluster 1
#
# bash mesh_voted_objects.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
OBJ=${OBJ:-${ROOT}/output/${SCENE}/objects_voted}
DATA=${DATA:-${ROOT}/data/${SCENE}/masks}      # per-gid dirs with masks/
IT=${IT:-30000}
DEPTH_TRUNC=${DEPTH_TRUNC:-5.0}
VOXEL=${VOXEL:-0.004}
SDF_TRUNC=${SDF_TRUNC:-0.02}
NUM_CLUSTER=${NUM_CLUSTER:-1}

cd "${ROOT}" || exit 1
ok=0; ng=0; skip=0
for MDIR in "${OBJ}"/*/; do
  gid=$(basename "${MDIR}")
  [[ "${gid}" =~ ^[0-9]+$ ]] || continue
  [ -f "${MDIR}/point_cloud/iteration_${IT}/point_cloud.ply" ] || {
    echo "  [skip ${gid}] no ply at iteration_${IT}"; skip=$((skip+1)); continue; }
  if [ -f "${MDIR}/train/ours_${IT}/fuse_post.ply" ]; then
    skip=$((skip+1)); continue
  fi
  D="${DATA}/${gid}"
  [ -d "${D}/masks" ] || { echo "  [skip ${gid}] no ${D}/masks"; skip=$((skip+1)); continue; }

  echo "  [${gid}] mesh"
  python render.py -m "${MDIR}" -s "${D}" --iteration ${IT} --skip_test \
    --depth_ratio 1 --depth_trunc ${DEPTH_TRUNC} --voxel_size ${VOXEL} \
    --sdf_trunc ${SDF_TRUNC} --num_cluster ${NUM_CLUSTER} \
    > "${MDIR}/mesh.log" 2>&1 \
    && ok=$((ok+1)) \
    || { echo "    fail (tail ${MDIR}/mesh.log)"; tail -5 "${MDIR}/mesh.log"; ng=$((ng+1)); }
done

echo ""
echo "meshed ${ok}, failed ${ng}, skipped ${skip}"
echo "next:  ITER=${IT} OUT=${OBJ} bash run_field_fusion_batch.sh"
