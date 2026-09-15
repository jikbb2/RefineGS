#!/usr/bin/env bash
# Preflight and driver for extending the pipeline to more Replica rooms.
#
# Every room needs five things. Missing SAM3 masks are the usual blocker, and they cannot
# be produced by this script. Run without RUN=1 first: it only prints what each room has.
#
#   bash run_rooms.sh                      # report readiness, change nothing
#   ROOMS="room1 room2" RUN=1 bash run_rooms.sh
#
# Training is the long pole and uses the GPU, so rooms are processed one at a time.
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
ROOMS=${ROOMS:-"room0 room1 room2 room3 room4 room5"}
RUN=${RUN:-0}
ITER=${ITER:-30000}
RES=${RES:-2}                       # -r ; room0 was trained at 2, keep it matched

# Placeholders are @room@ -> room1 and @room_us@ -> room_1 (Replica ships both spellings).
# Braces cannot be used here: inside ${VAR:-default} the first unescaped } ends the
# expansion, so "replica_{room}_v2" silently became "replica_{room_v2}".
DATA_T=${DATA_T:-${ROOT}/data/replica_@room@_v2}
OUTD_T=${OUTD_T:-${ROOT}/output/replica_@room@_v2}
# Replica room0 keeps depth*.png next to frame*.jpg; a separate nice-slam export also
# works. Both spellings are accepted below.
GTD_T=${GTD_T:-${ROOT}/data/replica_@room@_v2/images}
POSE_DIRS=${POSE_DIRS:-"sparse/0 sparse_dense/0"}
GTMESH_T=${GTMESH_T:-$HOME/@room_us@/habitat/mesh_semantic.ply}
GTINFO_T=${GTINFO_T:-$HOME/@room_us@/habitat/info_semantic.json}

sub() {  # sub TEMPLATE ROOM
  local t=$1 r=$2 u="${2/room/room_}"
  t=${t//@room_us@/$u}; t=${t//@room@/$r}; echo "$t"
}

printf "%-8s %-7s %-7s %-7s %-7s %-7s %-7s  %s\n" \
       room images poses masks labels gtdepth gtmesh status
ready=()
for r in ${ROOMS}; do
  D=$(sub "${DATA_T}" "$r"); O=$(sub "${OUTD_T}" "$r")
  G=$(sub "${GTD_T}" "$r"); M=$(sub "${GTMESH_T}" "$r")
  ok=1; cells=(); missing=()
  # poses live in sparse/0 or sparse_dense/0 depending on how the room was built
  POSE=""
  for sd in ${POSE_DIRS}; do [ -d "${D}/${sd}" ] && { POSE=${D}/${sd}; break; }; done
  for pair in "images:${D}/images" "poses:${POSE}" "masks:${D}/masks" \
              "labels:${D}/labels_scene/id_map.json" "gtdepth:${G}" "gtmesh:${M}"; do
    k=${pair%%:*}; p=${pair#*:}
    if [ -n "${p}" ] && [ -e "${p}" ]; then cells+=("yes")
    else cells+=("NO"); ok=0; missing+=("${k}"); fi
  done
  st="ready"
  [ "${ok}" -eq 1 ] || st="missing: ${missing[*]}"
  [ -f "${O}/scene/point_cloud/iteration_${ITER}/point_cloud.ply" ] && st="${st}, trained"
  printf "%-8s %-7s %-7s %-7s %-7s %-7s %-7s  %s\n" "$r" "${cells[@]}" "${st}"
  [ "${ok}" -eq 1 ] && ready+=("$r")
done

echo ""
if [ ${#ready[@]} -eq 0 ]; then
  echo "no room is ready. What each item means:"
  echo "  images  <data>/images          frames, and depth*.png alongside them"
  echo "  poses   <data>/sparse[_dense]/0"
  echo "  masks   <data>/masks/<gid>/masks    per-object SAM3 instance masks"
  echo "  labels  <data>/labels_scene/        make_label_maps.py output"
  echo "  gtmesh  Replica habitat mesh_semantic.ply + info_semantic.json"
  echo ""
  echo "masks and labels cannot be produced here; they need the SAM3 pass per room."
  exit 1
fi
echo "ready: ${ready[*]}"
[ "${RUN}" = "1" ] || { echo "(report only; RUN=1 to execute)"; exit 0; }

for r in "${ready[@]}"; do
  D=$(sub "${DATA_T}" "$r"); O=$(sub "${OUTD_T}" "$r")
  G=$(sub "${GTD_T}" "$r"); M=$(sub "${GTMESH_T}" "$r"); I=$(sub "${GTINFO_T}" "$r")
  S=${O}/scene
  echo ""; echo "################ ${r} ################"
  if [ ! -f "${S}/point_cloud/iteration_${ITER}/point_cloud.ply" ]; then
    # No regularisation: measured on room0, lambda_dist and lambda_normal both raise the
    # depth error at scene level (median 5.5mm -> 23.1 / 32.2) by suppressing densification
    # (628k gaussians -> 533k / 523k). They help per-object, where masking leaves one
    # surface per ray.
    echo "=== train (-r ${RES}, no regularisation) ==="
    (cd "${ROOT}" && python train_scene.py -s "${D}" -m "${S}" -r "${RES}" \
       --label_dir "${D}/labels_scene" --iterations "${ITER}") || { echo "train FAILED"; continue; }
  fi
  POSE=""
  for sd in ${POSE_DIRS}; do [ -d "${D}/${sd}" ] && { POSE=${D}/${sd}; break; }; done
  SCENE_MODEL="${S}" OBJ="${O}/objects_voted" PRIOR="$HOME/prior_${r}" \
  DATA="${D}" COLMAP="${POSE}" GTD="${G}" GT_MESH="${M}" GT_INFO="${I}" \
  LABEL_DIR="${D}/labels_scene" CSV="$HOME/prior/_${r}.csv" ITER="${ITER}" \
    bash run_scene_pipeline.sh || echo "pipeline FAILED for ${r}"
done

echo ""
echo "compare when done:"
echo "  python compare_stages.py --csv $(for r in "${ready[@]}"; do printf '%s=~/prior/_%s.csv ' "$r" "$r"; done)"
