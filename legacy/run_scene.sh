#!/usr/bin/env bash
# One Replica scene, by name -- resolves every path, then hands off to run_refinegs.sh.
#
#   bash run_scene.sh room2                        # whole pipeline
#   bash run_scene.sh office0 FROM=train TO=name   # resume a span
#   bash run_scene.sh room2 --dry                  # print the resolved paths, run nothing
#
# Why this exists: run_refinegs.sh needs five paths to agree (SCENE, TRAJ, GTD, GT_MESH,
# GT_INFO), and they come from two directory trees with different naming -- nice-slam writes
# `room1`, the Replica v1 tarball writes `room_1`. room0 sits in a third place again
# (~/room_0), because it was extracted before ~/replica_dl existed. Typing those five by hand
# per scene is how a run ends up evaluating one room's reconstruction against another room's
# GT mesh: nothing downstream checks, and the numbers look plausible.
#
# Everything after the scene name is passed through untouched, so every run_refinegs.sh
# variable (FROM, TO, ONLY, RUN, PIPELINE, CLEAN, TRAIN_SCENE_ARGS, ...) still works.
set -uo pipefail

SCENE_NAME=${1:-}
[ -n "${SCENE_NAME}" ] || { echo "usage: bash run_scene.sh <room0|room1|room2|office0..office4> [VAR=val ...]"; exit 1; }
shift

ROOT=${ROOT:-$HOME/RefineGS}
NICE=${NICE:-$HOME/nice-slam/Datasets/Replica}
DL=${DL:-$HOME/replica_dl}

# nice-slam name -> Replica v1 directory name: room2 -> room_2, office0 -> office_0.
V1_NAME=$(echo "${SCENE_NAME}" | sed -E 's/^(room|office)([0-9]+)$/\1_\2/')

# The habitat/ tree moved twice. Take the first one that actually exists rather than
# encoding a rule that only holds for the scenes already run.
HAB=""
for cand in "${DL}/${V1_NAME}/habitat" "$HOME/${V1_NAME}/habitat" "${NICE}/${SCENE_NAME}/habitat"; do
  [ -f "${cand}/mesh_semantic.ply" ] && { HAB=${cand}; break; }
done

SCENE=${SCENE:-replica_${SCENE_NAME}_v2}
TRAJ=${TRAJ:-${NICE}/${SCENE_NAME}/traj.txt}
GTD=${GTD:-${NICE}/${SCENE_NAME}/results}
GT_MESH=${GT_MESH:-${HAB}/mesh_semantic.ply}
GT_INFO=${GT_INFO:-${HAB}/info_semantic.json}
DATA=${DATA:-${ROOT}/data/${SCENE}}

DRY=0
ARGS=()
for a in "$@"; do
  case "${a}" in --dry) DRY=1 ;; *) ARGS+=("${a}") ;; esac
done

echo "scene      ${SCENE_NAME}  ->  ${SCENE}"
echo "habitat    ${HAB:-<NOT FOUND>}"
for p in "${TRAJ}" "${GTD}" "${GT_MESH}" "${GT_INFO}"; do
  [ -e "${p}" ] && echo "  ok       ${p}" || echo "  MISSING  ${p}"
done

miss=0
for p in "${TRAJ}" "${GTD}" "${GT_MESH}" "${GT_INFO}"; do [ -e "${p}" ] || miss=1; done
[ "${miss}" -eq 0 ] || { echo "[abort] resolve the paths above before running"; exit 1; }

# Stage 0 is not part of run_refinegs.sh: it turns nice-slam's traj.txt + results/ into the
# RefineGS layout. Done once per scene, and skipped once data/<scene>/sparse/0 exists so a
# resumed run never rebuilds the pose set underneath an already-trained model.
if [ ! -d "${DATA}/sparse/0" ]; then
  echo "[stage0] ${DATA} does not exist yet -- converting"
  [ "${DRY}" -eq 1 ] && echo "  (dry) python tools/replica_to_refinegs.py --replica_scene ${NICE}/${SCENE_NAME} --out_dir data/${SCENE} --subsample 10" || {
    cd "${ROOT}" || exit 1
    python tools/replica_to_refinegs.py \
      --replica_scene "${NICE}/${SCENE_NAME}" \
      --out_dir "data/${SCENE}" --subsample 10 || exit 1
  }
else
  echo "[stage0] ${DATA}/sparse/0 exists -- skipping conversion"
fi

# The step that was missing from every script, and the reason room2's first run reached SAM3
# with 200 views and produced zero native tracks.
#
# replica_to_refinegs.py applies --subsample to the image symlinks as well as the poses (its
# own header says so), so a fresh scene has 200 links. make_dense_colmap.py then globs that
# directory and writes one pose per file it finds -- it creates no links of its own. The lift
# therefore produces 200 poses, the colmap stage's "poses < frames" test is false, it prints
# "up to date", and nothing anywhere reports that the run is using a tenth of the trajectory.
#
# room0 and room1 got their 2000 links from a command typed by hand between stage 0 and the
# colmap stage: `ln -sfn <scene>/results/* .` inside images/. That is why their manifests read
# "n=2000 .jpg of 4000 files" -- results/ holds 2000 frame*.jpg AND 2000 depth*.png, and the
# glob took both.
#
# This links results/* rather than just *.jpg on purpose. run_refinegs.sh counts only .jpg and
# make_dense_colmap.py globs only .jpg, so the depth PNGs change nothing there -- but whether
# some loader downstream globs images/ more broadly has not been verified, and room0 and room1
# were built and trained with those PNGs present. Reproducing the directory they actually had
# costs nothing; deviating from it would put an unchecked difference underneath a table that
# compares the scenes to each other.
NT=$([ -f "${TRAJ}" ] && awk 'NF{n++} END{print n+0}' "${TRAJ}" || echo 0)
count_jpg() { ls "${DATA}/images"/*"${IMG_EXT:-.jpg}" 2>/dev/null | wc -l; }
NF=$(count_jpg)
echo "[frames] linked ${NF} ${IMG_EXT:-.jpg} / trajectory ${NT}"
if [ "${NT}" -gt 0 ] && [ "${NF}" -lt "${NT}" ]; then
  if [ "${DRY}" -eq 1 ]; then
    echo "  (dry) would link ${GTD}/* into ${DATA}/images  (+$((NT - NF)) frames)"
  else
    echo "  linking ${GTD}/* into images/  (frames and depth, as room0/room1 have)"
    mkdir -p "${DATA}/images"
    # -f so re-running is idempotent: the 200 links stage 0 made point at the same files.
    ln -sfn "${GTD}"/* "${DATA}/images/" || exit 1
    NF=$(count_jpg)
    echo "  linked now ${NF} ${IMG_EXT:-.jpg}, $(ls "${DATA}/images" | wc -l) files total"
    [ "${NF}" -ge "${NT}" ] || {
      echo "[abort] ${NF} frames for a ${NT}-frame trajectory -- names in ${GTD} may not end in ${IMG_EXT:-.jpg}"
      exit 1
    }
  fi
fi

if [ "${DRY}" -eq 1 ]; then
  echo "(dry) SCENE=${SCENE} TRAJ=${TRAJ} GTD=${GTD} GT_MESH=${GT_MESH} GT_INFO=${GT_INFO} ${ARGS[*]} bash run_refinegs.sh"
  exit 0
fi

cd "${ROOT}" || exit 1
env SCENE="${SCENE}" TRAJ="${TRAJ}" GTD="${GTD}" \
    GT_MESH="${GT_MESH}" GT_INFO="${GT_INFO}" \
    "${ARGS[@]}" bash run_refinegs.sh