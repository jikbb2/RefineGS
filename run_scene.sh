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

if [ "${DRY}" -eq 1 ]; then
  echo "(dry) SCENE=${SCENE} TRAJ=${TRAJ} GTD=${GTD} GT_MESH=${GT_MESH} GT_INFO=${GT_INFO} ${ARGS[*]} bash run_refinegs.sh"
  exit 0
fi

cd "${ROOT}" || exit 1
env SCENE="${SCENE}" TRAJ="${TRAJ}" GTD="${GTD}" \
    GT_MESH="${GT_MESH}" GT_INFO="${GT_INFO}" \
    "${ARGS[@]}" bash run_refinegs.sh
