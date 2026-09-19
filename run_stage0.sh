#!/usr/bin/env bash
# Stage 0: raw scene -> everything run_scene_pipeline.sh's [check] block requires.
# Extracted from run_full_pipeline.sh, which also trains and Poisson-completes per object;
# that path is the per-object experiment, not an input to the scene pipeline.
#
# Two conda envs, so two invocations (conda run breaks on the cross-compiler activate hook):
#
#   conda activate sam3            && SCENE=<scene> bash run_stage0.sh relabel
#   conda activate split_and_splat && SCENE=<scene> bash run_stage0.sh masks
#   conda activate split_and_splat && SCENE=<scene> bash run_stage0.sh labels
#   conda activate split_and_splat && SCENE=<scene> bash run_stage0.sh check
#
# 'all' runs masks+labels+check in the current env; relabel always stands alone.
#
# Every stage is idempotent. Existing per-gid folders are rebuilt in place, so a rerun after
# adding frames is safe.
set -uo pipefail
shopt -s nullglob

STAGE=${1:-help}
ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
DATA=${DATA:-${ROOT}/data/${SCENE}}
FRAMES=${FRAMES:-${DATA}/images}
IMG_EXT=${IMG_EXT:-.jpg}               # images/ holds .JPEG and .jpg both; COLMAP uses .jpg
# One pose source for the whole project. run_full_pipeline.sh pointed relabel at a
# different scene folder than the fusion and evaluation read, which would silently label
# masks against poses nothing downstream uses. Detected the same way as the scene pipeline.
if [ -z "${COLMAP:-}" ]; then
  for sd in sparse/0 sparse_dense/0; do
    [ -d "${DATA}/${sd}" ] && { COLMAP=${DATA}/${sd}; break; }
  done
fi
COLMAP=${COLMAP:-${DATA}/sparse/0}
MASKS=${MASKS:-${DATA}/masks}
LABEL_DIR=${LABEL_DIR:-${DATA}/labels_scene}
RELABEL=${RELABEL:-$HOME/relabel_${SCENE}}
AMODAL=${AMODAL:-$HOME/amodal_${SCENE}}
VOCAB=${VOCAB:-$HOME/sam3/vocab.json}
BPE=${BPE:-$HOME/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
# Structure and background carry no instance identity and would dominate the vocabulary.
EXCLUDE=${EXCLUDE:-"door,blind,vent,window,wall,floor,ceiling,light switch,thermostat"}
STRIDE=${STRIDE:-2}                    # frame subsample; halves every object's view count
WINDOW=${WINDOW:-200}
PROMPT_FRAME=${PROMPT_FRAME:-0}
MIN_AREA=${MIN_AREA:-0.003}            # drops small objects entirely: raise coverage first
MIN_TRACK=${MIN_TRACK:-3}
REID=${REID:-0.3}; IOU=${IOU:-0.5}; CAND=${CAND:-0.1}
# make_label_maps.py's own default. It is the hardest object filter in the whole project:
# a gid seen in fewer views gets no label, so extract_objects.py never makes a dir for it
# and the scene pipeline cannot report it at all (measured room0: 12 of 48 gids). Lower it
# only together with a decision about how those objects appear in the table.
MIN_LABEL_VIEWS=${MIN_LABEL_VIEWS:-30}
OVERLAP=${OVERLAP:-ignore}             # contested pixels: teach nothing rather than guess

cd "${ROOT}" || exit 1

[ "${STAGE}" = "all" ] && echo "[note] relabel is not part of 'all' -- it needs the sam3 env"

if [ "${STAGE}" = "relabel" ]; then
  export LD_LIBRARY_PATH=              # split_and_splat cuDNN leaks into sam3 otherwise
  echo "=== [1] SAM3 video re-labeling ==="
  echo "  frames=${FRAMES}  colmap=${COLMAP}  stride=${STRIDE}"
  python sam3_relabel_video.py \
    --frames "${FRAMES}" --img_ext "${IMG_EXT}" --colmap_dir "${COLMAP}" \
    --vocab_json "${VOCAB}" --bpe "${BPE}" --stride "${STRIDE}" --window "${WINDOW}" \
    --prompt_frame "${PROMPT_FRAME}" --min_area "${MIN_AREA}" --min_track "${MIN_TRACK}" \
    --reid_th "${REID}" --iou_th "${IOU}" --cand_th "${CAND}" \
    --exclude_concepts "${EXCLUDE}" --out_root "${RELABEL}" \
    || { echo "[ERROR] relabel FAILED"; exit 1; }
  N=$(ls -d "${RELABEL}"/*/ 2>/dev/null | wc -l)
  echo "=== relabel done: ${N} objects -> ${RELABEL}"
  echo "    next: conda activate split_and_splat && SCENE=${SCENE} bash $0 masks"
  exit 0
fi

if [ "${STAGE}" = "masks" ] || [ "${STAGE}" = "all" ]; then
  [ -d "${RELABEL}" ] || { echo "[ERROR] ${RELABEL} missing -- run the relabel stage"; exit 1; }
  echo "=== [2] amodal masks ==="
  python amodal_mask.py --in_root "${RELABEL}" --out_root "${AMODAL}" || exit 1

  echo "=== [3] per-object folders ==="
  built=0
  for D in "${AMODAL}"/*/; do
    gid=$(basename "${D}"); dst=${MASKS}/${gid}
    mkdir -p "${dst}"
    cp "${D}"*.png "${dst}/" 2>/dev/null
    # the init point cloud is the COLMAP sparse filtered by this object's mask, and only
    # the relabel stage writes it
    cp "${RELABEL}/${gid}"/*.ply "${dst}/" 2>/dev/null
    built=$((built+1))
  done
  echo "  built ${built} object folders under ${MASKS}"
  [ "${built}" -gt 0 ] || { echo "[ERROR] 0 objects -- check the relabel output"; exit 1; }

  # moves the png into <gid>/masks/, copies sparse/, renames the ply to points3d.ply,
  # and discards objects with fewer than 2 masks
  bash bash_dir_utils/prepare_folder.sh "${SCENE}" || exit 1

  echo "=== [3b] instance folders (mask frames only + GT depth + object init) ==="
  python setup_instance_folders.py "${SCENE}" || exit 1
fi

if [ "${STAGE}" = "labels" ] || [ "${STAGE}" = "all" ]; then
  echo "=== [4] scene label maps -> ${LABEL_DIR}/id_map.json ==="
  # run_scene_pipeline.sh requires id_map.json and does not create it. The GT-raycast
  # alternative is make_gt_masks.py, which writes labels_scene directly (control runs);
  # the two must not share a vote dir.
  if [ -f "${LABEL_DIR}/id_map.json" ]; then
    echo "  exists, skipping (delete it to rebuild)"
  else
    python make_label_maps.py --masks_root "${MASKS}" --out "${LABEL_DIR}" \
      --min_views "${MIN_LABEL_VIEWS}" --overlap "${OVERLAP}" || exit 1
  fi
fi

if [ "${STAGE}" = "check" ] || [ "${STAGE}" = "all" ]; then
  echo ""; echo "=== [5] contract check (what run_scene_pipeline.sh requires) ==="
  fail=0
  for p in "${COLMAP}" "${FRAMES}" "${MASKS}" "${LABEL_DIR}/id_map.json" "${GTD}"; do
    [ -e "${p}" ] && echo "  ok      ${p}" || { echo "  MISSING ${p}"; fail=1; }
  done

  # Objects disappear at four places before the fusion ever sees them, and none of them
  # leaves a line in the fusion log. Print the funnel so the evaluation can state its
  # denominator instead of silently reporting whatever survived.
  echo ""
  n_rel=$(ls -d "${RELABEL}"/*/ 2>/dev/null | wc -l)
  n_amo=$(ls -d "${AMODAL}"/*/ 2>/dev/null | wc -l)
  n_dis=$(ls -d "${DATA}/discard"/*/ 2>/dev/null | wc -l)
  n_obj=$(ls -d "${MASKS}"/*/ 2>/dev/null | wc -l)
  echo "  objects: relabel ${n_rel} -> amodal ${n_amo} -> prepared ${n_obj} (discarded ${n_dis})"

  # The label stage is where most objects are lost, and it happens before the pipeline
  # starts: a gid without a label gets no dir from extract_objects.py, so the scene run
  # cannot report it even as a failure.
  if [ -f "${LABEL_DIR}/id_map.json" ]; then
    python - "${LABEL_DIR}/id_map.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
rare = m.get("rare_gids", [])
print(f"  labels:  {m['K']} kept, {len(rare)} dropped below min_views "
      f"{m.get('min_views')} -> {rare}")
print(f"           overlap policy {m.get('overlap')}")
PY
  fi

  # grid_wcap is 8: alpha = views/8, so an object below 8 masked frames never reaches
  # alpha = 1 and the prior keeps a share of its OBSERVED surface. min_views above already
  # rules this out for the scene path; it can still bite the per-object pipeline, which
  # trains from 5 masks.
  echo ""
  lo=0
  for D in "${MASKS}"/*/; do
    gid=$(basename "${D}"); [ -d "${D}masks" ] || continue
    n=$(find "${D}masks" -iname "*.png" | wc -l)
    [ "${n}" -lt 8 ] && { printf "    gid %-5s views %-4s alpha_max %.2f\n" \
      "${gid}" "${n}" "$(echo "${n}" | awk '{print $1/8}')"; lo=$((lo+1)); }
  done
  [ "${lo}" -eq 0 ] && echo "  views: every object has >= 8 (grid_wcap saturates)" \
                    || echo "  views: ${lo} object(s) below grid_wcap 8 -- prior mixes into their observed surface"

  n_img=$(ls "${FRAMES}"/*"${IMG_EXT}" 2>/dev/null | wc -l)
  echo ""
  echo "  frames ${n_img} (stride ${STRIDE} caps any object at about $((n_img / STRIDE)) views)"
  [ "${fail}" -eq 0 ] && echo "" && echo "  -> ready: bash run_scene_pipeline.sh" || exit 1
fi

case "${STAGE}" in
  relabel|masks|labels|check|all) ;;
  *)
    echo "Usage: SCENE=<scene> bash $0 relabel|masks|labels|check|all"
    echo "  relabel  sam3 env            SAM3 video re-labeling -> ${RELABEL}"
    echo "  masks    split_and_splat env amodal + per-object folders -> ${MASKS}"
    echo "  labels   split_and_splat env make_label_maps.py -> ${LABEL_DIR}/id_map.json"
    echo "  check    split_and_splat env verify the scene-pipeline contract"
    ;;
esac
