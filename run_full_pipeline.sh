#!/usr/bin/env bash
# RefineGS 최종 파이프라인 — 2-stage (conda run 미사용; env별 수동 activate 후 실행).
# conda run 이 cross-compiler activate 훅(ld.gold)으로 깨지므로, 각 env를 직접 activate.
#
#   STAGE 1 (sam3 env):           re-labeling
#     conda activate sam3
#     SCENE=replica_room0_v2 STRIDE=2 NPF=10 \
#       SCENE_COLMAP=/home/elicer/RefineGS/data/replica_room0/sparse/0 \
#       bash run_full_pipeline.sh relabel
#
#   STAGE 2 (split_and_splat env): amodal→prepare→train→complete→eval
#     conda activate split_and_splat
#     SCENE=replica_room0_v2 \
#       bash run_full_pipeline.sh recon
#
set -uo pipefail
shopt -s nullglob                      # 빈 glob → 루프 skip (literal '*' 폴더 방지)
STAGE=${1:-help}

SCENE=${SCENE:-replica_room0_v2}
ROOT=/home/elicer/RefineGS
FRAMES=${FRAMES:-${ROOT}/data/${SCENE}/images}
IMG_EXT=${IMG_EXT:-.jpg}               # images/엔 .JPEG/.jpg 둘 다; COLMAP은 .jpg
SCENE_COLMAP=${SCENE_COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
VOCAB=${VOCAB:-/home/elicer/sam3/vocab.json}
BPE=${BPE:-/home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}
GT_MESH=${GT_MESH:-/home/elicer/room_0/habitat/mesh_semantic.ply}
STRIDE=${STRIDE:-2}; NPF=${NPF:-10}; DEDUP=${DEDUP:-0.3}
MAX=${MAX:-0}                          # >0이면 관측 많은 상위 MAX개 객체만 학습
EXCLUDE=${EXCLUDE:-"door,blind,vent,window,wall,floor,ceiling,light switch,thermostat"}
ITERS=${ITERS:-7000}; LDIST=${LDIST:-300}; LNORM=${LNORM:-0.05}
RELABEL=${RELABEL:-$HOME/relabel_${SCENE}}
AMODAL=${AMODAL:-$HOME/amodal_${SCENE}}
OUT=${OUT:-output/${SCENE}/refinegs_full}
SR=sam3_relabel_video.py; AM=amodal_mask.py; AC=amodal_complete_general.py
cd ${ROOT}

if [ "${STAGE}" = "relabel" ]; then
  export LD_LIBRARY_PATH=               # sam3: split_and_splat cuDNN 오염 방지
  echo "=== [1] SAM3 video re-labeling (sam3 env) ==="
  python ${SR} \
    --frames ${FRAMES} --img_ext ${IMG_EXT} --colmap_dir ${SCENE_COLMAP} \
    --vocab_json ${VOCAB} --bpe ${BPE} --stride ${STRIDE} \
    --prompt_frame ${PROMPT_FRAME:-0} --min_area 0.003 --min_track 3 \
    --reid_th ${REID:-0.3} --iou_th ${IOU:-0.5} --cand_th ${CAND:-0.1} \
    --exclude_concepts "${EXCLUDE}" --out_root ${RELABEL}
  N=$(ls -d ${RELABEL}/*/ 2>/dev/null | wc -l)
  echo "=== relabel DONE: ${N} objects → ${RELABEL}.  다음: conda activate split_and_splat && bash $0 recon ==="
  exit 0
fi

if [ "${STAGE}" = "recon" ]; then
  [ -d "${RELABEL}" ] || { echo "[ERROR] ${RELABEL} 없음 — 먼저 relabel stage 실행"; exit 1; }
  echo "=== [2] amodal 마스크 ==="
  python ${AM} --in_root ${RELABEL} --out_root ${AMODAL}

  echo "=== [3] per-object 폴더 + prepare_folder ==="
  built=0
  for D in ${AMODAL}/*/; do
    gid=$(basename "$D"); dst=data/${SCENE}/masks/${gid}
    mkdir -p "${dst}"
    cp ${D}*.png "${dst}/" 2>/dev/null
    cp ${RELABEL}/${gid}/*.ply "${dst}/" 2>/dev/null
    built=$((built+1))
  done
  echo "  built ${built} object folders"
  [ "${built}" -gt 0 ] || { echo "[ERROR] 객체 0 — relabel 출력 확인"; exit 1; }
  bash bash_dir_utils/prepare_folder.sh ${SCENE}

  echo "=== [3b] instance 폴더 보정 (images 트림 + scene ply 삭제 → filterPLY 객체 init) ==="
  python setup_instance_folders.py ${SCENE}

  echo "=== [4]+[5] per-object 학습/메시/완성 ==="
  mkdir -p ${OUT}
  # 관측(마스크 수) 내림차순 정렬 → MAX개로 제한 (가장 잘 관측된 객체 우선)
  mapfile -t DIRS < <(for D in data/${SCENE}/masks/*/; do [ -d "${D}masks" ] || continue; \
    n=$(find "${D}masks" -iname "*.png"|wc -l); echo "${n} ${D}"; done | sort -rn | awk '{print $2}')
  [ "${MAX}" -gt 0 ] && DIRS=("${DIRS[@]:0:${MAX}}")
  echo "training ${#DIRS[@]} objects (MAX=${MAX})"
  for D in "${DIRS[@]}"; do
    gid=$(basename "$D"); [ -d "${D}masks" ] || continue
    NM=$(find "${D}masks" -iname "*.png" | wc -l); [ "${NM}" -ge 5 ] || continue
    IT=${ITERS}; [ "${NM}" -lt 20 ] && IT=$((ITERS+3000))      # ❻ adaptive
    MDIR=${OUT}/${gid}
    if [ ! -f "${MDIR}/point_cloud/iteration_${IT}/point_cloud.ply" ]; then
      echo "  [train] ${gid} (views=${NM}, iters=${IT})"
      python train.py -s "${D}" -m "${MDIR}" --iterations ${IT} --is_instance \
        --disable_viewer --lambda_dist ${LDIST} --lambda_normal ${LNORM} \
        || { echo "    train fail ${gid}"; continue; }
    fi
    [ -f "${MDIR}/train/ours_${IT}/fuse_post.ply" ] || \
      python render.py -m "${MDIR}" -s "${D}" --iteration ${IT} --skip_test \
        --depth_ratio 1 --depth_trunc 5.0 --voxel_size 0.004 --sdf_trunc 0.02 --num_cluster 1 \
        || { echo "    mesh fail ${gid}"; continue; }
    python ${AC} --recon_ply "${MDIR}/train/ours_${IT}/fuse_post.ply" \
      --out_ply "${MDIR}/train/ours_${IT}/fuse_completed.ply" \
      --poisson_depth 9 --density_trim 0.1 --support_dist 0.05 2>/dev/null || true
  done

  echo "=== [6] 평가 (recon vs completed) ==="
  python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
    --recon_glob "${OUT}/*/train/ours_*/fuse_post.ply" --label_from_path -4 --auto_match \
    --out ${OUT}/_eval_recon.csv 2>/dev/null || true
  python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
    --recon_glob "${OUT}/*/train/ours_*/fuse_completed.ply" --label_from_path -4 --auto_match \
    --out ${OUT}/_eval_completed.csv 2>/dev/null || true
  echo "=== DONE. 결과:${OUT}  평가 CSV 2개 ==="
  exit 0
fi

echo "Usage:"
echo "  conda activate sam3            && bash $0 relabel"
echo "  conda activate split_and_splat && bash $0 recon"
