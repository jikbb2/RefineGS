#!/usr/bin/env bash
# RefineGS 최종 전체 파이프라인 (한 방 실행) — ❶..❻ 통합.
#   1) SAM3 re-labeling(+conflation)  [sam3 env]
#   2) amodal 마스크(구멍 메움)         [any]
#   3) per-object 데이터 폴더 + prepare_folder
#   4) per-object 2DGS 학습+메시        [split_and_splat]  (❻ adaptive iters)
#   5) 관측-일관 완성 (+옵션 Amodal3R gated)
#   6) (옵션) scene 합성 / 평가
#
# 사전: sam3 env(SAM3) + split_and_splat env(2DGS), /home/elicer/RefineGS 에서 실행,
#       tools/ 또는 루트에 우리 스크립트들(sam3_relabel.py, amodal_mask.py,
#       amodal_complete_general.py, tools/eval_object_mesh.py).
# 사용:
#   bash run_full_pipeline.sh
#   SCENE=replica_room0 STRIDE=2 bash run_full_pipeline.sh
set -uo pipefail

SCENE=${SCENE:-replica_room0}
ROOT=/home/elicer/RefineGS
FRAMES=${FRAMES:-${ROOT}/data/${SCENE}/images}
IMG_EXT=${IMG_EXT:-.jpg}
SCENE_COLMAP=${SCENE_COLMAP:-${ROOT}/data/${SCENE}/sparse/0}      # scene COLMAP(cameras/images/points3D)
IMG_EXT=${IMG_EXT:-.JPEG}
VOCAB=${VOCAB:-/home/elicer/sam3/vocab.json}
BPE=${BPE:-/home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}
GT_MESH=${GT_MESH:-/home/elicer/room_0/habitat/mesh_semantic.ply}
STRIDE=${STRIDE:-2}
NPF=${NPF:-8}; DEDUP=${DEDUP:-0.3}                                # video relabel: keyframe수, cross-concept 병합
EXCLUDE=${EXCLUDE:-"door,blind,vent,window,wall,floor,ceiling,light switch,thermostat"}
ITERS=${ITERS:-7000}; LDIST=${LDIST:-300}; LNORM=${LNORM:-0.05}   # reg_strong
RELABEL=${RELABEL:-$HOME/relabel_${SCENE}}
AMODAL=${AMODAL:-$HOME/amodal_${SCENE}}
OUT=${OUT:-output/${SCENE}/refinegs_full}
SR=sam3_relabel_video.py; AM=amodal_mask.py; AC=amodal_complete_general.py   # ❶=video predictor 기반

cd ${ROOT}

echo "=== [1] SAM3 video re-labeling (native 추적 + cross-concept dedup) ==="
if [ ! -d "${RELABEL}" ]; then
  LD_LIBRARY_PATH= conda run -n sam3 python ${SR} \
    --frames ${FRAMES} --img_ext ${IMG_EXT} --colmap_dir ${SCENE_COLMAP} \
    --vocab_json ${VOCAB} --bpe ${BPE} --stride ${STRIDE} \
    --n_prompt_frames ${NPF} --min_area 0.003 --min_track 3 --dedup_th ${DEDUP} \
    --exclude_concepts "${EXCLUDE}" --out_root ${RELABEL}
fi
NOBJ=$(ls -d ${RELABEL}/*/ 2>/dev/null | wc -l); echo "re-labeled objects: ${NOBJ}"

echo "=== [2] amodal 마스크(구멍 메움) ==="
conda run -n split_and_splat python ${AM} --in_root ${RELABEL} --out_root ${AMODAL}

echo "=== [3] per-object 데이터 폴더 + prepare_folder ==="
# amodal 마스크를 data/<scene>/masks/<gid>/ 에 배치 → prepare_folder가 images/sparse 복사+masks/ 정리
for D in ${AMODAL}/*/; do
  gid=$(basename "$D"); dst=data/${SCENE}/masks/${gid}
  mkdir -p "${dst}"
  cp ${D}/*.png "${dst}/" 2>/dev/null
  cp ${RELABEL}/${gid}/*.ply "${dst}/" 2>/dev/null   # per-object 초기 포인트(→points3d.ply)
done
bash bash_dir_utils/prepare_folder.sh ${SCENE}

echo "=== [4]+[5] per-object 학습/메시/완성 ==="
mkdir -p ${OUT}
for D in data/${SCENE}/masks/*/; do
  gid=$(basename "$D")
  [ -d "${D}/masks" ] || continue
  NM=$(find "${D}/masks" -iname "*.png" | wc -l)
  # ❻ adaptive: 관측 뷰 적은 객체는 iter↑(데이터 부족 보완), 많으면 기본
  IT=${ITERS}; [ "${NM}" -lt 20 ] && IT=$((ITERS+3000))
  MDIR=${OUT}/${gid}
  if [ ! -f "${MDIR}/point_cloud/iteration_${IT}/point_cloud.ply" ]; then
    echo "  [train] ${gid} (views=${NM}, iters=${IT})"
    conda run -n split_and_splat python train.py -s "${D}" -m "${MDIR}" \
      --iterations ${IT} --is_instance --disable_viewer \
      --lambda_dist ${LDIST} --lambda_normal ${LNORM} >/dev/null 2>&1 \
      || { echo "    train fail ${gid}"; continue; }
  fi
  if [ ! -f "${MDIR}/train/ours_${IT}/fuse_post.ply" ]; then
    conda run -n split_and_splat python render.py -m "${MDIR}" -s "${D}" \
      --iteration ${IT} --skip_test --depth_ratio 1 --depth_trunc 5.0 \
      --voxel_size 0.004 --sdf_trunc 0.02 --num_cluster 1 >/dev/null 2>&1 \
      || { echo "    mesh fail ${gid}"; continue; }
  fi
  # 관측-일관 완성
  conda run -n split_and_splat python ${AC} \
    --recon_ply "${MDIR}/train/ours_${IT}/fuse_post.ply" \
    --out_ply  "${MDIR}/train/ours_${IT}/fuse_completed.ply" \
    --poisson_depth 9 --density_trim 0.1 --support_dist 0.05 >/dev/null 2>&1 || true
done

echo "=== [6] 평가 (recon vs completed, auto-match) ==="
conda run -n split_and_splat python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
  --recon_glob "${OUT}/*/train/ours_*/fuse_post.ply" --label_from_path -4 --auto_match \
  --out ${OUT}/_eval_recon.csv 2>/dev/null || true
conda run -n split_and_splat python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
  --recon_glob "${OUT}/*/train/ours_*/fuse_completed.ply" --label_from_path -4 --auto_match \
  --out ${OUT}/_eval_completed.csv 2>/dev/null || true

echo "=== DONE. 객체:${NOBJ}  결과:${OUT}  평가:_eval_recon.csv / _eval_completed.csv ==="
echo "다음: 두 CSV로 aggregate(recon vs completed) 정리 + scene 합성(run_composition_pipeline)."
