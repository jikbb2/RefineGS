#!/usr/bin/env bash
# obj1 단일 객체 — RC depth 항 검증 사이클 (fuse → sdf → seen/unseen 평가)
#   bash run_obj1_rcdepth.sh
# 기존 fused_prior.ply 는 보존하고 _rcd 접미사로 저장해 이전 수치와 비교 가능.
set -euo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
GID=${GID:-1}
ITER=${ITER:-7000}
SCENE=${SCENE:-replica_room0_v2}
MDIR=${ROOT}/output/${SCENE}/refinegs_full/${GID}
OUTD=${MDIR}/train/ours_${ITER}
PRIOR=${PRIOR:-$HOME/prior}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
STEMS=${STEMS:-$HOME/See3D/dataset/stage6/clean_stems/${GID}.txt}

cd "${ROOT}"
mkdir -p "${PRIOR}"

echo "=== [1/3] fuse 정합 (RC depth 항 포함) ==="
python fuse_generated_mesh.py \
  --recon "${OUTD}/fuse_post.ply" --gen "${HOME}/gen_out/obj${GID}.glb" \
  --gen_up y --world_up z --isotropic --refine \
  --init_all_up --band_warp --min_iou 0.45 \
  --rc_depth_w 1.0 --rc_depth_cap 0.05 --rc_views 8 \
  --colmap data/${SCENE}/sparse/0 --gid ${GID} \
  --masks_root data/${SCENE}/masks \
  ${STEMS:+--stems "${STEMS}"} \
  --export_points "${PRIOR}/obj${GID}_rcd.ply" --save_aligned --no_mesh --overwrite

ALIGNED=${PRIOR}/obj${GID}_rcd_gen_aligned.glb
[ -f "${ALIGNED}" ] || { echo "정합 glb 없음: ${ALIGNED}"; exit 1; }

echo "=== [2/3] sdf_distill (grid_fuse) ==="
python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
  --data_device cpu --mask_dir auto --require_mask --mask_dist 0 \
  --prior_mesh "${ALIGNED}" \
  --grid_fuse --shell_delta 0.024 --shell_delta_min 0.006 --shell_ramp 0.10 \
  --prior_carve_views 150 --free_min_views 2 --num_cluster 10000 \
  --voxel_size 0.005 --max_grid 512 --keep_connected \
  --gt_depth_dir "${GTD}" \
  --out "${OUTD}/fused_prior_rcd.ply"

echo "=== [3/3] seen/unseen 평가 ==="
EVAL_ARGS=(--gt_mesh "${GT_MESH}"
           --recon "${OUTD}/fuse_post.ply" --recon2 "${OUTD}/fused_prior_rcd.ply"
           --colmap data/${SCENE}/sparse/0 --gid ${GID}
           --masks_root data/${SCENE}/masks --use_mask)
[ -n "${STEMS}" ] && EVAL_ARGS+=(--stems "${STEMS}")

echo "--- (a) 자동 매칭 ---"
python eval_seen_unseen.py "${EVAL_ARGS[@]}" | tee "${OUTD}/_eval_su_auto.txt"

echo "--- (b) 다중 라벨 대조(GT 매칭 아티팩트 분리) ---"
python eval_seen_unseen.py "${EVAL_ARGS[@]}" --gt_labels 9,70,18,71,8 \
  | tee "${OUTD}/_eval_su_multi.txt"

echo ""
echo "완료. 비교 포인트:"
echo "  1) fuse 로그의 'render-compare IoU ... depth A → B mm'"
echo "  2) (a) 의 unseen accuracy 93.16mm 대비 감소폭  ← RC depth 효과"
echo "  3) (a) vs (b) 차이                              ← GT 라벨 아티팩트 크기"
echo "  결과: ${OUTD}/_eval_su_auto.txt , _eval_su_multi.txt"
