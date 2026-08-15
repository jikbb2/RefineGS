#!/usr/bin/env bash
# 전체 객체 prior-fusion 배치: fuse(정합) → sdf_distill(--prior_mesh) → 평가.
# TRELLIS 생성 glb(obj{gid}.glb)는 미리 준비돼 있어야 함(GEN_DIR). 없으면 해당 객체 skip.
#
#   conda activate split_and_splat
#   GEN_DIR=~/gen_out bash run_prior_fusion_batch.sh
#
# 객체별 probe_box 는 정합 glb bbox 에서 자동 계산. shell/carve flag 는 obj6 v8 튜닝값 사용.
set -uo pipefail
shopt -s nullglob

ROOT=${ROOT:-/home/elicer/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
GEN_DIR=${GEN_DIR:-$HOME/gen_out}                 # obj{gid}.glb 위치
STEMS_DIR=${STEMS_DIR:-$HOME/See3D/dataset/stage6/clean_stems}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-/home/elicer/room_0/habitat/mesh_semantic.ply}
PRIOR=${PRIOR:-$HOME/prior}                       # 정합 glb / 점군 저장
ITER=${ITER:-7000}
MARGIN=${MARGIN:-0.05}                            # probe_box 여유(m)
# sdf_distill 공통 flag (obj6 v8)
SDF_FLAGS=${SDF_FLAGS:-"--grid_fuse --shell_delta 0.024 --shell_delta_min 0.006 --shell_ramp 0.10 --prior_carve_views 150 --free_min_views 2 --num_cluster 10000 --voxel_size 0.005 --max_grid 512"}

cd ${ROOT}
mkdir -p ${PRIOR}
done_n=0; skip_n=0

# fuse 정합 품질 로그(객체별 IoU/RMSE/graft) — FUSE_LOG=0 이면 비활성
FUSE_LOG=${FUSE_LOG:-1}
LOGDIR=${LOGDIR:-${PRIOR}/fuse_logs}
IOU_CSV=${IOU_CSV:-${OUT}/_fuse_iou.csv}
if [ "${FUSE_LOG}" = "1" ]; then
  mkdir -p "${LOGDIR}"
  echo "gid,scale,rmse_mm,iou_rc_before,iou_rc_after,iou_final,graft_pct" > "${IOU_CSV}"
fi
for MDIR in ${OUT}/*/; do
  gid=$(basename "${MDIR}")
  [[ "${gid}" =~ ^[0-9]+$ ]] || continue
  RECON=${MDIR}train/ours_${ITER}/fuse_post.ply
  GLB=${GEN_DIR}/obj${gid}.glb
  STEMS=${STEMS_DIR}/${gid}.txt
  [ -f "${RECON}" ] || { echo "[skip] ${gid}: recon 없음"; skip_n=$((skip_n+1)); continue; }
  [ -f "${GLB}" ]   || { echo "[skip] ${gid}: 생성 glb 없음(${GLB})"; skip_n=$((skip_n+1)); continue; }
  [ -f "${STEMS}" ] || STEMS=""                    # stems 없으면 전체 뷰 사용

  echo "=== [${gid}] fuse 정합 + prior 추출 ==="
  ALIGNED=${PRIOR}/obj${gid}_gen_aligned.glb
  FLOG=${LOGDIR:-/tmp}/fuse_obj${gid}.log
  python fuse_generated_mesh.py \
    --recon "${RECON}" --gen "${GLB}" --gen_up y --world_up z --isotropic --refine \
    --colmap "${COLMAP}" --gid ${gid} --masks_root "${MASKS}" \
    ${STEMS:+--stems "${STEMS}"} \
    --export_points "${PRIOR}/obj${gid}_unseen.ply" --save_aligned --no_mesh \
    2>&1 | tee "${FLOG}"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "  fuse 실패 ${gid}"; skip_n=$((skip_n+1)); continue; }

  # 정합 품질 파싱 → CSV 한 줄
  if [ "${FUSE_LOG}" = "1" ]; then
    _scale=$(grep -oP 'scale=\(\K[0-9.]+' "${FLOG}" | tail -1)
    _rmse=$(grep -oP 'trimmed-RMSE=\K[0-9.]+' "${FLOG}" | tail -1)
    _rcln=$(grep 'render-compare' "${FLOG}" | tail -1)
    _rcb=$(echo "${_rcln}" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    _rca=$(echo "${_rcln}" | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    _iouf=$(grep -oP 'silhouette IoU=\K[0-9.]+' "${FLOG}" | tail -1)
    _graft=$(grep -oP '\(\K[0-9]+(?=%\))' "${FLOG}" | tail -1)
    echo "${gid},${_scale:-},${_rmse:-},${_rcb:-},${_rca:-},${_iouf:-},${_graft:-}" >> "${IOU_CSV}"
  fi
  # fuse 는 <export_points>_gen_aligned.glb 로 저장 → 경로 맞추기
  ALIGNED=${PRIOR}/obj${gid}_unseen_gen_aligned.glb
  [ -f "${ALIGNED}" ] || { echo "  정합 glb 없음 ${gid}"; skip_n=$((skip_n+1)); continue; }

  # probe_box 자동: 정합 glb bbox(다리 포함) + 여유
  BOX=$(python - "$ALIGNED" "$MARGIN" <<'PY'
import sys, trimesh, numpy as np
m = trimesh.load(sys.argv[1], process=False, force="mesh")
mg = float(sys.argv[2]); b = m.bounds
lo = b[0]-mg; hi = b[1]+mg
print("%.5f,%.5f,%.5f,%.5f,%.5f,%.5f" % (lo[0],lo[1],lo[2],hi[0],hi[1],hi[2]))
PY
)
  echo "  probe_box=${BOX}"

  echo "=== [${gid}] sdf_distill (--prior_mesh) ==="
  python sdf_distill_depth.py -m "${MDIR%/}" --iteration ${ITER} \
    --mask_dir auto --require_mask --mask_dist 0 \
    --prior_mesh "${ALIGNED}" ${SDF_FLAGS} \
    --gt_depth_dir "${GTD}" --probe_box "${BOX}" \
    --out "${MDIR}train/ours_${ITER}/fused_prior.ply" \
    || { echo "  sdf 실패 ${gid}"; skip_n=$((skip_n+1)); continue; }
  done_n=$((done_n+1))
  echo "=== [${gid}] 완료 → ${MDIR}train/ours_${ITER}/fused_prior.ply ==="
done

echo ""
echo "배치 완료: 성공 ${done_n}, skip ${skip_n}"

if [ "${FUSE_LOG}" = "1" ] && [ -f "${IOU_CSV}" ]; then
  echo "=== 정합 품질 요약 (IoU 오름차순 — 낮은 객체가 정합 부실) ==="
  { head -1 "${IOU_CSV}"; tail -n +2 "${IOU_CSV}" | sort -t, -k6 -n; } | column -t -s,
  echo "  → ${IOU_CSV} (객체별 로그: ${LOGDIR}/fuse_obj*.log)"
fi

echo "=== 평가: baseline(recon) vs prior-fused ==="
python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
  --recon_glob "${OUT}/*/train/ours_${ITER}/fuse_post.ply" --label_from_path -4 --auto_match \
  --out ${OUT}/_eval_recon.csv 2>/dev/null || echo "  recon 평가 skip"
python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
  --recon_glob "${OUT}/*/train/ours_${ITER}/fused_prior.ply" --label_from_path -4 --auto_match \
  --out ${OUT}/_eval_prior.csv 2>/dev/null || echo "  prior 평가 skip"
echo "=== 평가 CSV: ${OUT}/_eval_recon.csv , _eval_prior.csv ==="
