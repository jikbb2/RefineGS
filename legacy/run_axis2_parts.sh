#!/usr/bin/env bash
# 축2 end-to-end — granularity(part) 마스크로 per-object 2DGS 재학습 → 병합 → GT 비교.
# 각 part: 98 폴더 복제 + masks만 part 마스크로 교체 → train(reg_strong) → fuse_post.
# 그 다음 part mesh 병합 → merged-parts vs 기존 whole-98 을 gt_id 로 평가.
#
# 사전: conda activate split_and_splat, /home/elicer/RefineGS 에서 실행.
#        axis2_emit_masks.py 출력(~/axis2_masks_98/part*/*.png) 존재.
# 사용:
#   bash run_axis2_parts.sh
#   OBJ=98 GT_ID=7 PARTS_DIR=~/axis2_masks_98 bash run_axis2_parts.sh
set -uo pipefail
SCENE=replica_room0
OBJ=${OBJ:-98}
GT_ID=${GT_ID:-7}
PARTS_DIR=${PARTS_DIR:-$HOME/axis2_masks_${OBJ}}
DATA_ROOT=data/${SCENE}/masks
SRC=${DATA_ROOT}/${OBJ}
OUT=output/${SCENE}/axis2_parts
GT_MESH=${GT_MESH:-data/replica_gt/room_0/habitat/mesh_semantic.ply}
ITERS=7000; LDIST=300; LNORM=0.05      # reg_strong (run_axis3_reg_sweep와 동일)

NPARTS=$(ls -d ${PARTS_DIR}/part*/ 2>/dev/null | wc -l)
echo "parts=${NPARTS}  OBJ=${OBJ}  GT_ID=${GT_ID}"
[ "${NPARTS}" -lt 1 ] && { echo "part 마스크 없음: ${PARTS_DIR}"; exit 1; }

for k in $(seq 0 $((NPARTS-1))); do
  PID=${OBJ}_part${k}; PDIR=${DATA_ROOT}/${PID}; MDIR=${OUT}/${PID}
  # 1) per-part 입력 폴더 (98 복제 + masks 교체)
  if [ ! -d "${PDIR}" ]; then
    echo "=== build ${PID} ==="
    mkdir -p ${PDIR}/masks
    cp -r ${SRC}/images ${PDIR}/ 2>/dev/null
    cp -r ${SRC}/sparse ${PDIR}/ 2>/dev/null
    cp ${SRC}/points3d.ply ${PDIR}/ 2>/dev/null
    cp ${PARTS_DIR}/part${k}/*.png ${PDIR}/masks/ 2>/dev/null
  fi
  # 2) train (reg_strong)
  if [ ! -f "${MDIR}/point_cloud/iteration_${ITERS}/point_cloud.ply" ]; then
    echo "=== train ${PID} ==="
    python train.py -s ${PDIR} -m ${MDIR} --iterations ${ITERS} --is_instance \
      --disable_viewer --lambda_dist ${LDIST} --lambda_normal ${LNORM} \
      || { echo "TRAIN FAIL ${PID}"; continue; }
  fi
  # 3) mesh
  if [ ! -f "${MDIR}/train/ours_${ITERS}/fuse_post.ply" ]; then
    echo "=== mesh ${PID} ==="
    python render.py -m ${MDIR} -s ${PDIR} --iteration ${ITERS} --skip_test \
      --depth_ratio 1 --depth_trunc 5.0 --voxel_size 0.004 --sdf_trunc 0.02 --num_cluster 1 \
      || { echo "MESH FAIL ${PID}"; continue; }
  fi
done

# 4) part mesh 병합
echo "=== merge parts ==="
python - <<EOF
import trimesh, glob
paths = sorted(glob.glob("${OUT}/${OBJ}_part*/train/ours_${ITERS}/fuse_post.ply"))
print("merging:", paths)
ms = [trimesh.load(p, process=False) for p in paths if p]
if not ms:
    print("no part meshes"); raise SystemExit(1)
m = trimesh.util.concatenate(ms)
m.export("${OUT}/merged_${OBJ}.ply")
print("merged ->", "${OUT}/merged_${OBJ}.ply", "verts", len(m.vertices))
EOF

# 5) 평가: merged-parts vs 기존 whole-98 (둘 다 GT gt_id 로)
echo "=== eval merged-parts ==="
python tools/eval_object_mesh.py eval --gt_mesh ${GT_MESH} --gt_id ${GT_ID} \
  --recon ${OUT}/merged_${OBJ}.ply --out $HOME/eval_axis2_merged_${OBJ}.json
echo "=== eval whole baseline (axis3_sweep/reg_strong/${OBJ}) ==="
python tools/eval_object_mesh.py eval --gt_mesh ${GT_MESH} --gt_id ${GT_ID} \
  --recon output/${SCENE}/axis3_sweep/reg_strong/${OBJ}/train/ours_${ITERS}/fuse_post.ply \
  --out $HOME/eval_axis2_whole_${OBJ}.json

echo "================ COMPARE (whole vs merged-parts) ================"
echo "--- whole ---";  cat $HOME/eval_axis2_whole_${OBJ}.json
echo; echo "--- merged-parts ---"; cat $HOME/eval_axis2_merged_${OBJ}.json
