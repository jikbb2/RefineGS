#!/usr/bin/env bash
# 객체별 SDF 메쉬 일괄 추출 (world 좌표로 저장됨 — 정합 불필요, 이후 concat 만)
# 사용: bash run_per_object_sdf.sh "0 1 2 5 6 ..." 7000
set -uo pipefail
GIDS=${1:-"0 1 10 11 12 14 15 16 17 18 19 2 20 22 23 24 27 28 3 31 34 35 36 37 38 4 5 6 7 8"}
IT=${2:-7000}
ROOT=/home/elicer/RefineGS
CARVE=~/carve_depth_mono
for gid in $GIDS; do
  M=output/replica_room0_v2/refinegs_full/$gid
  OUT=$M/train/ours_${IT}/sdf_obj.ply
  [ -f "${OUT%.ply}_post.ply" ] && { echo "[skip] $gid (이미 있음)"; continue; }
  # 객체마다 iters 는 마스크 수로 이미 결정됨(7000 또는 10000) — 실제 존재하는 것 탐색
  RIT=$(ls -d $M/train/ours_* 2>/dev/null | grep -oE 'ours_[0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1)
  [ -z "$RIT" ] && { echo "[skip] $gid (train 없음)"; continue; }
  echo "=== gid $gid (iter $RIT) ==="
  python sdf_distill_depth.py -m $M -s data/replica_room0_v2/masks/$gid \
    --iteration $RIT --depth_ratio 1 --depth_trunc 5.0 --voxel_size 0.004 --num_cluster 1 \
    --alpha_thr 0.8 --mask_dir data/replica_room0_v2/masks/$gid/masks --require_mask \
    --carve_depth_dir $CARVE --mask_dist 0.03 \
    --out $M/train/ours_${RIT}/sdf_obj.ply \
    || echo "  [fail] $gid"
done
echo "=== 완료. 병합은 merge_object_meshes.py 로 ==="
