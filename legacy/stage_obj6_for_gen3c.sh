#!/usr/bin/env bash
# obj6 GEN3C 테스트에 필요한 데이터만 staging 후 40GB 서버로 rsync.
# 원본(데이터 보유) 서버에서 실행.
#
#   bash stage_obj6_for_gen3c.sh REMOTE
# 예) bash stage_obj6_for_gen3c.sh elicer@10.0.0.5
#
# make_gen3c_batch.py 가 기대하는 상대경로 그대로 재현하므로,
# 40GB 서버에서 --masks_root data/replica_room0_v2/masks 등을 그대로 쓸 수 있음.
set -euo pipefail

# ---- 원본 서버 경로 (필요시 수정) ----
DATA=data/replica_room0_v2
STEMS=$HOME/See3D/dataset/stage6/clean_stems/6.txt
TSDF=output/replica_room0_v2/refinegs_full/6/train/ours_7000/fuse_post.ply
GID=6

# ---- 대상 ----
REMOTE=${1:?"사용법: bash stage_obj6_for_gen3c.sh user@host [원격_RefineGS_경로]"}
RDIR=${2:-'~/RefineGS'}          # 40GB 서버의 RefineGS 루트
STAGE=$HOME/tmp/obj6_stage       # /tmp 금지 → ~/tmp

rm -rf "$STAGE"
mkdir -p "$STAGE/$DATA/images" \
         "$STAGE/$DATA/masks/$GID/masks" \
         "$STAGE/$DATA/sparse/0" \
         "$STAGE/$(dirname "$TSDF")"

echo "[1/4] stems 기반 RGB + 마스크 복사"
n=0
while read -r s; do
  [ -z "$s" ] && continue
  # RGB (확장자 자동)
  for ext in jpg jpeg png JPEG; do
    f="$DATA/images/$s.$ext"
    [ -f "$f" ] && cp "$f" "$STAGE/$DATA/images/" && break
  done
  # 마스크
  m="$DATA/masks/$GID/masks/$s.png"
  [ -f "$m" ] && cp "$m" "$STAGE/$DATA/masks/$GID/masks/"
  n=$((n+1))
done < "$STEMS"
echo "     stems $n 개 처리, RGB $(ls "$STAGE/$DATA/images" | wc -l)장 / 마스크 $(ls "$STAGE/$DATA/masks/$GID/masks" | wc -l)장"

echo "[2/4] COLMAP sparse/0 복사"
cp "$DATA"/sparse/0/{cameras,images}.* "$STAGE/$DATA/sparse/0/" 2>/dev/null || true
# points3D 는 make_gen3c_batch 에 불필요하지만 있으면 포함(작음)
cp "$DATA"/sparse/0/points3D.* "$STAGE/$DATA/sparse/0/" 2>/dev/null || true

echo "[3/4] TSDF 메쉬 복사"
cp "$TSDF" "$STAGE/$TSDF"

# stems 파일도 함께 (경로는 홈 기준이라 별도 위치)
mkdir -p "$STAGE/stems"; cp "$STEMS" "$STAGE/stems/6.txt"

echo "     staging 용량: $(du -sh "$STAGE" | cut -f1)"

echo "[4/4] rsync → $REMOTE:$RDIR"
# data/ 트리를 원격 RefineGS 루트로, stems 는 홈으로
rsync -avhP "$STAGE/$DATA" "$REMOTE:$RDIR/data/"
rsync -avhP "$STAGE/stems/6.txt" "$REMOTE:~/See3D/dataset/stage6/clean_stems/6.txt"

cat <<EOF

완료. 40GB 서버에서:
  cd $RDIR
  python make_gen3c_batch.py --gid 6 --contiguous \\
    --masks_root data/replica_room0_v2/masks \\
    --images data/replica_room0_v2/images \\
    --colmap data/replica_room0_v2/sparse/0 \\
    --stems ~/See3D/dataset/stage6/clean_stems/6.txt \\
    --tsdf $TSDF \\
    --n_frames 30 --H 384 --W 640 --out ~/GEN3C/assets/obj6_batch
EOF

