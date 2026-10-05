#!/usr/bin/env bash
# obj6 GEN3C 테스트에 필요한 데이터만 골라 zip 하나로 묶는다.
# 원본(데이터 보유) 서버에서 실행 → 생성된 zip 을 40GB 서버로 다운로드.
#
#   bash pack_obj6_for_gen3c.sh
#
# 결과: ~/tmp/obj6_gen3c.zip
# make_gen3c_batch.py 가 기대하는 상대경로 그대로 담으므로,
# 40GB 서버 RefineGS 루트에서 그대로 풀면 됨.
set -euo pipefail

# ---- 원본 서버 경로 (필요시 수정) ----
DATA=data/replica_room0_v2
STEMS=$HOME/See3D/dataset/stage6/clean_stems/6.txt
TSDF=output/replica_room0_v2/refinegs_full/6/train/ours_7000/fuse_post.ply
GID=6

STAGE=$HOME/tmp/obj6_stage       # /tmp 금지 → ~/tmp
ZIP=$HOME/tmp/obj6_gen3c.zip

rm -rf "$STAGE"
mkdir -p "$STAGE/$DATA/images" \
         "$STAGE/$DATA/masks/$GID/masks" \
         "$STAGE/$DATA/sparse/0" \
         "$STAGE/$(dirname "$TSDF")" \
         "$STAGE/stems"

echo "[1/4] stems 기반 RGB + 마스크 복사"
n=0
while read -r s; do
  [ -z "$s" ] && continue
  for ext in jpg jpeg png JPEG; do
    f="$DATA/images/$s.$ext"
    [ -f "$f" ] && cp "$f" "$STAGE/$DATA/images/" && break
  done
  m="$DATA/masks/$GID/masks/$s.png"
  [ -f "$m" ] && cp "$m" "$STAGE/$DATA/masks/$GID/masks/"
  n=$((n+1))
done < "$STEMS"
echo "     stems $n 개, RGB $(ls "$STAGE/$DATA/images" | wc -l)장 / 마스크 $(ls "$STAGE/$DATA/masks/$GID/masks" | wc -l)장"

echo "[2/4] COLMAP sparse/0 복사"
cp "$DATA"/sparse/0/{cameras,images}.* "$STAGE/$DATA/sparse/0/" 2>/dev/null || true
cp "$DATA"/sparse/0/points3D.* "$STAGE/$DATA/sparse/0/" 2>/dev/null || true

echo "[3/4] TSDF 메쉬 + stems 복사"
cp "$TSDF" "$STAGE/$TSDF"
cp "$STEMS" "$STAGE/stems/6.txt"

echo "[4/4] zip 생성"
rm -f "$ZIP"
( cd "$STAGE" && zip -r -q "$ZIP" . )
echo "     → $ZIP  ($(du -sh "$ZIP" | cut -f1))"

cat <<EOF

완료. 40GB 서버에서:
  1) zip 다운로드 후 RefineGS 루트에서 풀기
       cd ~/RefineGS && unzip -o ~/tmp/obj6_gen3c.zip -d .
     (stems/6.txt 는 See3D 경로로 별도 이동)
       mkdir -p ~/See3D/dataset/stage6/clean_stems
       cp stems/6.txt ~/See3D/dataset/stage6/clean_stems/6.txt

  2) batch 생성
       python make_gen3c_batch.py --gid 6 --contiguous \\
         --masks_root data/replica_room0_v2/masks \\
         --images data/replica_room0_v2/images \\
         --colmap data/replica_room0_v2/sparse/0 \\
         --stems ~/See3D/dataset/stage6/clean_stems/6.txt \\
         --tsdf $TSDF \\
         --n_frames 30 --H 384 --W 640 --out ~/GEN3C/assets/obj6_batch
EOF

