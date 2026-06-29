#!/bin/bash
# per-object 폴더 셋업 (멱등). 각 data/<scene>/masks/<gid>/ 를 학습 가능한 구조로:
#   sparse/ 복사(카메라/포즈) + masks/ 서브폴더로 png 이동 + ply→points3d.ply.
#   images/·depths/ 는 이후 setup_instance_folders.py 가 마스크 프레임만으로 재생성하므로 여기선 안 만듦.
#
# ★ 버그 수정: 기존엔 루프 안에 `set -e` 가 있어 첫 객체의 cp/mv 가 non-zero면 전체가 죽어
#   obj0 하나만 처리됐음. set -e 제거 + 자기참조 mv 가드 + nullglob 로 견고화.

PARENT_FOLDER="$1"
PARENT_DIR="./data/$PARENT_FOLDER/masks"
SOURCE_SPARSE="./data/$PARENT_FOLDER/sparse"
SOURCE_JSON="./data/$PARENT_FOLDER/transforms_train.json"
DISCARD_DIR="./data/$PARENT_FOLDER/discard"
mkdir -p "$DISCARD_DIR"

shopt -s nullglob                       # 빈 glob → 리터럴 '*' 방지

built=0; discarded=0
for SUB in "$PARENT_DIR"/*/; do
    echo "Processing: $SUB"

    # 1) sparse (카메라/포즈) — 각 객체에 필요. 멱등(기존 제거 후 재복사).
    if [ -d "$SOURCE_SPARSE" ]; then
        rm -rf "${SUB}sparse"
        cp -r "$SOURCE_SPARSE" "${SUB}sparse"
    elif [ -f "$SOURCE_JSON" ]; then
        cp -f "$SOURCE_JSON" "$SUB"
    fi

    # 2) masks/ 서브폴더로 top-level png 이동
    mkdir -p "${SUB}masks"
    for P in "$SUB"*.png; do
        mv -f "$P" "${SUB}masks/"
    done

    # 3) *.ply → points3d.ply (자기 자신 mv 방지)
    for FILE in "$SUB"*.ply; do
        [ -f "$FILE" ] || continue
        [ "$FILE" = "${SUB}points3d.ply" ] && continue
        mv -f "$FILE" "${SUB}points3d.ply"
    done

    # 4) mask 2개 미만이면 discard
    NUM_MASKS=$(find "${SUB}masks" -type f \( -iname "*.png" -o -iname "*.jpg" \) 2>/dev/null | wc -l)
    if [ "$NUM_MASKS" -lt 2 ]; then
        echo "  → masks $NUM_MASKS < 2, discard"
        mv "$SUB" "$DISCARD_DIR/" 2>/dev/null
        discarded=$((discarded+1))
    else
        built=$((built+1))
    fi
done
echo "Done! prepared=${built}, discarded=${discarded}"
