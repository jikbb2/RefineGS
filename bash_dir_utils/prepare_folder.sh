#!/bin/bash
# Per-object folder setup (idempotent). Turns each data/<scene>/masks/<gid>/ into a trainable
# structure: copy sparse/ (cameras and poses), move the top-level png into a masks/ subfolder,
# and rename the ply to points3d.ply.
#   images/ and depths/ are NOT created here: setup_instance_folders.py rebuilds them from the
#   masked frames only, afterwards.
#
# Bug fix: this loop used to contain `set -e`, so a non-zero cp or mv on the FIRST object
# killed the whole script and only obj0 was ever processed. set -e is gone, a self-move guard
# was added, and nullglob keeps an empty glob from becoming a literal '*'.

PARENT_FOLDER="$1"
PARENT_DIR="./data/$PARENT_FOLDER/masks"
SOURCE_SPARSE="./data/$PARENT_FOLDER/sparse"
SOURCE_JSON="./data/$PARENT_FOLDER/transforms_train.json"
DISCARD_DIR="./data/$PARENT_FOLDER/discard"
mkdir -p "$DISCARD_DIR"

shopt -s nullglob                       # an empty glob must not become a literal '*'

built=0; discarded=0
for SUB in "$PARENT_DIR"/*/; do
    echo "Processing: $SUB"

    # 1) sparse (cameras and poses) -- every object needs it. Idempotent: remove, then re-copy.
    if [ -d "$SOURCE_SPARSE" ]; then
        rm -rf "${SUB}sparse"
        cp -r "$SOURCE_SPARSE" "${SUB}sparse"
    elif [ -f "$SOURCE_JSON" ]; then
        cp -f "$SOURCE_JSON" "$SUB"
    fi

    # 2) move the top-level png into the masks/ subfolder
    mkdir -p "${SUB}masks"
    for P in "$SUB"*.png; do
        mv -f "$P" "${SUB}masks/"
    done

    # 3) *.ply -> points3d.ply (guard against moving the file onto itself)
    for FILE in "$SUB"*.ply; do
        [ -f "$FILE" ] || continue
        [ "$FILE" = "${SUB}points3d.ply" ] && continue
        mv -f "$FILE" "${SUB}points3d.ply"
    done

    # 4) fewer than 2 masks -> discard
    NUM_MASKS=$(find "${SUB}masks" -type f \( -iname "*.png" -o -iname "*.jpg" \) 2>/dev/null | wc -l)
    if [ "$NUM_MASKS" -lt 2 ]; then
        echo "  -> masks $NUM_MASKS < 2, discard"
        mv "$SUB" "$DISCARD_DIR/" 2>/dev/null
        discarded=$((discarded+1))
    else
        built=$((built+1))
    fi
done
echo "Done! prepared=${built}, discarded=${discarded}"