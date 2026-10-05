#!/usr/bin/env bash
# RefineGS — 객체별 See3D 입력 생성 일반 드라이버 (모든 gid 동일 루프).
#   base ⊕ recon ⊕ gen 조립 → confidence → hole 라벨 → reachable novel pose 렌더 → See3D 입력(warp+mask)
#
# 사용:
#   conda activate split_and_splat
#   bash build_object_holes.sh 24                 # 단일 객체
#   for g in 1 8 24 27 28 29 36; do bash build_object_holes.sh $g; done   # 여러 객체(=모든 데이터)
#   # gen 있는 객체 전부 자동:
#   for g in $(ls ~/Amodal3R/poc_output); do bash build_object_holes.sh $g; done
#
# 객체별 reachability는 wide orbit(-20~70°) + free-space + obs-cone 필터가 자동 처리
# → table은 아랫면, cabinet은 옆/윗면이 reachable로 잡히고 벽-flush 뒷면은 reject(prior-bound).
set -u
GID="${1:?gid 인자 필요 (예: bash build_object_holes.sh 24)}"
SCENE="${SCENE:-replica_room0_v2}"
BASE_SCENE="${BASE_SCENE:-replica_room0}"
ROOT="${ROOT:-/home/elicer/RefineGS}"
TMP="${TMP:-$HOME/tmp}"; mkdir -p "$TMP"
cd "$ROOT" || exit 1

GEN="$HOME/Amodal3R/poc_output/$GID/seed_1/mesh_registered_clean.ply"
RECON=$(ls output/$SCENE/refinegs_fix/$GID/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | head -1)
BASE="output/$BASE_SCENE/scene_base/point_cloud/iteration_30000/point_cloud.ply"
BASE_MESH="output/$BASE_SCENE/scene_base/train/ours_30000/fuse_cropped.ply"
MASKS="$HOME/relabel_$SCENE/$GID"
COLMAP="data/$SCENE/sparse/0"
SBDIR="output/$SCENE/scene_b1_obj$GID"
SB="$SBDIR/point_cloud/iteration_1/point_cloud.ply"
SOFT_OUT="$HOME/See3D/dataset/refinegs_obj$GID/soft_in"

echo "=== gid $GID ==="
[ -f "$GEN" ] || { echo "[skip] gen 없음: $GEN (register/clean 먼저)"; exit 0; }

# 1) gen mesh → surfel
python mesh_to_surfels.py --mesh "$GEN" --out "$TMP/gen_surfels_$GID.ply" \
  --n_samples 200000 --scale_mult 1.0 --opacity 0.99 || exit 1

# 2) 조립.  NO_RECON=1 이면 base⊕gen (gen 실루엣이 recon에 안 가려져 weight가 실림 — 권장)
#    기본(NO_RECON=0)은 base⊕recon⊕gen.  ⚠️ recon이 gen을 occlude하면 weight≈0 됨(obj24 사례).
mkdir -p "$(dirname "$SB")"
if [ "${NO_RECON:-1}" != "1" ] && [ -n "$RECON" ] && [ -f "$RECON" ]; then
  python assemble_gaussians.py --base "$BASE" --gen "$RECON" "$TMP/gen_surfels_$GID.ply" --tag --out "$SB" || exit 1
  GEN_TAG=2
else
  echo "[info] base⊕gen (recon 제외: gen 실루엣 노출). recon은 joint 학습 때 실측 앵커로 사용."
  python assemble_gaussians.py --base "$BASE" --gen "$TMP/gen_surfels_$GID.ply" --tag --out "$SB" || exit 1
  GEN_TAG=1
fi
# render_hole_novel(get_combined_args)가 -m 폴더의 cfg_args 를 읽음 → 없으면 known-good 템플릿 복사.
# (cmdline -s/-m 가 cfg 값을 override 하므로 obj29 것 그대로 써도 됨)
if [ ! -f "$SBDIR/cfg_args" ]; then
  for SRC in "output/$SCENE/scene_b1_obj29/cfg_args" "output/$BASE_SCENE/scene_base/cfg_args"; do
    [ -f "$SRC" ] && cp "$SRC" "$SBDIR/cfg_args" && break
  done
fi
[ -f "$SBDIR/cfg_args" ] || echo "[warn] cfg_args 없음 — render 단계에서 get_combined_args 실패 가능(수동 생성 필요)"

# 3) confidence (관측성)
python confidence_map.py --gaussians "$SB" \
  --colmap_dir "$COLMAP" --masks_dir "$MASKS" \
  --min_views 2 --op_thr 0.3 --scale_thr 0.1 --out_prefix "$TMP/sb_$GID" || exit 1

# 4) hole = gen ∧ ¬observed
python make_hole_labels.py --gaussians "$SB" --gen_tags "$GEN_TAG" \
  --conf_npy "${TMP}/sb_${GID}_conf.npy" --out "$TMP/hole_$GID.npy" || exit 1

# 5) reachable novel pose 렌더 + soft-weight 출력(view+weight+poses)
python render_hole_novel.py -m "$SBDIR" -s "data/$SCENE" \
  --iteration 1 --hole_npy "$TMP/hole_$GID.npy" \
  --path orbit --n_frames 90 --max_views 40 \
  --elev_min -20 --elev_max 70 --up_axis "${UP_AXIS:-1}" --radius_scale "${RADIUS_SCALE:-0.7}" \
  --occluder_mesh "$BASE_MESH" --obs_cone_deg "${OBS_CONE:-85}" \
  --min_weight "${MIN_WEIGHT:-0.01}" \
  --out_dir "$SBDIR/holes_novel" \
  --soft_out "$SOFT_OUT" || exit 1

echo "=== gid $GID 완료 → soft-weight 입력: $SOFT_OUT ==="
