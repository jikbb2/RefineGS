#!/usr/bin/env bash
# RefineGS end to end: raw scene -> seen/unseen numbers, in one command.
#
#   bash run_refinegs.sh                          # whole thing, scene pipeline
#   SCENE=replica_room1 bash run_refinegs.sh      # another room
#   PIPELINE=perobj bash run_refinegs.sh          # the per-object variant
#   FROM=cond bash run_refinegs.sh                # resume
#   FROM=extract TO=cond CLEAN=1 bash run_refinegs.sh
#   RUN=0919a FROM=fuse bash run_refinegs.sh      # add rows to an earlier run
#
# Stages (linear; FROM/TO select a span):
#
#   colmap   poses for every frame                       convert.py, make_dense_colmap.py
#   relabel  SAM3 video instances -> per-gid masks       [sam3 env]
#   masks    amodal + per-object folders                 amodal_mask.py, prepare_folder.sh,
#                                                        setup_instance_folders.py
#   labels   per-view label maps + id_map.json           make_label_maps.py      [scene only]
#   train    scene 2DGS model                            train.py                [scene only]
#   carve    rendered scene depth -> free-space ref      dump_scene_depth.py
#   objects  scene: vote + slice     perobj: train each  vote_labels.py, extract_objects.py
#   name     GT class per object -> names.tsv            name_objects.py
#   mesh     per-object TSDF -> fuse_post.ply  (side A)  mesh_tsdf_views.py
#   cond     boundary-filtered TSDF -> tsdf_clean.ply    mesh_tsdf_views.py
#   pkl      ShapeR input                                run_field_fusion_batch.sh
#   field    ShapeR signed SDF grid                      run_field_fusion_batch.sh [shaper env]
#   fuse     grid fusion + evaluation           (side B) run_field_fusion_batch.sh
#
# Why this file exists: the same run used to be three drivers and three manual steps, and
# each boundary lost something -- the conditioning surface, the RUN tag, the pose set, the
# pkl directory. Everything that must agree is resolved once, here, and printed.
#
# It does NOT reimplement the two drivers that carry hard-won logic: mesh_voted_objects.sh
# and run_field_fusion_batch.sh (stale-prior guard, progress lines, per-object failure
# rows, summary table) are called as they are.
set -uo pipefail
shopt -s nullglob

# ---------------------------------------------------------------- identity
ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
PIPELINE=${PIPELINE:-scene}            # scene = slice one trained model | perobj = train each
# One tag for the whole run. Every result file carries it, so a number that was reported
# can always be traced back to the inputs that produced it.
RUN=${RUN:-$(date +%m%d_%H%M)}
FROM=${FROM:-colmap}
TO=${TO:-fuse}
CLEAN=${CLEAN:-0}                      # delete this stage's outputs and everything after
ONLY=${ONLY:-}                         # restrict to some gids, e.g. ONLY="2 5 6"

# ---------------------------------------------------------------- data
DATA=${DATA:-${ROOT}/data/${SCENE}}
IMAGES=${IMAGES:-${DATA}/images}
MASKS=${MASKS:-${DATA}/masks}
LABEL_DIR=${LABEL_DIR:-${DATA}/labels_scene}
IMG_EXT=${IMG_EXT:-.jpg}
# One pose source for relabel, training, fusion and evaluation alike. Splitting these is
# how masks end up labelled against poses nothing downstream reads.
if [ -z "${COLMAP:-}" ]; then
  for sd in sparse/0 sparse_dense/0; do
    [ -d "${DATA}/${sd}" ] && { COLMAP=${DATA}/${sd}; break; }
  done
fi
COLMAP=${COLMAP:-${DATA}/sparse/0}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
GT_INFO=${GT_INFO:-$HOME/room_0/habitat/info_semantic.json}
TRAJ=${TRAJ:-$HOME/room_0/imap/00/traj_w_c.txt}

# ---------------------------------------------------------------- outputs
# PRIOR and PKL_SUBDIR are derived from the pipeline, not chosen per experiment. A field
# npz records nothing about its origin and the two pipelines write the same obj<gid>.pkl,
# so a shared directory lets the stale guard rebuild one pipeline's field from the other's
# input, silently. Deriving them makes that impossible rather than merely discouraged.
case "${PIPELINE}" in
  scene)  OBJ_DEFAULT=${ROOT}/output/${SCENE}/objects_voted;  ITER_DEFAULT=30000 ;;
  perobj) OBJ_DEFAULT=${ROOT}/output/${SCENE}/objects_perobj; ITER_DEFAULT=7000  ;;
  *) echo "[abort] PIPELINE must be scene or perobj"; exit 1 ;;
esac
OBJ=${OBJ:-${OBJ_DEFAULT}}
ITER=${ITER:-${ITER_DEFAULT}}
SCENE_MODEL=${SCENE_MODEL:-${ROOT}/output/${SCENE}/scene}
PRIOR=${PRIOR:-${ROOT}/output/${SCENE}/prior/${PIPELINE}}
PKL_SUBDIR=${PKL_SUBDIR:-${SCENE}_${PIPELINE}}
RUNDIR=${RUNDIR:-${ROOT}/output/${SCENE}/runs/${RUN}}
CSV=${CSV:-${RUNDIR}/results.csv}
SHAPER_DIR=${SHAPER_DIR:-$HOME/ShapeR}
RELABEL=${RELABEL:-$HOME/relabel_${SCENE}}
AMODAL=${AMODAL:-$HOME/amodal_${SCENE}}
# The scene model's own rendered depth, used wherever the pipeline needs to know what a
# camera saw past: the vote's first-surface test and the fusion's free-space carve. GT
# depth answers the same question, but then the hard constraint is oracle-derived and the
# method cannot run on a dataset without it.
CARVE_DEPTH=${CARVE_DEPTH:-${ROOT}/output/${SCENE}/carve_depth}
# The vote is the exception. Its first-surface test needs a reference INDEPENDENT of the
# model being labelled, and the scene render is not one: a gaussian of that model is
# trivially within the margin of that model's own rendered surface, so the test partly
# answers itself. Measured on room0, switching the vote to it moved 19.24% of all
# gaussians -- unassigned +93%, background -27%, every object label +2..31% -- partly for
# that reason and partly because dump_scene_depth zeroes every pixel below alpha 0.5, so
# the background stops voting at all. The fusion's carve is a different question ("did a
# camera see PAST this point") that the scene model is entitled to answer, and it checks
# out at 1-3mm against the object render.
#   gt    = GT depth. Every recorded number comes from this.
#   carve = rendered scene depth, kept so the GT-free variant stays measurable.
VOTE_REF=${VOTE_REF:-gt}
# none = no depth loss in scene training (the reconstruction claim stops resting on GT).
# gt   = the earlier setting, kept so the two can be compared.
DEPTH_SUPERVISION=${DEPTH_SUPERVISION:-none}

# ---------------------------------------------------------------- knobs
# Settled values live in the tools' own defaults; only what this driver must choose is here.
CONDA_SAM3=${CONDA_SAM3:-sam3}
VOCAB=${VOCAB:-$HOME/sam3/vocab.json}
BPE=${BPE:-$HOME/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz}
EXCLUDE=${EXCLUDE:-"door,blind,vent,window,wall,floor,ceiling,light switch,thermostat"}
STRIDE=${STRIDE:-2}; WINDOW=${WINDOW:-200}; PROMPT_FRAME=${PROMPT_FRAME:-0}
MIN_AREA=${MIN_AREA:-0.003}; MIN_TRACK=${MIN_TRACK:-3}
REID=${REID:-0.3}; IOU=${IOU:-0.5}; CAND=${CAND:-0.1}
# make_label_maps.py's own default and the hardest object filter in the project: a gid
# below it gets no label, extract_objects.py makes no dir, and the run cannot report the
# object even as a failure. Measured room0: 12 of 48 gids. It is absolute, so it gets
# stricter as a scene gets shorter -- revisit it per room, not per experiment.
MIN_LABEL_VIEWS=${MIN_LABEL_VIEWS:-30}
OVERLAP=${OVERLAP:-ignore}
# Off, as upstream has it. Turning it on (0.5) looked right -- voted labels 4/9/14/20/30
# measure compactness 0.16-0.40, i.e. one label over several objects -- but measured on
# room0 it doubled the damage instead: obj2 unseen completion 1430mm off vs 2731mm on.
# Keeping the largest blob throws away real chair geometry. Merged labels are a
# segmentation problem and belong in a segmentation ablation, not in a silent default.
SPLIT_BELOW=${SPLIT_BELOW:-}
EXTRACT_EXTRA=${EXTRACT_EXTRA:-${SPLIT_BELOW:+--split_below ${SPLIT_BELOW}}}
COND_NAME=${COND_NAME:-tsdf_clean.ply}
# Both meshes come from mesh_tsdf_views.py now, at two filter strengths. render.py's TSDF
# has no alpha gate, so rays that miss the object still contribute a composited depth; on
# room0's chairs that sheet became the largest component and --num_cluster 1 reported it
# instead of the chair (a 2.4x3.4x0.13 m slab at floor level for a 1.6x1.7x1.1 m chair).
#
# Side A, the observed surface we report. Measured on gid2: min_alpha closes the holes
# (0.7 -> 0.3, the chair's gaussians have median opacity 0.61 and rarely reach 0.7),
# min_cos removes the panel its back faces produce (0.0 -> 0.35), erode only shrinks.
# min_cos 0.35 is not tunable downward: at 0.15 the grazing pixels that survive are the
# WALL behind the chair, the GT match then unions id89, and seen completion goes to 943mm.
# max_jump 0.05 over 0.02 closes the holes -- an armchair's own arms and cushion make
# depth steps larger than 2cm inside its silhouette, so a tight jump filter erases real
# surface. Measured gid2: seen completion 25.11 -> 7.93mm, NC 0.894 -> 0.926, against
# seen accuracy 3.60 -> 4.07mm and free 1.2 -> 3.6%. Holes are the worse error here: they
# move observed surface into the unseen bucket and flatter B's improvement.
MESH_ARGS=${MESH_ARGS:-"--min_alpha 0.3 --min_cos 0.35 --max_jump 0.05 --erode 0 --num_cluster 0 --min_comp_frac 0.02"}
# Conditioning, filtered harder on purpose: ShapeR anchors to these points so a ragged
# boundary is copied into the prior, and the holes that leaves are what the prior fills.
COND_ARGS=${COND_ARGS:-"--min_alpha 0.7 --min_cos 0.35 --max_jump 0.02 --erode 3 --num_cluster 0 --min_comp_frac 0.02"}
POINTS_FROM=${POINTS_FROM:-mesh}
FUSE_EXTRA=${FUSE_EXTRA:-}
# One resolution for BOTH pipelines, which is what removes the r1/r2 confound from the
# per-object vs scene-slice comparison. r1 does not fit in GPU memory and had to train on
# CPU, so r2 is the one both can actually run the same way.
RESOLUTION=${RESOLUTION:-2}
DATA_DEVICE=${DATA_DEVICE:-cuda}       # r1 needed cpu; r2 fits
# perobj training, verified against run_full_pipeline.sh
OBJ_ITERS=${OBJ_ITERS:-7000}; LDIST=${LDIST:-300}; LNORM=${LNORM:-0.05}
OBJ_MIN_MASKS=${OBJ_MIN_MASKS:-5}
# Scene training. Not filled in: train.py's scene-level invocation (label embedding,
# resolution) was never recorded, and guessing an argument name here would train for hours
# and produce the wrong model. Set it once and it is captured in every manifest.
TRAIN_SCENE_ARGS=${TRAIN_SCENE_ARGS:-}
# Replica frame dump. Same reason: set it, or prepare data/<scene>/images yourself.
FRAMES_CMD=${FRAMES_CMD:-}
# convert.py runs COLMAP SfM for hours and rewrites the source tree, so it is opt-in.
COLMAP_SFM=${COLMAP_SFM:-0}

cd "${ROOT}" || exit 1
mkdir -p "${RUNDIR}" "${PRIOR}"

# ---------------------------------------------------------------- helpers
ORDER="colmap relabel masks labels train carve objects name mesh cond pkl field fuse"
idx_of() { local i=0 s; for s in ${ORDER}; do i=$((i+1)); [ "$s" = "$1" ] && { echo "$i"; return; }; done; echo 0; }
I_FROM=$(idx_of "${FROM}"); I_TO=$(idx_of "${TO}")
[ "${I_FROM}" -gt 0 ] && [ "${I_TO}" -gt 0 ] || { echo "[abort] FROM/TO must be one of: ${ORDER}"; exit 1; }
want() { local i; i=$(idx_of "$1"); [ "$i" -ge "${I_FROM}" ] && [ "$i" -le "${I_TO}" ]; }
want_any() { local s; for s in "$@"; do want "${s}" && return 0; done; return 1; }
at_or_after() { local i; i=$(idx_of "$1"); [ "$i" -ge "${I_FROM}" ]; }

# A result is reusable only when it is newer than its generator and its inputs. Existence
# alone silently reuses a file built from something that has since changed; that is what
# invalidated two days of runs once, and the file names give no hint.
fresh() {
  local t=$1 d; shift
  [ -f "${t}" ] || return 1
  for d in "$@"; do [ -e "${d}" ] && [ "${d}" -nt "${t}" ] && return 1; done
  return 0
}

# sam3 and split_and_splat cannot share a process (cuDNN), and `conda run` breaks on the
# cross-compiler activate hook, so the env is entered the way the field phase already does.
in_env() {
  local env=$1; shift
  local base; base=$(conda info --base 2>/dev/null)
  [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ] \
    || { echo "[abort] conda not found; run the ${env} stages by hand"; return 1; }
  bash -c "source '${base}/etc/profile.d/conda.sh' && conda activate '${env}' && $*"
}

n_poses() { grep -cE '\.(jpg|jpeg|png|JPG|PNG)[[:space:]]*$' "$1/images.txt" 2>/dev/null || echo 0; }
n_files() { ls "$1" 2>/dev/null | wc -l; }
# `ls -d dir/*/ | wc -l` reports 1 on an empty dir under nullglob, because ls falls back to
# the working directory. Count the loop instead.
count_dirs() {                                # count_dirs ROOT [numeric-only]
  local d b n=0
  for d in "$1"/*/; do
    b=$(basename "${d}")
    [ -n "${2:-}" ] && ! [[ "${b}" =~ ^[0-9]+$ ]] && continue
    n=$((n + 1))
  done
  echo "${n}"
}
say() { echo ""; echo "=== $* ==="; }

# ---------------------------------------------------------------- preflight
GITREV=$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || echo "no-git")
MANIFEST=${RUNDIR}/manifest.txt
{
  echo "run           ${RUN}            $(date '+%F %T')"
  echo "git           ${GITREV}"
  echo "scene         ${SCENE}          pipeline=${PIPELINE}  iter=${ITER}"
  echo "resolution    -r ${RESOLUTION}         data_device=${DATA_DEVICE}"
  echo "depth         supervision=${DEPTH_SUPERVISION}  vote_ref=${VOTE_REF}"
  echo "carve         ${CARVE_DEPTH}"
  echo "stages        ${FROM} .. ${TO}  clean=${CLEAN}  only='${ONLY}'"
  echo "colmap        ${COLMAP}         poses=$(n_poses "${COLMAP}")"
  echo "images        ${IMAGES}         n=$(n_files "${IMAGES}")"
  echo "masks         ${MASKS}"
  echo "labels        ${LABEL_DIR}      min_views=${MIN_LABEL_VIEWS} overlap=${OVERLAP}"
  echo "gt_depth      ${GTD}"
  echo "gt_mesh       ${GT_MESH}"
  echo "scene_model   ${SCENE_MODEL}"
  echo "objects       ${OBJ}"
  echo "prior         ${PRIOR}"
  echo "pkl           ${SHAPER_DIR}/data/${PKL_SUBDIR}"
  echo "extract_extra ${EXTRACT_EXTRA:-<none>}"
  echo "mesh_args     ${MESH_ARGS}"
  echo "cond_args     ${COND_ARGS}"
  echo "fuse_extra    ${FUSE_EXTRA:-<none>}"
  echo "relabel       stride=${STRIDE} window=${WINDOW} min_area=${MIN_AREA} min_track=${MIN_TRACK}"
} > "${MANIFEST}"
echo "+- [run ${RUN}] --------------------------------------------------"
sed 's/^/| /' "${MANIFEST}"
echo "+-----------------------------------------------------------------"

# Require an input only when a stage in THIS span consumes it and no stage in this span
# produces it. Demanding a scene model for a FROM=labels TO=labels run is noise.
fail=0
chk() { [ -e "$1" ] || { echo "  MISSING $1${2:+   ($2)}"; fail=1; }; }
want colmap || { want_any relabel objects cond pkl fuse && { chk "${IMAGES}"; chk "${COLMAP}"; }; }
want masks  || { want_any labels objects mesh cond pkl fuse && chk "${MASKS}"; }
if [ "${PIPELINE}" = "scene" ] && want objects; then
  want labels || chk "${LABEL_DIR}/id_map.json" "stage: labels"
  want train  || chk "${SCENE_MODEL}/point_cloud/iteration_${ITER}/point_cloud.ply" "stage: train"
fi
want_any objects pkl fuse && chk "${GTD}"
want carve || { want_any objects fuse && [ ! -d "${CARVE_DEPTH}" ] \
  && echo "  note: no ${CARVE_DEPTH} -- vote and fusion will fall back to GT depth"; }
[ "${fail}" -eq 0 ] || { echo "[abort] start earlier with FROM=, or fix the paths above"; exit 1; }

# ---------------------------------------------------------------- clean
# CLEAN is whole-tree: this stage's outputs AND everything after, because a changed input
# makes those stale too. ONLY is per-object. Together they destroy every other object in
# order to rebuild one, which is not what anybody means by it.
if [ "${CLEAN}" = "1" ] && [ -n "${ONLY}" ]; then
  echo "[abort] CLEAN=1 with ONLY='${ONLY}': CLEAN retires the whole ${OBJ} tree, not just"
  echo "        those gids. Drop ONLY to rebuild everything, or drop CLEAN and let fresh()"
  echo "        rebuild whatever is actually stale."
  exit 1
fi
if [ "${CLEAN}" = "1" ]; then
  say "clean: retiring outputs from '${FROM}' onward"
  # Move, do not delete. These cost tens of minutes (vote, meshes) to hours (ShapeR
  # fields) to rebuild, and a mistyped command should not spend that.
  TRASH=${ROOT}/output/${SCENE}/_trash/${RUN}
  retire() {
    local q rel n=0
    for q in "$@"; do
      [ -e "${q}" ] || continue
      mkdir -p "${TRASH}"
      rel=${q#"${ROOT}/"}; rel=${rel//\//__}
      mv "${q}" "${TRASH}/${rel}" 2>/dev/null && n=$((n + 1))
    done
    [ "${n}" -gt 0 ] && echo "  retired ${n} under $(dirname "$1")"
    return 0
  }
  if [ "${TO}" != "fuse" ]; then
    echo "  NOTE TO=${TO}, but CLEAN still retires pkl/field/fuse outputs -- they go stale"
    echo "       the moment an earlier stage is rebuilt."
  fi
  at_or_after labels  && retire "${LABEL_DIR}"
  at_or_after objects && retire "${OBJ}"
  at_or_after mesh    && retire "${OBJ}"/*/train/ours_"${ITER}"/fuse.ply \
                                "${OBJ}"/*/train/ours_"${ITER}"/fuse_post.ply
  at_or_after cond    && retire "${OBJ}"/*/train/ours_"${ITER}"/"${COND_NAME}"
  at_or_after pkl     && retire "${SHAPER_DIR}/data/${PKL_SUBDIR}"
  at_or_after field   && retire "${PRIOR}"/obj*_field*.npz
  # Fused meshes and the CSV carry ${RUN}: CLEAN removes inputs to redo, never evidence
  # for numbers already reported.
  at_or_after fuse    && retire "${OBJ}"/*/train/ours_"${ITER}"/fused_"${RUN}"*.ply
  [ -d "${TRASH}" ] && echo "  -> ${TRASH}   (restore by moving back; '__' was '/')"
fi

# ---------------------------------------------------------------- colmap
if want colmap; then
  say "colmap: a pose for every frame"
  if [ ! -d "${IMAGES}" ] || [ "$(n_files "${IMAGES}")" -eq 0 ]; then
    if [ -n "${FRAMES_CMD}" ]; then eval "${FRAMES_CMD}" || exit 1
    else echo "  [STOP] ${IMAGES} is empty. Set FRAMES_CMD to the Replica frame dump."; exit 1; fi
  fi
  NP=$(n_poses "${COLMAP}"); NF=$(n_files "${IMAGES}")
  echo "  poses ${NP} / frames ${NF}"
  if [ "${NP}" -lt "${NF}" ]; then
    # convert.py's exhaustive_matcher is quadratic, so the SfM is run on a stride subset
    # and make_dense_colmap.py lifts it to every frame off the GT trajectory. It verifies
    # itself against the SfM poses (0.5 deg / 1 cm) and refuses to write on a mismatch.
    if [ "${NP}" -eq 0 ]; then
      [ "${COLMAP_SFM}" = "1" ] || { echo "  [STOP] no SfM yet. COLMAP_SFM=1 to run convert.py (hours)."; exit 1; }
      python convert.py -s "${DATA}" || exit 1
    fi
    [ -f "${TRAJ}" ] || { echo "  [STOP] no trajectory at ${TRAJ}; set TRAJ="; exit 1; }
    python make_dense_colmap.py --traj "${TRAJ}" --frames "${IMAGES}" --img_ext "${IMG_EXT}" \
      --colmap_in "${COLMAP}" --out "${DATA}/sparse_dense/0" || exit 1
    # One pose directory, so the drivers that hardcode sparse/0 cannot diverge from this run
    [ -d "${DATA}/sparse/0" ] && mv "${DATA}/sparse/0" "${DATA}/sparse/0_sfm_$(date +%m%d)"
    mkdir -p "${DATA}/sparse" && mv "${DATA}/sparse_dense/0" "${DATA}/sparse/0"
    COLMAP=${DATA}/sparse/0
    echo "  -> ${COLMAP}  poses=$(n_poses "${COLMAP}")"
  else
    echo "  up to date"
  fi
fi

# ---------------------------------------------------------------- relabel
if want relabel; then
  say "relabel: SAM3 video instances  [${CONDA_SAM3} env]"
  if [ -d "${RELABEL}" ] && [ "$(count_dirs "${RELABEL}")" -gt 0 ]; then
    echo "  ${RELABEL} has $(count_dirs "${RELABEL}") objects, reusing (CLEAN or rm to redo)"
  else
    in_env "${CONDA_SAM3}" "LD_LIBRARY_PATH= python sam3_relabel_video.py \
      --frames '${IMAGES}' --img_ext '${IMG_EXT}' --colmap_dir '${COLMAP}' \
      --vocab_json '${VOCAB}' --bpe '${BPE}' --stride ${STRIDE} --window ${WINDOW} \
      --prompt_frame ${PROMPT_FRAME} --min_area ${MIN_AREA} --min_track ${MIN_TRACK} \
      --reid_th ${REID} --iou_th ${IOU} --cand_th ${CAND} \
      --exclude_concepts '${EXCLUDE}' --out_root '${RELABEL}'" || exit 1
    echo "  ${RELABEL}: $(count_dirs "${RELABEL}") objects"
  fi
fi

# ---------------------------------------------------------------- masks
if want masks; then
  say "masks: amodal + per-object folders"
  [ -d "${RELABEL}" ] || { echo "[abort] ${RELABEL} missing"; exit 1; }
  python amodal_mask.py --in_root "${RELABEL}" --out_root "${AMODAL}" || exit 1
  built=0
  for D in "${AMODAL}"/*/; do
    gid=$(basename "${D}"); dst=${MASKS}/${gid}
    mkdir -p "${dst}"
    cp "${D}"*.png "${dst}/" 2>/dev/null
    cp "${RELABEL}/${gid}"/*.ply "${dst}/" 2>/dev/null   # mask-filtered COLMAP sparse = object init
    built=$((built+1))
  done
  echo "  ${built} object folders"
  [ "${built}" -gt 0 ] || { echo "[abort] 0 objects"; exit 1; }
  bash bash_dir_utils/prepare_folder.sh "${SCENE}" || exit 1   # png -> <gid>/masks/, sparse, points3d
  python setup_instance_folders.py "${SCENE}" || exit 1        # images/depths = mask frames only
fi

# ---------------------------------------------------------------- labels
if want labels; then
  if [ "${PIPELINE}" != "scene" ]; then
    echo ""; echo "=== labels: skipped (perobj trains from masks directly) ==="
  else
    say "labels: per-view label maps"
    if fresh "${LABEL_DIR}/id_map.json" make_label_maps.py; then echo "  up to date"
    else python make_label_maps.py --masks_root "${MASKS}" --out "${LABEL_DIR}" \
           --min_views "${MIN_LABEL_VIEWS}" --overlap "${OVERLAP}" || exit 1; fi
  fi
fi

# ---------------------------------------------------------------- train (scene)
if want train; then
  if [ "${PIPELINE}" != "scene" ]; then
    echo ""; echo "=== train: skipped (perobj trains in the objects stage) ==="
  else
    say "train: scene 2DGS model"
    PLY=${SCENE_MODEL}/point_cloud/iteration_${ITER}/point_cloud.ply
    if [ -f "${PLY}" ]; then echo "  ${PLY} exists, reusing"
    elif [ -n "${TRAIN_SCENE_ARGS}" ]; then
      _dsup=""
      [ "${DEPTH_SUPERVISION}" = "gt" ] && _dsup="--gt_depth_dir ${GTD} --lambda_gtdepth 0.5"
      python train.py -s "${DATA}" -m "${SCENE_MODEL}" --iterations "${ITER}" \
        -r "${RESOLUTION}" --data_device "${DATA_DEVICE}" \
        --disable_viewer ${_dsup} ${TRAIN_SCENE_ARGS} || exit 1
    else
      echo "  [STOP] set TRAIN_SCENE_ARGS (label-embedding and resolution flags)."
      echo "         Guessing them would train for hours and produce the wrong model."
      exit 1
    fi
  fi
fi

# ---------------------------------------------------------------- carve
if want carve; then
  say "carve: rendered scene depth -> ${CARVE_DEPTH}"
  PLY=${SCENE_MODEL}/point_cloud/iteration_${ITER}/point_cloud.ply
  if [ ! -f "${PLY}" ]; then
    echo "  [skip] no scene model -- the carve reference needs one (stage: train)"
  elif fresh "${CARVE_DEPTH}/.done" dump_scene_depth.py "${PLY}"; then
    echo "  up to date ($(ls "${CARVE_DEPTH}"/*.npz 2>/dev/null | wc -l) views)"
  else
    # --depth_ratio 1 explicitly: it is a PipelineParams value whose default is not
    # render.py's, and a mismatch makes this a different quantity from the meshes.
    python dump_scene_depth.py -m "${SCENE_MODEL}" -s "${DATA}" --iteration "${ITER}" \
      --depth_ratio 1 --out_dir "${CARVE_DEPTH}" || exit 1
    touch "${CARVE_DEPTH}/.done"
  fi
fi

# ---------------------------------------------------------------- objects
if want objects; then
  if [ "${PIPELINE}" = "scene" ]; then
    say "objects: vote + slice the scene model"
    PLY=${SCENE_MODEL}/point_cloud/iteration_${ITER}/point_cloud.ply
    # The vote depends on the LABEL SOURCE, not only the scene model: two label sets over
    # one scene must not share a vote dir, or the second run finds labels.npy, skips, and
    # extracts the first run's assignment under the second run's names.
    VOTE=${VOTE:-${OBJ}/vote}
    if fresh "${VOTE}/labels.npy" vote_labels.py "${PLY}" "${LABEL_DIR}/id_map.json"; then
      echo "  vote up to date"
    else
      _vref="--gt_depth_dir ${GTD}"
      [ "${VOTE_REF}" = "carve" ] && [ -d "${CARVE_DEPTH}" ] \
        && _vref="--carve_depth_dir ${CARVE_DEPTH}"
      python vote_labels.py --ply "${PLY}" --colmap "${COLMAP}" --label_dir "${LABEL_DIR}" \
        ${_vref} --out "${VOTE}" || exit 1
      echo "  --- label coherence (previous run: mean compactness 0.754, 17 classes >= 0.8) ---"
      python check_scene_labels.py --ply "${PLY}" --labels "${VOTE}/labels.npy" | tail -6
    fi
    if fresh "${OBJ}/objects.json" extract_objects.py "${VOTE}/labels.npy"; then
      echo "  extract up to date"
    else
      python extract_objects.py --ply "${PLY}" --labels "${VOTE}/labels.npy" \
        --scene_dir "${SCENE_MODEL}" --id_map "${LABEL_DIR}/id_map.json" \
        --source_root "${MASKS}" --vote_dir "${VOTE}" \
        --out "${OBJ}" --iter "${ITER}" ${EXTRACT_EXTRA} || exit 1
    fi
  else
    say "objects: train one model per object"
    mkdir -p "${OBJ}"
    # Best observed first, so a truncated run still holds the objects worth reporting.
    mapfile -t DIRS < <(for D in "${MASKS}"/*/; do [ -d "${D}masks" ] || continue
      echo "$(find "${D}masks" -iname '*.png' | wc -l) ${D}"; done | sort -rn | awk '{print $2}')
    for D in "${DIRS[@]}"; do
      gid=$(basename "${D}")
      [ -z "${ONLY}" ] || [[ " ${ONLY} " == *" ${gid} "* ]] || continue
      NM=$(find "${D}masks" -iname "*.png" | wc -l)
      [ "${NM}" -ge "${OBJ_MIN_MASKS}" ] || continue
      IT=${OBJ_ITERS}; [ "${NM}" -lt 20 ] && IT=$((OBJ_ITERS + 3000))
      # grid_wcap is 8, so an object below 8 views keeps a share of generated surface even
      # where it was observed. Loud here, because the fusion cannot tell you afterwards.
      [ "${NM}" -lt 8 ] && echo "  [warn ${gid}] ${NM} views < grid_wcap 8"
      [ -f "${OBJ}/${gid}/point_cloud/iteration_${IT}/point_cloud.ply" ] && continue
      echo "  [train ${gid}] views=${NM} iters=${IT}"
      python train.py -s "${D}" -m "${OBJ}/${gid}" --iterations "${IT}" --is_instance \
        -r "${RESOLUTION}" --data_device "${DATA_DEVICE}" \
        --disable_viewer --lambda_dist "${LDIST}" --lambda_normal "${LNORM}" \
        --gt_depth_dir "${GTD}" --lambda_gtdepth 0.5 --front_mult 3.0 \
        || { echo "    FAILED"; continue; }
    done
    [ "${ITER}" = "${OBJ_ITERS}" ] || echo "  [note] ITER=${ITER} but objects trained to ${OBJ_ITERS}+"
  fi
fi

# ---------------------------------------------------------------- name
if want name; then
  say "name: GT class per object"
  # Captions change the completion: without names.tsv every object is generated from
  # "a 3D object in a room". Not optional, even though name_objects.py may fail on some.
  if fresh "${OBJ}/names.tsv" name_objects.py "${GT_MESH}"; then echo "  up to date"
  elif [ -f "${GT_MESH}" ] && [ -d "${OBJ}" ]; then
    python name_objects.py --gt_mesh "${GT_MESH}" --gt_info "${GT_INFO}" \
      --root "${OBJ}" --iter "${ITER}" || true
  fi
  [ -f "${OBJ}/names.tsv" ] || echo "  WARN no names.tsv -- generation falls back to generic text"
fi

# ---------------------------------------------------------------- mesh (side A)
if want mesh; then
  say "mesh: per-object TSDF -> fuse_post.ply (side A)"
  # Same tool as cond, weaker filters, so the only difference between the reported
  # surface and the conditioning surface is one line of arguments.
  for MDIR in "${OBJ}"/*/; do
    gid=$(basename "${MDIR}")
    [[ "${gid}" =~ ^[0-9]+$ ]] || continue
    [ -z "${ONLY}" ] || [[ " ${ONLY} " == *" ${gid} "* ]] || continue
    [ -f "${MDIR}point_cloud/iteration_${ITER}/point_cloud.ply" ] || continue
    MSH=${MDIR}train/ours_${ITER}/fuse_post.ply
    fresh "${MSH}" mesh_tsdf_views.py "${MDIR}point_cloud" \
      && { echo "  [${gid}] reuse"; continue; }
    python mesh_tsdf_views.py -m "${MDIR%/}" --load_iteration "${ITER}" \
      --out "${MSH}" ${MESH_ARGS} 2>&1 \
      | grep -E "^\[comp\]|^\[out\]|kept" | tail -3 | sed "s/^/  [${gid}] /"
    [ -f "${MSH}" ] || echo "  [${gid}] FAILED -- no side A for this object"
  done
fi

# ---------------------------------------------------------------- cond
if want cond; then
  say "cond: conditioning surface (${COND_NAME})"
  # NOT the surface we report. fuse_post.ply keeps the rough band where observation runs
  # out; it sits ON the surface, so make_shaper_input.py's free-space filter passes it and
  # ShapeR anchors to it. Filtering it out moved obj6 unseen F@2 0.5942 -> 0.6382.
  for MDIR in "${OBJ}"/*/; do
    gid=$(basename "${MDIR}")
    [[ "${gid}" =~ ^[0-9]+$ ]] || continue
    [ -z "${ONLY}" ] || [[ " ${ONLY} " == *" ${gid} "* ]] || continue
    [ -f "${MDIR}point_cloud/iteration_${ITER}/point_cloud.ply" ] || continue
    CLN=${MDIR}train/ours_${ITER}/${COND_NAME}
    fresh "${CLN}" mesh_tsdf_views.py "${MDIR}point_cloud" && { echo "  [${gid}] reuse"; continue; }
    python mesh_tsdf_views.py -m "${MDIR%/}" --load_iteration "${ITER}" \
      --out "${CLN}" ${COND_ARGS} 2>&1 | grep -E "^\[out\]|kept" | tail -2 | sed "s/^/  [${gid}] /"
    [ -f "${CLN}" ] || echo "  [${gid}] FAILED -- pkl will report a missing recon"
  done
fi

# ---------------------------------------------------------------- pkl / field / fuse
for ph in pkl field fuse; do
  want "${ph}" || continue
  say "${ph}"
  # RECON_NAME is what makes the cond stage count: without it the batch script falls back
  # to fuse_post.ply and that stage is dead weight. RUN is shared so all three phases land
  # in one log dir; FAILCSV is per phase because the batch script truncates it on entry.
  PRIOR="${PRIOR}" ITER="${ITER}" OUT="${OBJ}" CSV="${CSV}" PHASE="${ph}" \
    RUN="${RUN}" FAILCSV="${RUNDIR}/failures_${ph}.csv" LOGDIR="${RUNDIR}/logs" \
    PKL_SUBDIR="${PKL_SUBDIR}" CAPTIONS="${OBJ}/names.tsv" COLMAP="${COLMAP}" \
    MASKS="${MASKS}" IMAGES="${IMAGES}" GT_MESH="${GT_MESH}" GTD="${GTD}" ONLY="${ONLY}" \
    CARVE_DEPTH="$([ -d "${CARVE_DEPTH}" ] && echo "${CARVE_DEPTH}")" \
    POINTS_FROM="${POINTS_FROM}" RECON_NAME="${COND_NAME}" FUSE_EXTRA="${FUSE_EXTRA}" \
    SHAPER_DIR="${SHAPER_DIR}" SCENE="${SCENE}" ROOT="${ROOT}" \
    bash run_field_fusion_batch.sh || exit 1
done

# ---------------------------------------------------------------- report
say "done  run=${RUN}"
{
  echo ""
  echo "objects   $(count_dirs "${OBJ}" numeric) extracted"
  [ -f "${LABEL_DIR}/id_map.json" ] && python - "${LABEL_DIR}/id_map.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"labels    {m['K']} kept, {len(m.get('rare_gids', []))} dropped below "
      f"min_views {m.get('min_views')}")
PY
  [ -f "${CSV}" ] && echo "results   ${CSV}"
} | tee -a "${MANIFEST}"
echo "manifest  ${MANIFEST}"