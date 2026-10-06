#!/usr/bin/env bash
# ShapeR field fusion over every object.
#
#   pkl   : make_shaper_input.py                                  (env: refinegs)
#   field : shaper_field.py                     (env: shaper; LD_LIBRARY_PATH cleared to
#                                                avoid a cuDNN clash)
#   fuse  : sdf_distill_depth.py --prior_field + eval_seen_unseen.py  (env: refinegs)
#
# The two envs are why this runs in phases:
#   PHASE=pkl|field|fuse|eval|all  bash run_field_fusion_batch.sh
#
#   PHASE=eval re-runs ONLY eval_seen_unseen.py on the fused meshes already on disk. The
#   ShapeR field and the fusion are the expensive parts and neither depends on how the GT
#   instance is matched, so a change to the matching costs minutes per scene, not hours.
# SHAPER_DIRECT=1 runs the field phase in the current env instead of via conda.
#
# Fusion parameters are NOT set here. They are defaults in sdf_distill_depth.py and are
# printed, with their origin, in the [config] table at the top of every fusion log. Pass
# only what you are testing:
#   FUSE_EXTRA="--grid_wcap 3" PHASE=fuse bash run_field_fusion_batch.sh
set -uo pipefail
shopt -s nullglob

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-}
[ -n "${SCENE}" ] || { echo "[abort] SCENE is empty -- set SCENE=<name>"; exit 1; }
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
ITER=${ITER:-7000}
PRIOR=${PRIOR:-${ROOT}/output/${SCENE}/prior}
SHAPER_DIR=${SHAPER_DIR:-$HOME/ShapeR}
# PKL_SUBDIR separates the pipelines: without it, the per-object and scene runs write the same
# obj<gid>.pkl, and the stale-prior guard then rebuilds one pipeline's field from the
# other's input, silently.
PKL_SUBDIR=${PKL_SUBDIR:-}
SHAPER_ENV=${SHAPER_ENV:-shaper}
SHAPER_DIRECT=${SHAPER_DIRECT:-0}
# shaper_field.py and infer_shape_pinhole.py are OURS -- cloning ShapeR does not produce
# them, which is why a fresh checkout of this repository could not run the field phase at
# all. They live here and run against the ShapeR checkout: sys.path[0] is their directory,
# so shaper_field.py's `import infer_shape_pinhole` finds the copy beside it, and
# SHAPER_DIR on PYTHONPATH supplies dataset.* and model.*. shaper_field.py chdir's into
# --shaper_root itself, because ShapeR resolves checkpoints/ relative to the working
# directory. Keep ONLY those two files in SHAPER_PY's directory: a dataset.py or model.py
# there would shadow ShapeR's own packages.
SHAPER_PY=${SHAPER_PY:-${ROOT}/shaper/shaper_field.py}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
IMAGES=${IMAGES:-${ROOT}/data/${SCENE}/images}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}

# Dataset-dependent defaults, derived from SCENE rather than hard-coded.
#
# GTD and GT_MESH used to default to room0's absolute paths whatever SCENE said, so a
# standalone run on room2 silently evaluated room2's reconstruction against room0's GT
# depth and GT mesh. Nothing downstream checks that, and the numbers still look plausible
# -- the same failure run_scene.sh was written to prevent. Derive both from SCENE and
# abort when they are missing, instead of falling back to another room's data.
REPLICA_ROOT=${REPLICA_ROOT:-}
REPLICA_SEMANTIC=${REPLICA_SEMANTIC:-}
_scene=$(echo "${SCENE}" | sed -E 's/^replica_//')                            # room0
_scene_v1=$(echo "${_scene}" | sed -E 's/^(room|office)([0-9]+)$/\1_\2/')     # room_0
GTD=${GTD:-${REPLICA_ROOT:+${REPLICA_ROOT}/${_scene}/results}}
GT_MESH=${GT_MESH:-}
if [ -z "${GT_MESH}" ] && [ -n "${REPLICA_SEMANTIC}" ]; then
  for _c in "${REPLICA_SEMANTIC}/${_scene_v1}/habitat" "${REPLICA_SEMANTIC}/${_scene}/habitat"; do
    [ -f "${_c}/mesh_semantic.ply" ] && { GT_MESH=${_c}/mesh_semantic.ply; break; }
  done
fi

# Per-object view lists produced by the per-object pipeline. When this directory is
# absent, --stems is simply not passed and the evaluation falls back to EVERY colmap view
# -- which changes the numbers without saying so. Warn loudly instead.
STEMS_DIR=${STEMS_DIR:-${ROOT}/data/${SCENE}/clean_stems}

# Free-space reference for the fusion. When set, the scene model's own rendered depth
# (dump_scene_depth.py) replaces GT depth, and the fuse phase passes no GT depth at all --
# that is the whole point, so it is all or nothing rather than a per-view fallback.
# The pkl phase still reads GTD for its seen/unseen test (make_shaper_input --depth_dir).
CARVE_DEPTH=${CARVE_DEPTH:-}
# ShapeR is text conditioned, so the caption changes the completion. name_objects.py
# writes <OUT>/names.tsv as "gid<TAB>class", which is exactly this format; falling back to
# "a 3D object in a room" asks the model to complete a generic blob.
CAPTIONS=${CAPTIONS:-$([ -f "${OUT}/names.tsv" ] && echo "${OUT}/names.tsv" \
                                                || echo "${ROOT}/captions.tsv")}
NPTS=${NPTS:-20000}
GRID=${GRID:-256}

# CFG 5: implemented in ShapeR but never passed by infer_shape.py. Enabling it removes
#   mode averaging, which had been turning table legs into a ladder or a net.
# MIN_COMP_FRAC: drop negative components smaller than this fraction of the largest.
CFG=${CFG:-5}
MIN_COMP_FRAC=${MIN_COMP_FRAC:-0.02}

# Ensemble 3 / combine=best, from an isolated A/B that changed only --prior_field (obj6):
#   unseen P@2cm 0.6054 -> 0.6872 (+8.2%p), unseen acc 24.52 -> 23.24mm. The seen cost was
#   +0.047mm and -0.0008 F@1cm, i.e. noise. Costs 3x generation time.
#   Mean/median combination erases thin structure, so only "best" is used.
#   The same A/B left sigma weighting in the third decimal, hence prior_sigma_w 0.
ENSEMBLE=${ENSEMBLE:-3}
COMBINE=${COMBINE:-best}

# Fixed seeds. Without them the same command draws different conditioning points and the
# generation drifts, which invalidates any A/B.
SEED=${SEED:-0}
EVAL_SEED=${EVAL_SEED:-0}
# The flow-matching initial noise. SEED above is the conditioning-point subsample, a
# different thing, and shaper_field.py was never given a seed at all -- it ran on its own
# default of 0. Separate names, so changing one cannot silently move the other.
FIELD_SEED=${FIELD_SEED:-0}

BOUNDS_MARGIN=${BOUNDS_MARGIN:-1.15}
SEEN_MARGIN=${SEEN_MARGIN:-0.02}
SEEN_MIN_VIEWS=${SEEN_MIN_VIEWS:-2}
FREE_POINTS=${FREE_POINTS:-0}                # free-space constraint at generation (0=off)
# mesh: sample the reconstruction (density-bound).  depth: back-project masked pixels.
POINTS_FROM=${POINTS_FROM:-mesh}
GUIDE_FREE_W=${GUIDE_FREE_W:-0}
PHASE=${PHASE:-all}
ONLY=${ONLY:-}                               # e.g. ONLY="1 6 11"
FUSE_EXTRA=${FUSE_EXTRA:-}
PKL_FORCE=${PKL_FORCE:-0}                    # 1 = rebuild the pkl even if it is current
# Which reconstruction conditions the generation. filter_observed drops points that miss
# the GT depth, so free-floating skirt is already removed -- but the rough grazing-angle
# band sits ON the surface and passes, and ShapeR anchors to it. tsdf_clean.ply is the
# same TSDF with that band filtered out (mesh_tsdf_views.py).
RECON_NAME=${RECON_NAME:-fuse_post.ply}
# Every invocation gets its own tag, so no output is ever written over an earlier one and
# the numbers already reported stay derivable. Pass RUN=<tag> to resume a run instead.
RUN=${RUN:-$(date +%m%d_%H%M)}
FUSE_NAME=${FUSE_NAME:-fused_${RUN}}

# GT label auto-matching. A SAM3 instance is not 1:1 with a dataset semantic id -- one
# object spans several (obj1: id9 81% plus four ids at ~5%). At the old 0.10 threshold the
# 5% ones all dropped out, under-matching the GT and inflating baseline seen acc from
# 4.64mm to 26mm.
# 0.10, which is what eval_seen_unseen.py documents as its own default. It was overridden
# to 0.03 here with no note, and that is half of how the GT ended up containing neighbouring
# instances: a label holding 3% of the vote was unioned in, and the bbox-gap guard meant to
# catch that reports 0mm for anything the object touches. The other half is fixed in
# eval_seen_unseen.py, which now also requires that we reconstructed most of the instance.
MATCH_MIN_SHARE=${MATCH_MIN_SHARE:-0.10}

# The fuse phase resets the CSV, so an ONLY= run would replace the full-batch results
# with its handful of rows. Subsets get their own file.
CSV=${CSV:-${OUT}/_field_${RUN}.csv}
FAILCSV=${FAILCSV:-${OUT}/_field_${RUN}_failures.csv}
LOGDIR=${LOGDIR:-${PRIOR}/logs/${RUN}}
PKL_DIR=${PKL_DIR:-${ROOT}/output/${SCENE}/prior/pkl}
PKL_REL=${PKL_DIR}
mkdir -p "${PRIOR}" "${LOGDIR}" "${PKL_DIR}"
cd "${ROOT}" || exit 1

# The GT mesh is read by eval_seen_unseen.py and by nothing else here: it supplies the
# per-face object_id and the visibility oracle. Demanding it for every phase is what used to
# make a pkl or field run impossible without one, and the way past it was to point GT_MESH at
# some other scene's file -- which evaluates one room against another and still looks
# plausible. So: required for PHASE=eval, and otherwise the fusion runs and the evaluation is
# skipped. Reconstruction needs no GT; the seen/unseen metrics do.
HAVE_GT_MESH=0
[ -n "${GT_MESH}" ] && [ -f "${GT_MESH}" ] && HAVE_GT_MESH=1
if [ "${HAVE_GT_MESH}" -eq 0 ]; then
  if [ "${PHASE}" = "eval" ]; then
    echo "[abort] PHASE=eval needs a GT mesh for SCENE=${SCENE} (looked for"
    echo "        ${_scene_v1}/habitat and ${_scene}/habitat under"
    echo "        REPLICA_SEMANTIC='${REPLICA_SEMANTIC}')."
    echo "        Set REPLICA_SEMANTIC, or set GT_MESH explicitly."
    exit 1
  fi
  echo "  note: no GT mesh -- fusion will run, the seen/unseen evaluation will be SKIPPED"
fi
[ -d "${GTD}" ] || [ -n "${CARVE_DEPTH}" ] || {
  echo "[abort] no GT depth directory for SCENE=${SCENE}: ${GTD}"
  echo "        Set GTD explicitly, or set CARVE_DEPTH to use rendered scene depth."
  exit 1; }
# A missing stems directory is not fatal, but it silently switches the evaluation to every
# colmap view, which moves the numbers. Say so once, up front.
[ -d "${STEMS_DIR}" ] || echo "  WARN no stems dir (${STEMS_DIR}) -- evaluation will use" \
                              "ALL colmap views, not the per-object view list"

gids=()
for MDIR in ${OUT}/*/; do
  gid=$(basename "${MDIR}")
  [[ "${gid}" =~ ^[0-9]+$ ]] || continue
  [ -f "${MDIR}train/ours_${ITER}/fuse_post.ply" ] || continue
  [ -d "${MASKS}/${gid}/masks" ] || continue
  [ -z "${ONLY}" ] || [[ " ${ONLY} " == *" ${gid} "* ]] || continue
  gids+=("${gid}")
done
[ ${#gids[@]} -gt 0 ] || { echo "no target object under ${OUT}"; exit 1; }
echo "targets (${#gids[@]}): ${gids[*]}   run=${RUN}"
echo "  out=${OUT} iter=${ITER} prior=${PRIOR} pkl=${PKL_DIR}"
echo "  scene=${SCENE} -> gt_mesh=${GT_MESH}"
echo "  carve=$([ -n "${CARVE_DEPTH}" ] && echo "rendered scene depth ${CARVE_DEPTH}" || echo "GT depth ${GTD}")"
echo "  recon=${RECON_NAME} -> ${FUSE_NAME}_post.ply   grid=${GRID} cfg=${CFG} ensemble=${ENSEMBLE}/${COMBINE}${FUSE_EXTRA:+   ${FUSE_EXTRA}}"
echo "  field=${SHAPER_PY}   shaper_root=${SHAPER_DIR}   seed=${FIELD_SEED} (pkl seed=${SEED})"
echo "  eval=$([ "${HAVE_GT_MESH}" -eq 1 ] && echo "on" || echo "SKIPPED -- no GT mesh")"
[ -f "${CAPTIONS}" ] || echo "  WARN no caption file (${CAPTIONS}); generating from generic text"
# RUN tags every output, so nothing here can overwrite an earlier result; a field npz is
# shared on purpose and the stale guard moves the old one aside rather than deleting it.

name_of() { [ -f "${CAPTIONS}" ] && awk -F'\t' -v g="$1" '$1==g{print $2; exit}' "${CAPTIONS}"; }

caption_of() {
  local g=$1 c
  if [ -f "${CAPTIONS}" ]; then
    c=$(awk -F'\t' -v g="$g" '$1==g{print $2; exit}' "${CAPTIONS}")
    [ -n "${c}" ] && { echo "${c}"; return; }
  fi
  echo "a 3D object in a room"
}

note_fail() { echo "$1,$2,$3" >> "${FAILCSV}"; }
echo "gid,stage,detail" > "${FAILCSV}"        # reset, so old failures cannot confuse

# Long stages write to a log file, so the terminal shows nothing for minutes and it is
# impossible to tell a slow run from a hung one. Run in the background and keep one line
# updated with the elapsed time and the log's last line.
PROGRESS_EVERY=${PROGRESS_EVERY:-5}
run_progress() {                              # run_progress LOG LABEL -- cmd...
  local log=$1 label=$2; shift 2
  "$@" > "${log}" 2>&1 &
  local pid=$! t0=$SECONDS cur
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${PROGRESS_EVERY}"
    # tqdm rewrites a single line with \r, so the file's last LINE holds every update
    # concatenated and deleting \r shows the FIRST one forever ("0%|" for the whole run).
    # Split on \r and take the last segment instead.
    # -a: a log with NUL or invalid UTF-8 makes grep print "binary file matches" and drop
    # the line entirely. Keep reading it as text.
    # The old sanitizer was `tr -cd '\11\40-\176'`, which keeps only ASCII -- and tqdm draws
    # its bar with Unicode blocks (U+2588 and friends). It deleted the FILLED part of every
    # bar and left the unfilled spaces, so "44%|#####    |" rendered as "44%|     |" and 100%
    # as "||": the bar looked wrong while the numbers were right. Strip ANSI escapes and
    # control characters instead, and leave every byte >= 0x80 alone.
    cur=$(tail -c 8192 "${log}" 2>/dev/null | tr '\r' '\n' | grep -a -v '^[[:space:]]*$' \
          | tail -1 | sed $'s/\033\\[[0-9;]*[a-zA-Z]//g' \
          | tr -d '\000-\010\013\014\016-\037\177' | cut -c1-72)
    printf "\r    %-18s %4ds  %-72s" "${label}" "$((SECONDS - t0))" "${cur}"
  done
  wait "${pid}"; local rc=$?
  printf "\r    %-18s %4ds  %-72s\n" "${label}" "$((SECONDS - t0))" \
         "$([ ${rc} -eq 0 ] && echo ok || echo FAILED)"
  return ${rc}
}

show_tail() {                                 # failures otherwise hide in the log file
  local f=$1 n=${2:-15}
  [ -f "${f}" ] && { echo "    ---- ${f} (last ${n}) ----";
                     tail -n "${n}" "${f}" | sed 's/^/    /'; } \
                || echo "    (no log: ${f})"
}

# ---------------- pkl ----------------
if [ "${PHASE}" = "pkl" ] || [ "${PHASE}" = "all" ]; then
  echo "=== [1/3] ShapeR input pkl ==="
  # make_shaper_input decides which reconstruction points count as observed. Same
  # reference as the carve, or the conditioning and the constraint disagree.
  if [ -n "${CARVE_DEPTH}" ]; then PKL_DEPTH="--carve_depth_dir ${CARVE_DEPTH}"
  else PKL_DEPTH="--depth_dir ${GTD}"; fi
  echo "  observation reference: ${PKL_DEPTH}"
  for gid in "${gids[@]}"; do
    RECON=${OUT}/${gid}/train/ours_${ITER}/${RECON_NAME}
    STEMS=${STEMS_DIR}/${gid}.txt
    # Rebuilding the pkl makes it newer than the npz, and the stale guard then regenerates
    # the field -- the most expensive stage. Rebuild only when the generator has changed,
    # OR when the conditioning surface it reads has.
    #
    # That second test was missing until 0929, and the gap is not theoretical: the driver's
    # COND_ARGS decides what ${RECON_NAME} contains, changing it correctly rebuilt that mesh,
    # and this guard then reused a pkl built from the PREVIOUS conditioning surface, so the
    # field, the fusion and every reported number still answered the old question. Nothing
    # printed a warning -- "pkl up to date" was the only sign.
    if [ "${PKL_FORCE}" = "0" ] && [ -f "${PKL_DIR}/obj${gid}.pkl" ] \
       && [ "${PKL_DIR}/obj${gid}.pkl" -nt make_shaper_input.py ] \
       && [ -f "${RECON}" ] && [ "${PKL_DIR}/obj${gid}.pkl" -nt "${RECON}" ]; then
      echo "  [${gid}] pkl up to date (PKL_FORCE=1 to rebuild)"; continue
    fi
    # These stems come from the per-object pipeline. If its gid numbering differs from
    # this OUT tree's, the conditioning points get filtered against another object's views.
    if [ -f "${STEMS}" ] && [ -d "${MASKS}/${gid}/masks" ]; then
      ns=$(wc -l < "${STEMS}"); nm=$(ls "${MASKS}/${gid}/masks" | wc -l)
      [ "${ns}" -gt 0 ] && [ $(( ns > nm ? ns - nm : nm - ns )) -gt $(( nm / 2 )) ] \
        && echo "  [${gid}] WARN stems ${ns} vs masks ${nm}: check they are the same object"
    fi
    python make_shaper_input.py --gid "${gid}" --n_points "${NPTS}" \
      --seed "${SEED}" --bounds_margin "${BOUNDS_MARGIN}" \
      --recon "${RECON}" --colmap "${COLMAP}" --images "${IMAGES}" \
      --masks_root "${MASKS}" ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      ${PKL_DEPTH} --seen_margin "${SEEN_MARGIN}" \
      --points_from "${POINTS_FROM}" \
      --seen_min_views "${SEEN_MIN_VIEWS}" --free_points "${FREE_POINTS}" \
      --caption "$(caption_of "${gid}")" --out "${PKL_DIR}/obj${gid}.pkl" \
      > "${LOGDIR}/pkl_${gid}.log" 2>&1
    if [ -f "${PKL_DIR}/obj${gid}.pkl" ]; then
      grep -hE "^\[filter\]|^\[frame\].*raw/robust|RELAXED" \
        "${LOGDIR}/pkl_${gid}.log" | sed 's/^/  /'
    else
      echo "  [${gid}] pkl FAILED ($([ -f "${RECON}" ] && echo "recon ok" || echo "no ${RECON}"))"
      note_fail "${gid}" pkl "make_shaper_input"
      show_tail "${LOGDIR}/pkl_${gid}.log"
    fi
  done
fi

# ---------------- field ----------------
if [ "${PHASE}" = "field" ] || [ "${PHASE}" = "all" ]; then
  # Without this the missing file surfaces as "field FAILED" plus a 20-line python traceback
  # per object, which reads like a model problem rather than a missing file.
  [ -f "${SHAPER_PY}" ] || {
    echo "[abort] no ${SHAPER_PY}"
    echo "        shaper_field.py belongs to THIS repository (cloning ShapeR does not"
    echo "        provide it). Put it, and infer_shape_pinhole.py, in ${ROOT}/shaper/,"
    echo "        or point SHAPER_PY at them."
    exit 1; }
  [ -f "$(dirname "${SHAPER_PY}")/infer_shape_pinhole.py" ] || {
    echo "[abort] no infer_shape_pinhole.py beside ${SHAPER_PY}"
    echo "        shaper_field.py imports it to bypass the fisheye rectify; without it the"
    echo "        pinhole images are unwarped as if they were Aria fisheye."
    exit 1; }
  [ -d "${SHAPER_DIR}" ] || {
    echo "[abort] SHAPER_DIR=${SHAPER_DIR} does not exist -- it holds ShapeR and its checkpoints/"
    exit 1; }
  echo "=== [2/3] ShapeR signed field (grid=${GRID}) ==="
  for gid in "${gids[@]}"; do
    PKL=${PKL_DIR}/obj${gid}.pkl
    NPZ=${PRIOR}/obj${gid}_field.npz
    [ -f "${PKL}" ] || { echo "  [${gid}] no pkl"; note_fail "${gid}" field "no pkl"; continue; }
    # Stale-prior guard. Skipping an existing npz silently reuses an old field, and the
    # filename does not change, so it is easy to miss: after fixing the visibility test we
    # rebuilt the pkl and three experiments came out identical to the decimal.
    #   1) pkl newer than npz -> the input changed, regenerate
    #   2) ENSEMBLE>1 but the npz has no field_std -> it is a single sample, regenerate
    if [ -f "${NPZ}" ]; then
      if [ "${PKL}" -nt "${NPZ}" ]; then
        mv -f "${NPZ}" "${NPZ%.npz}_stale.npz"
      elif [ "${ENSEMBLE}" -gt 1 ] && ! python -c \
          "import numpy,sys; sys.exit(0 if 'field_std' in numpy.load(sys.argv[1]).files else 1)" \
          "${NPZ}" 2>/dev/null; then
        mv -f "${NPZ}" "${NPZ%.npz}_single.npz"
      else
        continue
      fi
    fi
    # Run the repository's copy; shaper_field.py chdir's into --shaper_root itself, so there
    # is no `cd` here any more and nothing is written into the user's ShapeR checkout.
    CMD="LD_LIBRARY_PATH= PYTHONPATH='$(dirname "${SHAPER_PY}"):${SHAPER_DIR}' \
         python '${SHAPER_PY}' --shaper_root '${SHAPER_DIR}' \
         --input_pkl ${PKL_REL}/obj${gid}.pkl --config balance --grid ${GRID} \
         --cfg ${CFG} --min_comp_frac ${MIN_COMP_FRAC} --seed ${FIELD_SEED} \
         $([ "${ENSEMBLE}" -gt 1 ] && echo --ensemble ${ENSEMBLE} --combine ${COMBINE}) \
         $([ "${GUIDE_FREE_W}" != "0" ] && echo --guide_free_w ${GUIDE_FREE_W}) \
         --out '${NPZ}'"
    LBL="[${gid}] field$([ "${ENSEMBLE}" -gt 1 ] && echo " x${ENSEMBLE}")"
    if [ "${SHAPER_DIRECT}" = "1" ]; then
      run_progress "${LOGDIR}/field_${gid}.log" "${LBL}" bash -c "${CMD}"
    else
      CONDA_BASE=$(conda info --base 2>/dev/null)
      if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        run_progress "${LOGDIR}/field_${gid}.log" "${LBL}" bash -c \
          "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${SHAPER_ENV}' && ${CMD}"
      else
        run_progress "${LOGDIR}/field_${gid}.log" "${LBL}" \
          conda run -n "${SHAPER_ENV}" bash -c "${CMD}"
      fi
    fi
    [ -f "${NPZ}" ] || { echo "  [${gid}] field FAILED"; note_fail "${gid}" field "shaper_field";
                         show_tail "${LOGDIR}/field_${gid}.log" 20; }
  done
fi

# ---------------- fuse + eval ----------------
if [ "${PHASE}" = "fuse" ] || [ "${PHASE}" = "eval" ] || [ "${PHASE}" = "all" ]; then
  EVAL_ONLY=0; [ "${PHASE}" = "eval" ] && EVAL_ONLY=1
  echo "=== [3/3] $([ "${EVAL_ONLY}" = 1 ] && echo "re-evaluation only (fused meshes reused)" \
                                          || echo "fusion + seen/unseen evaluation") ==="
  # One reference, named in the log. sdf_distill_depth.py also auto-fills a default GT
  # depth dir when none is given, so the GT flag has to be absent, not merely unused.
  if [ -n "${CARVE_DEPTH}" ]; then
    [ -d "${CARVE_DEPTH}" ] || { echo "[abort] CARVE_DEPTH not a directory: ${CARVE_DEPTH}"; exit 1; }
    DEPTH_ARGS="--carve_depth_dir ${CARVE_DEPTH}"
  else
    DEPTH_ARGS="--gt_depth_dir ${GTD}"
  fi
  echo "  free-space reference: ${DEPTH_ARGS}"
  # Retire, do not delete. PHASE=eval rewrites this file from meshes that did not change,
  # so the previous CSV is the only record of what the previous evaluation said -- and the
  # evaluation is exactly the thing that has been changing. Deleting it makes "the numbers
  # moved" unverifiable after the fact.
  if [ -f "${CSV}" ]; then
    _old=${CSV%.csv}_$(date +%m%d_%H%M%S).csv
    mv "${CSV}" "${_old}" && echo "  previous results kept at $(basename "${_old}")"
  fi
  ok=0; ng=0
  for gid in "${gids[@]}"; do
    MDIR=${OUT}/${gid}; OUTD=${MDIR}/train/ours_${ITER}
    NPZ=${PRIOR}/obj${gid}_field.npz
    STEMS=${STEMS_DIR}/${gid}.txt
    if [ "${EVAL_ONLY}" = 1 ]; then
      # No field and no fusion: the mesh being evaluated is the one already written under
      # this RUN, so the numbers change only because the evaluation changed.
      [ -f "${OUTD}/${FUSE_NAME}_post.ply" ] \
        || { echo "  [${gid}] no ${FUSE_NAME}_post.ply -- run PHASE=fuse for this RUN first";
             note_fail "${gid}" eval "no fused mesh"; ng=$((ng+1)); continue; }
      echo "  [${gid}] $(name_of "${gid}") -- re-evaluating"
    else
      [ -f "${NPZ}" ] || { echo "  [${gid}] no field"; note_fail "${gid}" fuse "no field";
                           ng=$((ng+1)); continue; }
      run_progress "${LOGDIR}/fuse_${gid}.log" "[${gid}] $(name_of "${gid}")" \
        python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
        --prior_field "${NPZ}" ${DEPTH_ARGS} \
        --passthrough_mesh "${OUTD}/fuse_post.ply" \
        --out "${OUTD}/${FUSE_NAME}.ply" ${FUSE_EXTRA} \
        || { note_fail "${gid}" fuse "sdf_distill";
             show_tail "${LOGDIR}/fuse_${gid}.log" 20; ng=$((ng+1)); continue; }
    fi
    if [ "${HAVE_GT_MESH}" -eq 0 ]; then
      echo "    [${gid}] fused; eval skipped (no GT mesh)"
      ok=$((ok+1)); continue
    fi
    python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
      --recon "${OUTD}/fuse_post.ply" --recon2 "${OUTD}/${FUSE_NAME}_post.ply" \
      --colmap "${COLMAP}" --gid "${gid}" --masks_root "${MASKS}" --use_mask \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      --match_min_share "${MATCH_MIN_SHARE}" --seed "${EVAL_SEED}" \
      --tag "obj${gid}" --csv_all --csv "${CSV}" \
      > "${LOGDIR}/eval_${gid}.log" 2>&1 \
      || { echo "    eval FAILED"; note_fail "${gid}" eval "eval_seen_unseen";
           show_tail "${LOGDIR}/eval_${gid}.log" 20; ng=$((ng+1)); continue; }
    # A bad GT match makes the metrics meaningless, so keep that line visible.
    # [gt-check] is the line that says whether the carve reference is registered at all,
    # and grid-fuse prints two lines ahead of it -- head -3 used to cut it off.
    # PHASE=eval runs no fusion, so there is no fuse log to quote -- reading it anyway
    # printed a "No such file" line per object into the log the user actually reads.
    [ -f "${LOGDIR}/fuse_${gid}.log" ] \
      && grep -hE "^\[carve-src\]|^\[grid-fuse\] free|^\[gt-check\]" "${LOGDIR}/fuse_${gid}.log" \
         | head -4 | sed "s/^/    [${gid}] /"
    # the absolute distance matters more than the vote share: a recon 3 m from every GT
    # object still gets a confident-looking 95% match
    grep -hE "auto-match" "${LOGDIR}/eval_${gid}.log" | head -3 | sed "s/^/    [${gid}] /"
    ok=$((ok+1))
  done
  echo "fusion/eval done: ${ok} ok, ${ng} failed"
fi

# ---------------- summary ----------------
if [ -f "${CSV}" ]; then
  echo ""
  # In a phase that did not fuse, this table is the PREVIOUS run's CSV. We once ran
  # PHASE=pkl three times and read identical numbers before noticing.
  if [ "${PHASE}" != "fuse" ] && [ "${PHASE}" != "eval" ] && [ "${PHASE}" != "all" ]; then
    echo "WARN PHASE=${PHASE} did not fuse or evaluate."
    echo "     The table below is the previous run ($(date -r "${CSV}" '+%m-%d %H:%M'))."
  fi
  echo "=== per-object seen/unseen (${CSV}) ==="
  python - "${CSV}" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.OrderedDict()
for r in rows:
    by.setdefault(r["tag"], []).append(r)
hdr = ["obj", "seenAcc A->B", "seenF1 A->B", "unsComp A->B", "unsF2 A->B", "free% A->B"]
print("  " + "  ".join(f"{h:>18}" for h in hdr))
agg = collections.defaultdict(list)
for tag, rs in by.items():
    if len(rs) < 2:
        continue
    a, b = rs[0], rs[1]                      # with --csv_all: A (baseline), then B (ours)
    def p(k, f="{:.3f}"):
        try: return f.format(float(a[k])) + "->" + f.format(float(b[k]))
        except Exception: return "-"
    print("  " + "  ".join(f"{v:>18}" for v in [
        tag, p("seen_acc", "{:.2f}"), p("seen_F1.0"), p("unseen_comp", "{:.1f}"),
        p("unseen_F2.0"), p("free_pct", "{:.2f}")]))
    for k in ("seen_acc", "seen_F1.0", "unseen_comp", "unseen_F2.0", "free_pct"):
        try: agg[k].append((float(a[k]), float(b[k])))
        except Exception: pass
print("\n  === mean over %d objects ===" % len(agg.get("unseen_F2.0", [])))
for k, lab in (("seen_acc", "seen accuracy(mm)"), ("seen_F1.0", "seen F@1cm"),
               ("unseen_comp", "unseen completion(mm)"), ("unseen_F2.0", "unseen F@2cm"),
               ("free_pct", "free violation(%)")):
    v = agg.get(k, [])
    if not v:
        continue
    A = sum(x for x, _ in v) / len(v); B = sum(y for _, y in v) / len(v)
    print(f"  {lab:>24}: {A:8.3f} -> {B:8.3f}  ({B - A:+.3f})")
PY
fi
if [ -s "${FAILCSV}" ] && [ "$(wc -l < "${FAILCSV}")" -gt 1 ]; then
  echo ""; echo "=== failures (${FAILCSV}) ==="
  awk -F, 'NR>1{printf "  %-6s %-8s %s\n", $1, $2, $3}' "${FAILCSV}"
fi