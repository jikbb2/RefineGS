#!/usr/bin/env bash
# 전 객체 ShapeR 필드 융합 배치 (핸드오프 7절 1번).
#
#   pkl   : make_shaper_input.py       (env: split_and_splat)
#   field : shaper_field.py            (env: shaper — cuDNN 충돌 회피 위해 LD_LIBRARY_PATH 비움)
#   fuse  : sdf_distill_depth.py --prior_field  +  eval_seen_unseen.py   (env: split_and_splat)
#
# env 가 둘로 갈리므로 단계별 실행을 지원한다.
#   PHASE=pkl   bash run_field_fusion_batch.sh     # split_and_splat 에서
#   PHASE=field bash run_field_fusion_batch.sh     # split_and_splat 에서 conda run 으로 shaper 호출
#   PHASE=fuse  bash run_field_fusion_batch.sh     # split_and_splat 에서
#   PHASE=all   bash run_field_fusion_batch.sh     # 전부 (conda run 사용)
# conda run 이 말썽이면 PHASE=field 만 shaper env 를 직접 activate 해서 돌려도 된다
# (그때는 SHAPER_DIRECT=1 로 두면 conda run 없이 현재 env 로 실행).
set -uo pipefail
shopt -s nullglob

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
ITER=${ITER:-7000}
PRIOR=${PRIOR:-$HOME/prior}
SHAPER_DIR=${SHAPER_DIR:-$HOME/ShapeR}
SHAPER_ENV=${SHAPER_ENV:-shaper}
SHAPER_DIRECT=${SHAPER_DIRECT:-0}
COLMAP=${COLMAP:-${ROOT}/data/${SCENE}/sparse/0}
IMAGES=${IMAGES:-${ROOT}/data/${SCENE}/images}
MASKS=${MASKS:-${ROOT}/data/${SCENE}/masks}
STEMS_DIR=${STEMS_DIR:-$HOME/See3D/dataset/stage6/clean_stems}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
GT_MESH=${GT_MESH:-$HOME/room_0/habitat/mesh_semantic.ply}
CAPTIONS=${CAPTIONS:-${ROOT}/captions.tsv}     # 선택: "gid<TAB>caption" 줄. 없으면 기본 문구
NPTS=${NPTS:-20000}
GRID=${GRID:-256}
# ── obj6 튜닝에서 확정된 생성 설정 ────────────────────────────────────────
# CFG   : ShapeR 에 구현돼 있으나 infer_shape.py 가 안 넘겨 비활성이던 값.
#         켜면 mode-averaging(다리가 사다리/그물로 뭉개짐)이 사라진다. 5 가 최적.
# 관측필터: recon 의 미관측 영역 쓰레기를 조건 포인트에서 제외(ShapeR 는 포인트를
#         geometric anchor 로 충실히 따르므로, 안 거르면 그 오류가 생성물로 전파).
# 부유물 : 필드의 음수 연결성분 중 최대 성분 대비 min_comp_frac 미만 제거.
CFG=${CFG:-5}
MIN_COMP_FRAC=${MIN_COMP_FRAC:-0.02}
# 앙상블: ab_ensemble.sh 로 --prior_field 만 바꾼 격리 A/B 결과(obj6) —
#   unseen P@2cm 0.6054 → 0.6872 (+8.2%p), unseen acc 24.52 → 23.24mm.
#   seen 비용은 acc +0.047mm, F@1cm -0.0008 로 노이즈 수준. 생성 시간만 3배.
#   (같은 A/B 에서 σ 가중은 세 번째 자리만 움직여 무효 → prior_sigma_w 0 유지)
#   평균/중앙값 결합은 얇은 구조를 지우므로 combine 은 best 만 쓴다.
ENSEMBLE=${ENSEMBLE:-3}
COMBINE=${COMBINE:-best}
# 포인트 샘플링 시드 — 고정해야 재실행/설정비교가 재현된다.
# (이게 없으면 같은 명령도 매번 다른 조건 포인트를 만들어 결과가 흔들린다)
SEED=${SEED:-0}
BOUNDS_MARGIN=${BOUNDS_MARGIN:-1.15}
SEEN_MARGIN=${SEEN_MARGIN:-0.02}
SEEN_MIN_VIEWS=${SEEN_MIN_VIEWS:-2}
FREE_POINTS=${FREE_POINTS:-0}                  # 생성 단계 free 구속(0=off, 효과 미미했음)
GUIDE_FREE_W=${GUIDE_FREE_W:-0}
PHASE=${PHASE:-all}
ONLY=${ONLY:-}                                 # 예: ONLY="1 6 11" 이면 해당 gid 만
# GT 라벨 자동 매칭 임계값. SAM3 인스턴스는 데이터셋 semantic id 와 1:1 이 아니라
# 한 객체가 여러 id 에 걸친다(obj1: id9 81% + id70/18/71/8 각 5%).
# 기본 0.10 이면 5%대가 전부 탈락 → GT 과소 매칭 → baseline seen acc 4.64mm 가 26mm 로 왜곡.
MATCH_MIN_SHARE=${MATCH_MIN_SHARE:-0.03}
# ── 융합 파라미터는 여기 없다 ────────────────────────────────────────────
# 확정된 값(min_unknown_frac 0.20, free_min_views 2, hull_min_frac 0, unseen_open 0,
# obs_erode 2, obs_cos_min 0.2, obs_cos_weight on, grid_wcap 5.0, voxel_size 0.005,
# keep_connected on, prior_sigma_w 0 …)은 전부 sdf_distill_depth.py 의 기본값이며,
# 매 실행 로그 맨 위 [config] 표에 '지정/기본' 출처와 함께 찍힌다.
#
# 실험할 값만 FUSE_EXTRA 로 덧붙인다:
#   FUSE_EXTRA="--grid_wcap 3" bash run_field_fusion_batch.sh
#   FUSE_EXTRA="--prior_sigma_w 1.0" PHASE=fuse bash run_field_fusion_batch.sh
FUSE_EXTRA=${FUSE_EXTRA:-}

CSV=${CSV:-${OUT}/_field_batch.csv}
FAILCSV=${FAILCSV:-${OUT}/_field_batch_failures.csv}
LOGDIR=${LOGDIR:-${PRIOR}/logs}
mkdir -p "${PRIOR}" "${LOGDIR}" "${SHAPER_DIR}/data"

cd "${ROOT}"

# ---- 대상 객체 수집 ----
gids=()
for MDIR in ${OUT}/*/; do
  gid=$(basename "${MDIR}")
  [[ "${gid}" =~ ^[0-9]+$ ]] || continue
  [ -f "${MDIR}train/ours_${ITER}/fuse_post.ply" ] || continue
  [ -d "${MASKS}/${gid}/masks" ] || continue
  if [ -n "${ONLY}" ]; then
    [[ " ${ONLY} " == *" ${gid} "* ]] || continue
  fi
  gids+=("${gid}")
done
echo "대상 객체 ${#gids[@]}개: ${gids[*]}"
echo "설정: n_points=${NPTS} seed=${SEED} bounds_margin=${BOUNDS_MARGIN} grid=${GRID}"
echo "      cfg=${CFG} min_comp_frac=${MIN_COMP_FRAC} ensemble=${ENSEMBLE}/${COMBINE}"
echo "      융합 파라미터는 sdf_distill_depth.py 기본값 (로그의 [config] 표 참조)"
[ -n "${FUSE_EXTRA}" ] && echo "      FUSE_EXTRA=${FUSE_EXTRA}"
echo "      관측필터 |z-d|<${SEEN_MARGIN}m×${SEEN_MIN_VIEWS}뷰, free_points=${FREE_POINTS}"
if [ -f "${CAPTIONS}" ]; then
  echo "      캡션 ${CAPTIONS} ($(grep -c . "${CAPTIONS}")줄)"
else
  echo "      캡션 파일 없음(${CAPTIONS}) — 전 객체가 기본 문구 사용."
  echo "        ShapeR 는 T5/CLIP 텍스트 조건을 받으므로 영향 가능성은 있으나,"
  echo "        시드 고정 전이라 캡션 효과를 단독으로 검증한 적은 없습니다."
fi
[ ${#gids[@]} -gt 0 ] || { echo "대상 없음"; exit 1; }

caption_of() {  # gid → caption
  local g=$1
  if [ -f "${CAPTIONS}" ]; then
    local c; c=$(awk -F'\t' -v g="$g" '$1==g{print $2; exit}' "${CAPTIONS}")
    [ -n "${c}" ] && { echo "${c}"; return; }
  fi
  echo "a 3D object in a room"
}

note_fail() { echo "$1,$2,$3" >> "${FAILCSV}"; }
echo "gid,stage,detail" > "${FAILCSV}"          # 매 실행 초기화(이전 실행 잔여물 혼동 방지)

# 실패 원인이 로그 파일에만 남아 안 보이는 것을 막는다 — 꼬리를 즉시 출력
show_tail() {
  local f=$1 n=${2:-15}
  [ -f "${f}" ] || { echo "      (로그 없음: ${f})"; return; }
  echo "      ---- ${f} (마지막 ${n}줄) ----"
  tail -n "${n}" "${f}" | sed 's/^/      /'
  echo "      ------------------------------"
}

# ---------------- PHASE: pkl ----------------
if [ "${PHASE}" = "pkl" ] || [ "${PHASE}" = "all" ]; then
  echo "=== [1/3] ShapeR 입력 pkl 생성 (n_points=${NPTS}) ==="
  for gid in "${gids[@]}"; do
    RECON=${OUT}/${gid}/train/ours_${ITER}/fuse_post.ply
    STEMS=${STEMS_DIR}/${gid}.txt
    CAP=$(caption_of "${gid}")
    echo "  [${gid}] caption='${CAP}'"
    python make_shaper_input.py --gid "${gid}" --n_points "${NPTS}" \
      --seed "${SEED}" --bounds_margin "${BOUNDS_MARGIN}" \
      --recon "${RECON}" --colmap "${COLMAP}" --images "${IMAGES}" \
      --masks_root "${MASKS}" ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      --depth_dir "${GTD}" --seen_margin "${SEEN_MARGIN}" \
      --seen_min_views "${SEEN_MIN_VIEWS}" --free_points "${FREE_POINTS}" \
      --caption "${CAP}" --out "${SHAPER_DIR}/data/obj${gid}.pkl" \
      > "${LOGDIR}/pkl_${gid}.log" 2>&1
    if [ ! -f "${SHAPER_DIR}/data/obj${gid}.pkl" ]; then
      echo "    pkl 실패"; note_fail "${gid}" pkl "make_shaper_input"
      show_tail "${LOGDIR}/pkl_${gid}.log"
    else
      grep -h "^\[filter\]\|^\[free\]\|^\[frame\]\|^\[views\]" \
        "${LOGDIR}/pkl_${gid}.log" | sed 's/^/    /'
    fi
  done
fi

# ---------------- PHASE: field ----------------
if [ "${PHASE}" = "field" ] || [ "${PHASE}" = "all" ]; then
  echo "=== [2/3] ShapeR 부호 필드 추출 (grid=${GRID}) ==="
  for gid in "${gids[@]}"; do
    PKL=${SHAPER_DIR}/data/obj${gid}.pkl
    NPZ=${PRIOR}/obj${gid}_field.npz
    [ -f "${PKL}" ] || { echo "  [skip ${gid}] pkl 없음"; note_fail "${gid}" field "pkl 없음"; continue; }
    # [stale prior 가드] 기존 npz 를 무조건 skip 하면, ENSEMBLE 을 켜도 예전 단일 샘플
    # prior 를 조용히 재사용한다(파일명이 같아서 눈치채기 어렵다). 앙상블 산출물은
    # field_std 를 갖고 있으므로 그걸로 구분해 필요하면 다시 만든다.
    if [ -f "${NPZ}" ]; then
      if [ "${ENSEMBLE}" -gt 1 ] && \
         ! python -c "import numpy,sys; sys.exit(0 if 'field_std' in numpy.load(sys.argv[1]).files else 1)" "${NPZ}" 2>/dev/null; then
        echo "  [${gid}] 기존 필드가 단일 샘플(field_std 없음) — 앙상블로 재생성"
        mv -f "${NPZ}" "${NPZ%.npz}_single.npz"
      else
        echo "  [skip ${gid}] 필드 이미 있음"; continue
      fi
    fi
    echo "  [${gid}] field 추출"
    CMD="cd '${SHAPER_DIR}' && LD_LIBRARY_PATH= python shaper_field.py \
         --input_pkl data/obj${gid}.pkl --config balance --grid ${GRID} \
         --cfg ${CFG} --min_comp_frac ${MIN_COMP_FRAC} \
         $([ "${ENSEMBLE}" -gt 1 ] && echo --ensemble ${ENSEMBLE} --combine ${COMBINE}) \
         $([ "${GUIDE_FREE_W}" != "0" ] && echo --guide_free_w ${GUIDE_FREE_W}) \
         --out '${NPZ}'"
    if [ "${SHAPER_DIRECT}" = "1" ]; then
      bash -c "${CMD}" > "${LOGDIR}/field_${gid}.log" 2>&1
    else
      # conda 를 비대화형 셸에서 쓰려면 conda.sh 를 source 해야 하는 경우가 많다
      CONDA_BASE=$(conda info --base 2>/dev/null)
      if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        bash -c "source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${SHAPER_ENV}' && ${CMD}" \
          > "${LOGDIR}/field_${gid}.log" 2>&1
      else
        conda run -n "${SHAPER_ENV}" bash -c "${CMD}" \
          > "${LOGDIR}/field_${gid}.log" 2>&1
      fi
    fi
    if [ ! -f "${NPZ}" ]; then
      echo "    field 실패"; note_fail "${gid}" field "shaper_field"
      show_tail "${LOGDIR}/field_${gid}.log" 20
    else
      grep -h "^\[field\]\|^\[floater\]\|^\[ensemble\]\|^  \[guide\]" \
        "${LOGDIR}/field_${gid}.log" | sed 's/^/    /'
    fi
  done
fi

# ---------------- PHASE: fuse + eval ----------------
if [ "${PHASE}" = "fuse" ] || [ "${PHASE}" = "all" ]; then
  echo "=== [3/3] 융합 + seen/unseen 평가 ==="
  rm -f "${CSV}"
  ok=0; ng=0
  for gid in "${gids[@]}"; do
    MDIR=${OUT}/${gid}
    OUTD=${MDIR}/train/ours_${ITER}
    NPZ=${PRIOR}/obj${gid}_field.npz
    STEMS=${STEMS_DIR}/${gid}.txt
    [ -f "${NPZ}" ] || { echo "  [skip ${gid}] 필드 없음"; note_fail "${gid}" fuse "필드 없음"; ng=$((ng+1)); continue; }

    # 융합은 객체당 수 분~십수 분. 진행 상황은 로그로만 보이므로 경로를 안내한다
    echo "  [${gid}] 융합  (진행: tail -f ${LOGDIR}/fuse_${gid}.log)"
    # 확정 파라미터는 전부 sdf_distill_depth.py 의 기본값이다. 여기서는 경로와
    # '아직 실험 중인 값'만 넘긴다. 실제 적용값은 로그 맨 위 [config] 표에 찍힌다.
    python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
      --prior_field "${NPZ}" --gt_depth_dir "${GTD}" \
      --out "${OUTD}/fused_field.ply" \
      ${FUSE_EXTRA} \
      > "${LOGDIR}/fuse_${gid}.log" 2>&1 \
      || { echo "    융합 실패"; note_fail "${gid}" fuse "sdf_distill"; \
           show_tail "${LOGDIR}/fuse_${gid}.log" 20; ng=$((ng+1)); continue; }

    echo "  [${gid}] 평가 (GT 라벨 자동 매칭)"
    python eval_seen_unseen.py --gt_mesh "${GT_MESH}" \
      --recon "${OUTD}/fuse_post.ply" --recon2 "${OUTD}/fused_field_post.ply" \
      --colmap "${COLMAP}" --gid "${gid}" \
      --masks_root "${MASKS}" --use_mask \
      ${STEMS:+$([ -f "${STEMS}" ] && echo --stems "${STEMS}")} \
      --match_min_share "${MATCH_MIN_SHARE}" \
      --tag "obj${gid}" --csv_all --csv "${CSV}" \
      > "${LOGDIR}/eval_${gid}.log" 2>&1 \
      || { echo "    평가 실패"; note_fail "${gid}" eval "eval_seen_unseen"; \
           show_tail "${LOGDIR}/eval_${gid}.log" 20; ng=$((ng+1)); continue; }
    # GT 라벨 자동 매칭 결과를 요약에 남긴다 (매칭이 나쁘면 지표 해석이 무의미)
    grep -h "auto-match\|채택 라벨\|커버리지 낮음" "${LOGDIR}/eval_${gid}.log" \
      | sed "s/^/    [${gid}] /"
    ok=$((ok+1))
  done
  echo ""
  echo "융합/평가 완료: 성공 ${ok}, 실패 ${ng}"
fi

# ---------------- 요약 ----------------
if [ -f "${CSV}" ]; then
  echo ""
  echo "=== 객체별 seen/unseen 요약 (${CSV}) ==="
  python - "${CSV}" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.OrderedDict()
for r in rows:
    by.setdefault(r["tag"], []).append(r)
hdr = ["obj", "seenAcc A→B", "seenF1 A→B", "unsComp A→B", "unsF2 A→B", "free% A→B"]
print("  " + "  ".join(f"{h:>18}" for h in hdr))
agg = collections.defaultdict(list)
for tag, rs in by.items():
    if len(rs) < 2:
        continue
    a, b = rs[0], rs[1]                      # --csv_all 이면 A(baseline), B(ours) 순
    def p(k, f="{:.3f}"):
        try: return f.format(float(a[k])) + "→" + f.format(float(b[k]))
        except Exception: return "-"
    print("  " + "  ".join(f"{v:>18}" for v in [
        tag, p("seen_acc", "{:.2f}"), p("seen_F1.0"), p("unseen_comp", "{:.1f}"),
        p("unseen_F2.0"), p("free_pct", "{:.2f}")]))
    for k in ("seen_acc", "seen_F1.0", "unseen_comp", "unseen_F2.0", "free_pct"):
        try: agg[k].append((float(a[k]), float(b[k])))
        except Exception: pass
print("\n  === 평균 (객체 %d개) ===" % len(agg.get("unseen_F2.0", [])))
for k, lab in (("seen_acc", "seen accuracy(mm)"), ("seen_F1.0", "seen F@1cm"),
               ("unseen_comp", "unseen completion(mm)"), ("unseen_F2.0", "unseen F@2cm"),
               ("free_pct", "free 위반(%)")):
    v = agg.get(k, [])
    if not v: continue
    A = sum(x for x, _ in v) / len(v); B = sum(y for _, y in v) / len(v)
    print(f"  {lab:>24}: {A:8.3f} → {B:8.3f}  ({B - A:+.3f})")
PY
fi
if [ -s "${FAILCSV}" ] && [ "$(wc -l < "${FAILCSV}")" -gt 1 ]; then
  echo ""; echo "=== 실패 목록 (${FAILCSV}) ==="
  awk -F, '{printf "  %-6s %-8s %s\n", $1, $2, $3}' "${FAILCSV}"
fi
