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
# prior 적용 게이트: 생성 '표면' 중 unknown 비율이 이 값 미만이면 prior 미적용.
# 이미 충분히 관측된 객체(배치 실측 obj16/28/10)에서 prior 가 손해만 보는 것을 막는다.
MIN_UNKNOWN_FRAC=${MIN_UNKNOWN_FRAC:-0.20}
FREE_MIN_VIEWS=${FREE_MIN_VIEWS:-2}

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
echo "설정: n_points=${NPTS} grid=${GRID} cfg=${CFG} min_comp_frac=${MIN_COMP_FRAC}"
echo "      관측필터 |z-d|<${SEEN_MARGIN}m×${SEEN_MIN_VIEWS}뷰, free_points=${FREE_POINTS}"
echo "      융합 min_unknown_frac=${MIN_UNKNOWN_FRAC} free_min_views=${FREE_MIN_VIEWS}"
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
    [ -f "${NPZ}" ] && { echo "  [skip ${gid}] 필드 이미 있음"; continue; }
    echo "  [${gid}] field 추출"
    CMD="cd '${SHAPER_DIR}' && LD_LIBRARY_PATH= python shaper_field.py \
         --input_pkl data/obj${gid}.pkl --config balance --grid ${GRID} \
         --cfg ${CFG} --min_comp_frac ${MIN_COMP_FRAC} \
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
    python sdf_distill_depth.py -m "${MDIR}" --iteration ${ITER} \
      --data_device cpu --mask_dir auto --require_mask --mask_dist 0 \
      --prior_field "${NPZ}" --prior_sigma_w 0 \
      --grid_fuse --alpha_smooth 1.0 --color_blend_ramp 0.05 \
      --prior_carve_views 150 --free_min_views "${FREE_MIN_VIEWS}" --num_cluster 10000 \
      --min_unknown_frac "${MIN_UNKNOWN_FRAC}" \
      --voxel_size 0.005 --max_grid 512 --keep_connected \
      --gt_depth_dir "${GTD}" \
      --out "${OUTD}/fused_field.ply" \
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
