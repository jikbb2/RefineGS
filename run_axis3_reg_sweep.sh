#!/usr/bin/env bash
# RefineGS Axis 3 — regularization sweep (lever 4: reg strength + long training)
# train -> mesh -> eval per variant, per instance. Resumable (skips existing).
#
# Usage:
#   bash run_axis3_reg_sweep.sh                      # default instances
#   INSTANCES="3 17 42" bash run_axis3_reg_sweep.sh  # override
#
# Prereqs: conda env split_and_splat, tools/eval_object_mesh.py,
#          GT_MESH pointing at Replica room_0 habitat/mesh_semantic.ply
set -uo pipefail

SCENE=replica_room0
DATA_ROOT=data/${SCENE}/masks
OUT_ROOT=output/${SCENE}/axis3_sweep
GT_MESH=${GT_MESH:-data/replica_gt/room_0/habitat/mesh_semantic.ply}
# 2~3 representative instances: pick large/medium/small from your 130
INSTANCES=${INSTANCES:-"REPLACE_ME"}   # e.g. "3 17 42"
GT_MAP=${GT_MAP:-}                      # optional CSV label,gt_id; else auto-match

# variant name | iterations | lambda_dist | lambda_normal
VARIANTS=(
  "reg_off        7000   0     0"
  "reg_base       7000   100   0.05"
  "reg_strong     7000   300   0.05"
  "reg_strong_l   30000  300   0.05"
  "reg_vstrong_l  30000  1000  0.05"
)

if [ "$INSTANCES" = "REPLACE_ME" ]; then
  echo "Set INSTANCES (e.g. INSTANCES=\"3 17 42\" bash $0)"; exit 1
fi

for spec in "${VARIANTS[@]}"; do
  read -r NAME ITERS LDIST LNORM <<< "$spec"
  for ID in $INSTANCES; do
    MDIR=${OUT_ROOT}/${NAME}/${ID}
    MESH=${MDIR}/train/ours_${ITERS}/fuse_post.ply

    if [ ! -f "${MDIR}/point_cloud/iteration_${ITERS}/point_cloud.ply" ]; then
      echo "=== train ${NAME} / ${ID} (iters=${ITERS} dist=${LDIST} normal=${LNORM}) ==="
      python train.py -s ${DATA_ROOT}/${ID} -m ${MDIR} \
        --iterations ${ITERS} --is_instance --disable_viewer \
        --lambda_dist ${LDIST} --lambda_normal ${LNORM} \
        || { echo "TRAIN FAILED ${NAME}/${ID}"; continue; }
    fi

    if [ ! -f "${MESH}" ]; then
      echo "=== mesh ${NAME} / ${ID} ==="
      python render.py -m ${MDIR} -s ${DATA_ROOT}/${ID} \
        --iteration ${ITERS} --skip_test --depth_ratio 1 --depth_trunc 5.0 \
        --voxel_size 0.004 --sdf_trunc 0.02 --num_cluster 1 \
        || { echo "MESH FAILED ${NAME}/${ID}"; continue; }
    fi
  done

  echo "=== eval ${NAME} ==="
  MAPARG=""; [ -n "${GT_MAP}" ] && MAPARG="--map ${GT_MAP}"
  python tools/eval_object_mesh.py batch --gt_mesh "${GT_MESH}" \
    --recon_glob "${OUT_ROOT}/${NAME}/*/train/ours_*/fuse_post.ply" \
    --label_from_path -4 --auto_match ${MAPARG} \
    --out ${OUT_ROOT}/results_${NAME}.csv
done

# ---- comparison table ----
python - <<'EOF'
import csv, glob, os
rows = {}
taus = None
for f in sorted(glob.glob("output/replica_room0/axis3_sweep/results_*.csv")):
    name = os.path.basename(f)[8:-4]
    with open(f) as fp:
        rs = [r for r in csv.DictReader(fp) if r.get("chamfer_l1")]
    if not rs: continue
    if taus is None:
        taus = [k for k in rs[0] if k.startswith("f@")]
    rows[name] = {
        "n": len(rs),
        "chamfer(mm)": 1000*sum(float(r["chamfer_l1"]) for r in rs)/len(rs),
        "NC": sum(float(r["normal_consistency"]) for r in rs)/len(rs),
        **{t: sum(float(r[t]) for r in rs)/len(rs) for t in taus},
    }
if rows:
    keys = list(next(iter(rows.values())).keys())
    print(f"\n{'variant':<16}" + "".join(f"{k:>14}" for k in keys))
    for name, v in rows.items():
        print(f"{name:<16}" + "".join(f"{v[k]:>14.4f}" if isinstance(v[k], float) else f"{v[k]:>14}" for k in keys))
EOF
