#!/usr/bin/env bash
# [1a 검증] grid 융합의 GPU 경로가 CPU 경로와 같은 결과를 내는지, 얼마나 빠른지.
#
# 왜 이 단계를 따로 검증하나:
#   1b(제약 SDF 최적화)를 얹기 전에 이식이 정확한지 확정해야, 이후 성능 변화가
#   '최적화 덕분'인지 '이식 오류' 인지 구분된다. 이 프로젝트에서 교란된 비교로
#   여러 번 잘못된 결론을 냈다(앙상블 효과, 관측가중, unseen_open).
#
# 기대치:
#   · 정수 카운터(free 투표, hull 등)는 완전 일치 — 오프라인 전사 검증 완료
#   · TSDF 합은 float32(GPU) vs float64(CPU) 차이로 ~1e-7 (trunc 0.05 대비 3e-6)
#   · 메쉬 정점 수는 몇 개 다를 수 있다(영교차가 임계에 걸친 복셀). 표면 거리가
#     복셀 크기의 1% 이내면 동일한 것으로 본다.
#   · 속도: 실측 CPU 550~1700s/객체 → GPU 수십 초 기대
#
# 사용: bash verify_gpu_fuse.sh            # obj6
#       GID=22 bash verify_gpu_fuse.sh
set -uo pipefail

ROOT=${ROOT:-$HOME/RefineGS}
SCENE=${SCENE:-replica_room0_v2}
GID=${GID:-6}
ITER=${ITER:-7000}
PRIOR=${PRIOR:-$HOME/prior}
OUT=${OUT:-${ROOT}/output/${SCENE}/refinegs_full}
GTD=${GTD:-/home/elicer/nice-slam/Datasets/Replica/room0/results}
NPZ=${NPZ:-${PRIOR}/obj${GID}_field.npz}
LOGDIR=${LOGDIR:-${PRIOR}/logs}
mkdir -p "${LOGDIR}"
cd "${ROOT}" || exit 1
[ -f "${NPZ}" ] || { echo "prior 없음: ${NPZ}"; exit 1; }

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null \
  | sed 's/^/[GPU] /' || echo "[GPU] nvidia-smi 없음"
echo ""

run () {   # $1 = cpu|cuda
  local D=$1 t0=$(date +%s)
  echo -n "  ${D} ... "
  python sdf_distill_depth.py -m "${OUT}/${GID}" --iteration ${ITER} \
    --prior_field "${NPZ}" --gt_depth_dir "${GTD}" \
    --fuse_device "${D}" \
    --out "/tmp/vg_${GID}_${D}.ply" \
    > "${LOGDIR}/vg_${GID}_${D}.log" 2>&1 \
    || { echo "실패"; tail -15 "${LOGDIR}/vg_${GID}_${D}.log"; return 1; }
  echo "$(( $(date +%s)-t0 ))s"
  grep -h "^\[fuse-gpu\]\|^\[gate\]\|^\[grid-fuse\] 관측복셀\|^\[sanity\]" \
    "${LOGDIR}/vg_${GID}_${D}.log" | sed 's/^/      /'
}

echo "=== obj${GID} CPU vs GPU ==="
run cuda || exit 1
run cpu  || exit 1

echo ""
python - "/tmp/vg_${GID}_cuda_post.ply" "/tmp/vg_${GID}_cpu_post.ply" <<'PY'
import sys, numpy as np, open3d as o3d
A = o3d.io.read_triangle_mesh(sys.argv[1]); B = o3d.io.read_triangle_mesh(sys.argv[2])
va, vb = np.asarray(A.vertices), np.asarray(B.vertices)
print(f"정점  GPU {len(va):,}  CPU {len(vb):,}  차이 {abs(len(va)-len(vb)):,} "
      f"({abs(len(va)-len(vb))/max(len(vb),1)*100:.3f}%)")
if not len(va) or not len(vb):
    sys.exit("메쉬가 비었습니다")
# 양방향 최근접 거리 — 표면이 실제로 같은 자리에 있는지
ta = o3d.t.geometry.RaycastingScene(); ta.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(A))
tb = o3d.t.geometry.RaycastingScene(); tb.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(B))
sa = np.asarray(A.sample_points_uniformly(100000).points, np.float32)
sb = np.asarray(B.sample_points_uniformly(100000).points, np.float32)
d_ab = tb.compute_distance(o3d.core.Tensor(sa)).numpy()
d_ba = ta.compute_distance(o3d.core.Tensor(sb)).numpy()
print(f"표면 거리 GPU→CPU  평균 {d_ab.mean()*1000:.4f}mm  최대 {d_ab.max()*1000:.3f}mm")
print(f"          CPU→GPU  평균 {d_ba.mean()*1000:.4f}mm  최대 {d_ba.max()*1000:.3f}mm")
cd = (d_ab.mean() + d_ba.mean()) / 2 * 1000
print(f"\nChamfer {cd:.4f}mm  →  "
      + ("동일한 결과로 판단" if cd < 0.05 else
         "⚠ 차이가 큽니다 — 이식 오류를 의심할 것(복셀 5mm의 1%=0.05mm 기준)"))
PY
