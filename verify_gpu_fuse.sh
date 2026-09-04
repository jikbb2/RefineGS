#!/usr/bin/env bash
# [1a 검증] grid 융합의 GPU 경로가 CPU 경로와 같은 결과를 내는지, 얼마나 빠른지.
#
# 왜 이 단계를 따로 검증하나:
#   1b(제약 SDF 최적화)를 얹기 전에 이식이 정확한지 확정해야, 이후 성능 변화가
#   '최적화 덕분'인지 '이식 오류' 인지 구분된다. 이 프로젝트에서 교란된 비교로
#   여러 번 잘못된 결론을 냈다(앙상블 효과, 관측가중, unseen_open).
#
# 실측 결과 (obj6, float64):
#   CPU↔CPU  Chamfer 0.0497mm   ← 같은 명령 2회. 이게 파이프라인의 재현성 폭이다.
#   CPU↔GPU  Chamfer 0.0585mm   ← 1.18배. 이식 정확으로 판정.
#   (float32 일 때는 0.196mm = 노이즈의 4배였다 → 그래서 기본을 float64 로)
#   속도 523s → 116s (4.5배). 50~100배는 슬랩 루프만의 수치였고, 뷰 렌더링·sd_fn
#   보간·marching cubes·연결성분이 CPU 에 남아 있어 전체는 이만큼이다.
#
# ⚠ 노이즈 바닥이 0 이 아닌 이유: 상류 2DGS 렌더가 CUDA 원자연산·타일 정렬 때문에
#   비트 단위로 재현되지 않는다. 시드를 고정해도 게이트가 40.4/40.2/39.6% 로 흔들린다.
#   따라서 설정 A/B 는 이 폭보다 큰 차이만 해석해야 한다.
#   (최대 거리는 판정에 쓰지 말 것 — CPU↔CPU 에서도 22mm 이상치가 나온다. 한쪽에만
#    생긴 작은 부유물이며 평균에는 거의 영향이 없다)
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
: > /tmp/_vg_cd.txt

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

echo "=== obj${GID} 자기 재현성 + CPU vs GPU ==="
# ⚠ 먼저 '같은 설정 2회'의 차이를 재야 한다. 그게 이 파이프라인의 노이즈 바닥이고,
#   그보다 작은 CPU↔GPU 차이는 이식 오류가 아니다.
#   (실측: 시드를 고정해도 CPU 게이트가 38.4→38.8→40.0 으로 변한다. 2DGS 래스터라이저
#    렌더가 CUDA 원자연산·타일 정렬 때문에 비트 단위로 재현되지 않는 것으로 보인다.)
run cpu  || exit 1
mv -f "/tmp/vg_${GID}_cpu_post.ply" "/tmp/vg_${GID}_cpuA_post.ply"
run cpu  || exit 1
mv -f "/tmp/vg_${GID}_cpu_post.ply" "/tmp/vg_${GID}_cpuB_post.ply"
run cuda || exit 1

echo ""
echo "--- (1) 노이즈 바닥: CPU vs CPU (같은 설정 2회) ---"
python - "/tmp/vg_${GID}_cpuA_post.ply" "/tmp/vg_${GID}_cpuB_post.ply" <<'PY'
import sys, numpy as np, open3d as o3d
A = o3d.io.read_triangle_mesh(sys.argv[1]); B = o3d.io.read_triangle_mesh(sys.argv[2])
va, vb = np.asarray(A.vertices), np.asarray(B.vertices)
print(f"정점  1회차 {len(va):,}  2회차 {len(vb):,}  차이 {abs(len(va)-len(vb)):,} "
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
print(f"표면 거리 A→B  평균 {d_ab.mean()*1000:.4f}mm  최대 {d_ab.max()*1000:.3f}mm")
print(f"          B→A  평균 {d_ba.mean()*1000:.4f}mm  최대 {d_ba.max()*1000:.3f}mm")
cd = (d_ab.mean() + d_ba.mean()) / 2 * 1000
print(f"Chamfer {cd:.4f}mm")
open("/tmp/_vg_cd.txt", "a").write(f"{cd:.6f}\n")
PY

echo ""
echo "--- (2) CPU vs GPU ---"
python - "/tmp/vg_${GID}_cpuA_post.ply" "/tmp/vg_${GID}_cuda_post.ply" <<'PY'
import sys, numpy as np, open3d as o3d
A = o3d.io.read_triangle_mesh(sys.argv[1]); B = o3d.io.read_triangle_mesh(sys.argv[2])
va, vb = np.asarray(A.vertices), np.asarray(B.vertices)
print(f"정점  CPU {len(va):,}  GPU {len(vb):,}  차이 {abs(len(va)-len(vb)):,} "
      f"({abs(len(va)-len(vb))/max(len(va),1)*100:.3f}%)")
ta = o3d.t.geometry.RaycastingScene(); ta.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(A))
tb = o3d.t.geometry.RaycastingScene(); tb.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(B))
sa = np.asarray(A.sample_points_uniformly(100000).points, np.float32)
sb = np.asarray(B.sample_points_uniformly(100000).points, np.float32)
d_ab = tb.compute_distance(o3d.core.Tensor(sa)).numpy()
d_ba = ta.compute_distance(o3d.core.Tensor(sb)).numpy()
cd = (d_ab.mean() + d_ba.mean()) / 2 * 1000
print(f"표면 거리 CPU→GPU 평균 {d_ab.mean()*1000:.4f}mm / GPU→CPU 평균 {d_ba.mean()*1000:.4f}mm")
print(f"Chamfer {cd:.4f}mm")
open("/tmp/_vg_cd.txt", "a").write(f"{cd:.6f}\n")
PY

echo ""
python - <<'PY'
vals = [float(x) for x in open("/tmp/_vg_cd.txt")][-2:]
noise, gap = vals
print(f"노이즈 바닥(CPU↔CPU) {noise:.4f}mm   |   CPU↔GPU {gap:.4f}mm   "
      f"비율 {gap/max(noise,1e-9):.2f}×")
print("판정: " + ("이식 정확 — 차이가 파이프라인 자체의 재현성 폭 안에 있다"
                if gap <= 2 * noise else
                "⚠ 노이즈의 2배를 넘음 — 이식 오류를 의심할 것"))
print("\n※ 노이즈 바닥이 0 이 아니라면 융합 상류(2DGS 렌더)가 비결정적이라는 뜻이다.")
print("  그 경우 설정 A/B 는 이 폭보다 큰 차이만 해석해야 한다.")
PY
rm -f /tmp/_vg_cd.txt.bak; : > /tmp/_vg_cd.txt
