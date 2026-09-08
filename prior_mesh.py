#!/usr/bin/env python3
"""prior 필드(npz)를 world 좌표 메쉬로 뽑는다 — 생성물을 눈으로 보기 위한 도구.

왜 필요한가:
  융합 결과만 보면 '왜 안 채워졌는지'가 융합 탓인지 생성 탓인지 구분되지 않는다.
  실측 obj10(항아리): 게이트가 unknown 6.4% 로 prior 를 차단했는데, 게이트를 강제
  통과시켜도 빈 반쪽이 채워지지 않았다 → prior 자체가 그 부분을 만들지 않은 것.
  게이트는 'prior 표면 중 미관측에 있는 비율'을 재므로 옳게 보고한 것이고,
  문제는 생성 단계였다.

  이 스크립트로 prior 메쉬를 뽑아 fuse_post.ply 와 함께 띄우면 즉시 갈린다:
    · prior 가 온전한 물체다   → 융합/게이트 문제
    · prior 도 반쪽이다        → 생성 문제 (조건 포인트, bounds, CFG)

사용:
  python prior_mesh.py ~/prior/obj10_field.npz --out /tmp/obj10_prior.ply
  python prior_mesh.py ~/prior/obj10_field.npz --out /tmp/p.ply --level 0
"""
import argparse
import os

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=0.0, help="등위면 값(m). 0=표면")
    args = ap.parse_args()

    z = np.load(os.path.expanduser(args.npz))
    F = z["field"].astype(np.float32)
    G = F.shape[0]
    center, R, scale = z["center"], z["R_align"], float(z["scale"])
    vox = float(z["vox_world"])

    neg = float((F < args.level).mean())
    print(f"[prior] {os.path.basename(args.npz)}  G={G}  voxel={vox*1000:.2f}mm")
    print(f"  필드 범위 [{F.min():.4f}, {F.max():.4f}]m   등위면 {args.level}m 내부 {neg*100:.2f}%")
    if not (F.min() < args.level < F.max()):
        raise SystemExit(f"등위면 {args.level} 이 필드 범위 밖입니다 — --level 을 조정하세요")

    # 융합 쪽 정변환(sdf_distill_depth.py `_sd`):  n = (q - center) @ R.T * scale
    # 따라서 역변환은  q = (n / scale) @ R + center.  scale 은 '곱하는' 값이므로 나눈다.
    v, f, _, _ = marching_cubes(F, level=args.level, spacing=(2.0 / (G - 1),) * 3)
    v = (v - 1.0) / scale                      # [-1,1] → 정렬 좌표(미터)
    v = v @ R + center                         # → world

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    m.compute_vertex_normals()
    p = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    assert o3d.io.write_triangle_mesh(p, m), f"저장 실패: {p}"

    V = np.asarray(m.vertices)
    ext = V.max(0) - V.min(0)
    print(f"  정점 {len(V):,}  면 {len(np.asarray(m.triangles)):,}")
    print(f"  world bbox  min {np.round(V.min(0), 3)}  max {np.round(V.max(0), 3)}")
    print(f"              크기 {np.round(ext, 3)} m")
    # 좌표 변환 자체 검증: 그리드가 덮는 범위는 vox_world × G 여야 한다
    span = vox * (G - 1)
    if ext.max() > span * 1.05:
        print(f"  ⚠ bbox({ext.max():.3f}m)가 그리드 범위({span:.3f}m)보다 큽니다 "
              f"— 좌표 변환(scale 곱/나눗셈) 오류를 의심하세요")
    else:
        print(f"  (그리드 범위 {span:.3f}m — bbox 가 그 안이면 변환 정상)")

    # 축별 점유 — '어느 쪽이 비었는지'를 숫자로도 본다
    lv = " ▁▂▃▄▅▆▇█"
    for ax, nm in enumerate("xyz"):
        h, _ = np.histogram(V[:, ax], bins=16)
        bar = "".join("·" if c == 0 else lv[max(1, min(8, int(8 * c / h.max())))] for c in h)
        print(f"  {nm}축 |{bar}|  (· = 정점 없음)")
    print(f"\n→ {p}\n  fuse_post.ply 와 함께 MeshLab 에 띄워 비교하세요.")
    print("  prior 가 온전하면 융합 문제, prior 도 반쪽이면 생성 문제입니다.")


if __name__ == "__main__":
    main()
