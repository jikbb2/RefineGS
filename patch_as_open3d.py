#!/usr/bin/env python3
"""patch_as_open3d.py — utils/mesh_utils.py 의 `mesh = mesh.as_open3d` 를 수동 변환으로 교체 (멱등).

trimesh 버전에 as_open3d 가 없어서 extract_mesh_unbounded(=--unbounded)가 죽는 문제 수정.
trimesh(vertices/faces/vertex_colors) → open3d.geometry.TriangleMesh 직접 구성.

실행: cd /home/elicer/RefineGS && python patch_as_open3d.py
"""
import re

P = "utils/mesh_utils.py"
s = open(P).read()
orig = s

TARGET = "mesh = mesh.as_open3d"
if "as_open3d 수동변환" in s:
    print("이미 패치됨")
elif TARGET in s:
    # 해당 줄의 들여쓰기 캡처
    m = re.search(r"^([ \t]*)mesh = mesh\.as_open3d\b.*$", s, re.M)
    indent = m.group(1)
    repl = (
        f"{indent}# as_open3d 수동변환 (trimesh 버전 호환)\n"
        f"{indent}import open3d as _o3d, numpy as _np\n"
        f"{indent}_om = _o3d.geometry.TriangleMesh()\n"
        f"{indent}_om.vertices = _o3d.utility.Vector3dVector(_np.asarray(mesh.vertices))\n"
        f"{indent}_om.triangles = _o3d.utility.Vector3iVector(_np.asarray(mesh.faces))\n"
        f"{indent}_vc = getattr(mesh.visual, 'vertex_colors', None)\n"
        f"{indent}if _vc is not None and len(_vc) == len(mesh.vertices):\n"
        f"{indent}    _om.vertex_colors = _o3d.utility.Vector3dVector(_np.asarray(_vc)[:, :3].astype(_np.float64) / 255.0)\n"
        f"{indent}mesh = _om"
    )
    s = re.sub(r"^[ \t]*mesh = mesh\.as_open3d\b.*$", repl, s, count=1, flags=re.M)
    open(P + ".bak_o3d", "w").write(orig)
    open(P, "w").write(s)
    print(f"patched {P} (as_open3d 수동변환). 백업: {P}.bak_o3d")
else:
    print(f"'{TARGET}' 못 찾음 — 수동 확인 필요")
    import subprocess; print(subprocess.run(["grep", "-n", "as_open3d", P], capture_output=True, text=True).stdout)
