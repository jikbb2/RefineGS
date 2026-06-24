#!/usr/bin/env python3
"""fuse_carve.py 에 --no_silhouette (free-space only veto) 추가.

이유: silhouette(visual-hull) veto는 '관측 실루엣 밖 = 위반'이라, amodal 생성이
미관측 영역으로 확장한 부분(가구 밑 러그, 소파 뒷면 등)을 전부 제거한다.
free-space veto만 쓰면 관측 깊이에 모순되는(표면 앞) 기하만 제거하고,
뒤/밑의 미관측 확장은 보존된다.

멱등. 실행: python patch_fuse_carve_nosilhouette.py
"""
import sys
F = "fuse_carve.py"
s = open(F).read()
if "use_silhouette" in s:
    print("이미 패치됨 — 건너뜀"); sys.exit(0)

# 1) carve_faces 시그니처
a = ("def carve_faces(occ_mesh, recon_mesh, cameras, masks, margin=0.02,\n"
     "                use_freespace=True):")
b = ("def carve_faces(occ_mesh, recon_mesh, cameras, masks, margin=0.02,\n"
     "                use_freespace=True, use_silhouette=True):")
assert a in s, "carve_faces 시그니처 못 찾음"; s = s.replace(a, b, 1)

# 2) silhouette 계산을 조건부로
a = "    viol_sil, _ = silhouette_violating_vertices(gv, cameras, masks)\n"
b = ("    if use_silhouette:\n"
     "        viol_sil, _ = silhouette_violating_vertices(gv, cameras, masks)\n"
     "    else:\n"
     "        viol_sil = np.zeros(len(gv), dtype=bool)\n")
assert a in s, "viol_sil 라인 못 찾음"; s = s.replace(a, b, 1)

# 3) fuse_carve 시그니처
a = ("def fuse_carve(recon_path, gen_path, out_path, colmap_dir, masks_dir,\n"
     "               occ_thresh=0.05, margin=0.02, use_freespace=True,\n"
     "               gt_mesh_path=None, gt_id=None, verbose=True):")
b = ("def fuse_carve(recon_path, gen_path, out_path, colmap_dir, masks_dir,\n"
     "               occ_thresh=0.05, margin=0.02, use_freespace=True,\n"
     "               use_silhouette=True, gt_mesh_path=None, gt_id=None, verbose=True):")
assert a in s, "fuse_carve 시그니처 못 찾음"; s = s.replace(a, b, 1)

# 4) carve_faces 호출에 use_silhouette 전달
a = ("    kept_mesh, n_in, n_kept, n_vsil, n_vfs = carve_faces(\n"
     "        occ_mesh, recon, cameras, masks, margin=margin,\n"
     "        use_freespace=use_freespace)")
b = ("    kept_mesh, n_in, n_kept, n_vsil, n_vfs = carve_faces(\n"
     "        occ_mesh, recon, cameras, masks, margin=margin,\n"
     "        use_freespace=use_freespace, use_silhouette=use_silhouette)")
assert a in s, "carve_faces 호출 못 찾음"; s = s.replace(a, b, 1)

# 5) --no_silhouette 인자 추가
a = ('    ap.add_argument("--no_freespace", action="store_true",\n'
     '                    help="disable free-space(depth) veto, silhouette only")')
b = (a + "\n"
     '    ap.add_argument("--no_silhouette", action="store_true",\n'
     '                    help="disable silhouette(visual-hull) veto, free-space only "\n'
     '                         "(keeps amodal completion into unobserved regions)")')
assert a in s, "--no_freespace 인자 못 찾음"; s = s.replace(a, b, 1)

# 6) 모든 fuse_carve(...) 호출에 use_silhouette 전달 (single + batch)
a = "use_freespace=not args.no_freespace,"
b = "use_freespace=not args.no_freespace, use_silhouette=not args.no_silhouette,"
cnt = s.count(a)
s = s.replace(a, b)
print(f"  fuse_carve 호출 {cnt}곳에 use_silhouette 전달")

open(F, "w").write(s)
print("fuse_carve.py 패치 완료: --no_silhouette (free-space only) 추가")
