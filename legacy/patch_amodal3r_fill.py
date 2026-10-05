#!/usr/bin/env python3
"""patch_amodal3r_fill.py — prepare_amodal3r_input.py 에 --fill {amodal,hull,close} 추가 (멱등).

문제: 테이블 위 물건이 별도 인스턴스라 visible 마스크에 구멍 → occluded(0)가 안 찍힘 →
      Amodal3R가 그 자리를 배경으로 보고 "생선 뼈" 생성.
해결: filled(=객체 전체 footprint)를 visible의 convex hull(또는 morphological close)로 잡아
      occ = filled − visible 로 *물건 가린 자리를 occluded(0)* 로 표시 → 테이블로 완성.

실행: cd /home/elicer/RefineGS && python patch_amodal3r_fill.py
"""
P = "prepare_amodal3r_input.py"
s = open(P).read(); o = s

if "from scipy.ndimage import binary_closing" not in s:
    s = s.replace("import numpy as np",
                  "import numpy as np\n"
                  "from scipy.ndimage import binary_closing, binary_fill_holes\n"
                  "try:\n    import cv2\nexcept Exception:\n    cv2 = None")

arg_anchor = 'ap.add_argument("--out", default="/home/elicer/Amodal3R/input")'
if "--fill" not in s and arg_anchor in s:
    s = s.replace(arg_anchor,
                  arg_anchor +
                  '\n    ap.add_argument("--fill", choices=["amodal", "hull", "close"], default="amodal",\n'
                  '                    help="객체 footprint 정의: amodal(기존)/hull(convex)/close(morph)")\n'
                  '    ap.add_argument("--close_k", type=int, default=25)')

old = "        filled = to_bool(af) if os.path.exists(af) else vis\n"
new = ('        if a.fill == "hull" and cv2 is not None and vis.any():\n'
       '            ys, xs = np.where(vis); hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))\n'
       '            f8 = np.zeros(vis.shape, np.uint8); cv2.fillConvexPoly(f8, hull, 1); filled = f8.astype(bool)\n'
       '        elif a.fill == "close":\n'
       '            filled = binary_fill_holes(binary_closing(vis, np.ones((a.close_k, a.close_k))))\n'
       '        else:\n'
       '            filled = to_bool(af) if os.path.exists(af) else vis\n')
if "a.fill ==" not in s and old in s:
    s = s.replace(old, new)

if s != o:
    open(P + ".bak_fill", "w").write(o); open(P, "w").write(s)
    print("patched prepare_amodal3r_input.py (--fill). 백업: .bak_fill")
else:
    print("이미 패치됨 또는 anchor 불일치 — 확인:",
          "--fill" in s, "a.fill ==" in s)
