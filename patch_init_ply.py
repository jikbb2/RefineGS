#!/usr/bin/env python3
"""patch_init_ply.py — train.py 에 --init_ply 주입 (B4a, 멱등).

효과: scene = Scene(...) 직후 조립 ply(base ⊕ gen surfels)를 load_ply 로 덮어쓰고
      active_sh_degree 를 max 로 올려(학습된 SH 즉시 사용) 공동 최적화 시작.
unseen 보호는 별도 코드 없이 실행 시 --densify_until_iter 0 로 (densify/prune/opacity_reset off).

실행: cd /home/elicer/RefineGS && python patch_init_ply.py
"""
import os

P = "train.py"
s = open(P).read()
orig = s

# 1) CLI 인자 추가
anchor_arg = 'parser.add_argument("--start_checkpoint", type=str, default=None)'
if "--init_ply" not in s and anchor_arg in s:
    s = s.replace(anchor_arg,
                  anchor_arg + '\n    parser.add_argument("--init_ply", type=str, default=None)  # [B4a]')

# 2) Scene 직후 load_ply 주입
anchor_scene = "    scene = Scene(dataset, gaussians)\n"
inject = (anchor_scene +
          "    if getattr(args, 'init_ply', None):  # [B4a] 조립 ply 로 init 덮어쓰기\n"
          "        gaussians.load_ply(args.init_ply)\n"
          "        gaussians.active_sh_degree = gaussians.max_sh_degree\n"
          "        print('[B4a] init from ' + args.init_ply + ': ' + str(gaussians.get_xyz.shape[0]) + ' gaussians')\n")
if "[B4a] init from" not in s and anchor_scene in s:
    s = s.replace(anchor_scene, inject, 1)

if s != orig:
    open(P + ".bak_b4a", "w").write(orig)
    open(P, "w").write(s)
    print("patched train.py (--init_ply). 백업: train.py.bak_b4a")
else:
    print("이미 패치됨 또는 anchor 불일치 — 수동 확인 필요")
    print("  '--init_ply' present:", "--init_ply" in s)
    print("  '[B4a] init from' present:", "[B4a] init from" in s)
