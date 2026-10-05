#!/usr/bin/env python3
"""sam3_relabel_video.py 패치 — A1 instance individuation 1단계.

(1) PROBE: concept별 SAM3 obj_id 개수 출력 → merge(SAM3가 1개) vs 정상(N개) 확정.
(2) FIX : cross-concept dedup이 *같은 concept* 인스턴스를 병합하지 않도록
          (within-concept은 native track이라 병합 금지 — 주석 의도와 코드 일치).

멱등. 실행(파일 있는 폴더에서): python patch_relabel_instance.py
"""
import sys

F = "sam3_relabel_video.py"
s = open(F).read()
if "A1-instance-patch" in s:
    print("이미 패치됨 — 건너뜀"); sys.exit(0)

# (1) PROBE 삽입
a1 = ('            for oid,d in byid.items():\n'
      '                if len(d["masks"])<args.min_track: continue')
b1 = ('            print(f"  [PROBE A1-instance-patch] concept={c!r} SAM3 obj_ids={len(byid)} '
      'ids={list(byid.keys())[:8]}")\n'
      '            for oid,d in byid.items():\n'
      '                if len(d["masks"])<args.min_track: continue')
assert a1 in s, "probe anchor(byid 루프) 못 찾음 — 파일 버전 확인"
s = s.replace(a1, b1, 1)

# (2) dedup 같은-concept 제외
a2 = ('        for m in merged:\n'
      '            if jac(t["sig"],m["sig"])>args.dedup_th: hit=m; break')
b2 = ('        for m in merged:\n'
      '            if t["concept"] in m["concepts"]:   # 같은 concept = 다른 인스턴스 → 병합 X\n'
      '                continue\n'
      '            if jac(t["sig"],m["sig"])>args.dedup_th: hit=m; break')
assert a2 in s, "dedup anchor 못 찾음 — 파일 버전 확인"
s = s.replace(a2, b2, 1)

open(F, "w").write(s)
print("patched: PROBE + dedup 같은-concept 제외")
