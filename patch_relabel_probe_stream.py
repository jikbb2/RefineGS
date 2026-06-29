#!/usr/bin/env python3
"""sam3_relabel_video.py 에 'streaming detection' probe 삽입.

PROBE_CONCEPT 환경변수가 있으면: 그 concept를 frame 0 에만 프롬프트하고
propagate → 프레임별 obj_id 추이를 출력하고 즉시 종료.
  - 후반 프레임에서 NEW id가 생기면 → SAM3가 late 인스턴스를 자동 검출(streaming)
  - id 개수가 frame 0 수준에서 안 늘면 → frame 0에 없는 객체는 못 잡음

기존 셋업(프레임/predictor/idx2stem)을 그대로 재사용. 멱등.
실행: python patch_relabel_probe_stream.py
사용: PROBE_CONCEPT=cushion <relabel 실행>
"""
import sys
F = "sam3_relabel_video.py"
s = open(F).read()
if "PROBE single-prompt" in s:
    print("이미 적용됨 — 건너뜀"); sys.exit(0)

anchor = ('        sid=predictor.handle_request(dict(type="start_session",resource_path=tmp))["session_id"]\n'
          '        for c in VOCAB:')

probe = (
'        sid=predictor.handle_request(dict(type="start_session",resource_path=tmp))["session_id"]\n'
'        import os as _os\n'
'        _pc=_os.environ.get("PROBE_CONCEPT")\n'
'        if _pc:\n'
'            print(f"=== PROBE single-prompt concept={_pc!r} (frame 0 only) ===")\n'
'            predictor.handle_request(dict(type="reset_session",session_id=sid))\n'
'            predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=0,text=_pc))\n'
'            _opf=propagate(sid)\n'
'            _prev=set(); _maxn=0\n'
'            for _fi in sorted(_opf):\n'
'                _ids=set(int(x) for x in np.asarray(_opf[_fi]["out_obj_ids"]).reshape(-1))\n'
'                _new=_ids-_prev; _maxn=max(_maxn,len(_ids))\n'
'                if _fi==0 or _new or _fi%50==0:\n'
'                    print(f"  frame {_fi} ({idx2stem[_fi]}): n_ids={len(_ids)} ids={sorted(_ids)[:12]}"+(f"  NEW={sorted(_new)}" if _new else ""))\n'
'                _prev|=_ids\n'
'            print(f"=== 총 등장 ID={len(_prev)}, 동시최대={_maxn}.  후반 NEW 있으면 streaming detection ===")\n'
'            sys.exit(0)\n'
'        for c in VOCAB:')

assert anchor in s, "anchor(start_session→VOCAB 루프) 못 찾음 — 파일 버전 확인"
s = s.replace(anchor, probe, 1)
open(F, "w").write(s)
print("patched: PROBE_CONCEPT streaming probe 삽입")
