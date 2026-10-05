#!/usr/bin/env python3
"""
SAM3 video predictor probe — raw output 구조 + 프레임 로딩 확인(블라인드 빌드 방지).

확인:
  1) .JPEG 프레임을 정수명(0.jpg,1.jpg,...)으로 심링크 → video predictor가 순서대로 읽나.
  2) add_prompt(text=concept) → propagate_in_video 의 출력 구조
     (frame_index → outputs; outputs 안의 obj_id ↔ mask 추출법, mask shape/타입).
  3) prepare_masks_for_visualization 출력 구조.

실행 (sam3 env):
    conda activate sam3
    LD_LIBRARY_PATH= python sam3_video_probe.py \
        --frames /home/elicer/RefineGS/data/replica_room0/images --img_ext .JPEG \
        --concept couch --max_frames 30 \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz
"""
import argparse, glob, os, tempfile
import numpy as np, torch


def describe(x, prefix="  ", depth=0):
    if depth > 3: return
    t = type(x).__name__
    if hasattr(x, "shape"):
        print(f"{prefix}{t} shape={tuple(x.shape)} dtype={getattr(x,'dtype',None)}")
    elif isinstance(x, dict):
        print(f"{prefix}dict keys={list(x.keys())}")
        for k, v in list(x.items())[:8]:
            print(f"{prefix}  [{k!r}] ->", end=" ")
            describe(v, prefix+"    ", depth+1)
    elif isinstance(x, (list, tuple)):
        print(f"{prefix}{t} len={len(x)}", "first:" if x else "")
        if x: describe(x[0], prefix+"    ", depth+1)
    else:
        s = str(x)
        print(f"{prefix}{t} = {s[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--img_ext", default=".JPEG")
    ap.add_argument("--concept", default="couch"); ap.add_argument("--max_frames", type=int, default=30)
    ap.add_argument("--bpe", default=None)
    args = ap.parse_args()

    # 1) 정수명 심링크 frames 디렉토리 + index→stem 매핑
    src = sorted(glob.glob(os.path.join(args.frames, f"*{args.img_ext}")))[:args.max_frames]
    if not src:
        print("[ERROR] no frames"); return
    tmp = tempfile.mkdtemp(prefix="sam3vid_")
    idx2stem = []
    for i, f in enumerate(src):
        os.symlink(os.path.abspath(f), os.path.join(tmp, f"{i}.jpg"))
        idx2stem.append(os.path.splitext(os.path.basename(f))[0])
    print(f"frames={len(src)} symlinked → {tmp}  (0.jpg..{len(src)-1}.jpg)")
    print(f"idx0={idx2stem[0]} idx1={idx2stem[1]}")

    from sam3.model_builder import build_sam3_video_predictor
    try:
        predictor = build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()),
                                               bpe_path=args.bpe) if args.bpe else \
                    build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    except TypeError:
        predictor = build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))

    def propagate(sid):
        out = {}
        for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
            out[r["frame_index"]] = r["outputs"]
        return out

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sid = predictor.handle_request(dict(type="start_session", resource_path=tmp))["session_id"]
        predictor.handle_request(dict(type="reset_session", session_id=sid))
        resp = predictor.handle_request(dict(type="add_prompt", session_id=sid,
                                             frame_index=0, text=args.concept))
        print("\n=== add_prompt response keys ===", list(resp.keys()))
        print("=== outputs (frame0) 구조 ===")
        describe(resp["outputs"])

        opf = propagate(sid)
        print(f"\n=== propagate: {len(opf)} frames. frame0 outputs 구조 ===")
        describe(opf[0])

        try:
            from sam3.visualization_utils import prepare_masks_for_visualization
            vis = prepare_masks_for_visualization({0: opf[0]})
            print("\n=== prepare_masks_for_visualization({0:...}) 구조 ===")
            describe(vis)
        except Exception as e:
            print("prepare_masks_for_visualization 불가:", e)

        predictor.handle_request(dict(type="close_session", session_id=sid))
    try: predictor.shutdown()
    except Exception: pass
    print("\n위 구조에서 'obj_id ↔ per-frame mask' 추출 경로를 확인하면 video re-labeling 작성 가능.")


if __name__ == "__main__":
    main()
