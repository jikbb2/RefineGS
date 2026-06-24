#!/usr/bin/env python3
"""
객체 실루엣 마스크 dumper — SAM3 concept로 per-view 객체 마스크 PNG 저장.
(degenerate한 파이프라인 마스크 회피; visual_hull_recover.py 입력으로 사용)

규약: stage3(autocast bf16, mask squeeze). sam3 env.

실행:
    conda activate sam3
    LD_LIBRARY_PATH= python dump_concept_masks.py \
        --images_dir /home/elicer/RefineGS/data/replica_room0/masks/98/images \
        --concept table \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 1 --out_dir ~/obj_masks_98
"""
import argparse, glob, os, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_bool(m):
    if hasattr(m, "detach"):
        m = m.detach().float().cpu().numpy()
    m = np.squeeze(np.asarray(m))
    if m.ndim == 2:
        m = m[None]
    return [x > 0.5 if x.dtype != bool else x for x in m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mk = dict(enable_inst_interactivity=True)
    if args.bpe:
        mk["bpe_path"] = args.bpe
    model = build_sam3_image_model(**mk)
    proc = Sam3Processor(model)

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*")))[::args.stride]
    n_ok = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ip in imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            image = Image.open(ip).convert("RGB")
            st = proc.set_image(image)
            out = proc.set_text_prompt(state=st, prompt=args.concept)
            bm = to_bool(out.get("masks")) if isinstance(out, dict) else []
            if not bm:
                m = np.zeros((image.size[1], image.size[0]), bool)
            else:
                m = bm[int(np.argmax([x.sum() for x in bm]))]   # 최대 instance
                n_ok += 1
            Image.fromarray((m * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"{stem}.png"))
    print(f"saved {len(imgs)} masks ({n_ok} non-empty) -> {args.out_dir}")


if __name__ == "__main__":
    main()
