#!/usr/bin/env python3
"""
축2 가치 실증 — granularity 일관성 평가 (raw SAM vs 축2 일관 마스크, GT instance 기준).

문제정의 4.2: SAM이 뷰마다 granularity가 흔들려(whole↔parts) GT 매칭이 깨짐.
공정 비교(granularity 효과만 분리): 각 뷰에서 SAM3가 주는 마스크들 중 GT instance와
가장 잘 맞는 것(oracle instance 선택)의 IoU를 raw로 삼아, 그 *뷰 간 분산*을 측정.
축2 일관 마스크(obj_support 역투영, ~/axis2_whole_98/part0/)의 IoU 분산과 비교.
  raw: 분산 큼(granularity 흔들림) / 축2: 분산 작음(일관) 이면 축2 가치 실증.

규약: stage3(autocast bf16, mask squeeze). sam3 env에서 실행.

실행:
    conda activate sam3
    LD_LIBRARY_PATH= python axis2_consistency_eval.py \
        --gt_dir /home/elicer/room_0/imap/00/semantic_instance \
        --gt_id 7 --concept table \
        --images_dir /home/elicer/RefineGS/data/replica_room0/masks/98/images \
        --axis2_dir ~/axis2_whole_98/part0 \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 10
"""
import argparse
import glob
import os
import re
import numpy as np
import torch
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


def resize_to(mask, hw):
    if mask.shape == hw:
        return mask
    return np.array(Image.fromarray((mask*255).astype(np.uint8))
                    .resize((hw[1], hw[0]), Image.NEAREST)) > 127


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="semantic_instance PNG 폴더")
    ap.add_argument("--gt_id", type=int, required=True)
    ap.add_argument("--concept", default="table")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--axis2_dir", required=True, help="축2 일관 마스크 폴더(part0)")
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    mk = dict(enable_inst_interactivity=True)
    if args.bpe:
        mk["bpe_path"] = args.bpe
    model = build_sam3_image_model(**mk)
    proc = Sam3Processor(model)

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*")))[::args.stride]
    raw_ious, ax_ious = [], []
    print(f"{'frame':>12} {'raw(SAM best)':>14} {'axis2':>10}  (IoU to GT inst)")
    print("-" * 50)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ip in imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            idx = int(re.sub(r"\D", "", stem))
            gtp = os.path.join(args.gt_dir, f"semantic_instance_{idx}.png")
            if not os.path.exists(gtp):
                continue
            gt = (np.array(Image.open(gtp)).astype(np.int64) == args.gt_id)
            if gt.sum() == 0:
                continue                      # GT 인스턴스가 이 뷰에 없음
            image = Image.open(ip).convert("RGB")
            state = proc.set_image(image)
            out = proc.set_text_prompt(state=state, prompt=args.concept)
            sam_masks = to_bool(out.get("masks")) if isinstance(out, dict) else []
            # raw: GT와 가장 잘 맞는 SAM 마스크(oracle instance 선택) → granularity만 평가
            raw_best = max((iou(resize_to(m, gt.shape), gt) for m in sam_masks), default=0.0)
            # axis2 일관 마스크
            axp = os.path.join(args.axis2_dir, f"{stem}.png")
            if os.path.exists(axp):
                ax = np.array(Image.open(axp).convert("L")) > 0
                ax_iou = iou(resize_to(ax, gt.shape), gt)
            else:
                ax_iou = float("nan")
            raw_ious.append(raw_best); ax_ious.append(ax_iou)
            print(f"{stem:>12} {raw_best:>14.3f} {ax_iou:>10.3f}")

    raw = np.array(raw_ious); ax = np.array([a for a in ax_ious if not np.isnan(a)])
    print("-" * 50)
    print(f"raw  SAM(best): mean={raw.mean():.3f}  std={raw.std():.3f}  "
          f"min={raw.min():.3f}  n={len(raw)}")
    print(f"axis2 일관    : mean={ax.mean():.3f}  std={ax.std():.3f}  "
          f"min={ax.min():.3f}  n={len(ax)}")
    print("\n해석: axis2의 mean↑ 또는 std↓(특히 min↑)이면 granularity 일관성 개선 = 축2 가치.")
    print("raw의 std가 크면 SAM이 뷰마다 whole↔part로 흔들려 매칭이 불안정하다는 증거.")


if __name__ == "__main__":
    main()
