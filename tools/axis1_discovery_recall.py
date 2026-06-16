#!/usr/bin/env python3
"""
축1 — instance-dense scene에서 SAM3 discovery recall (scene-agnostic, inference-only).

stage3 sam3_discovery_recall.py를 일반화: scene/GT/vocab을 인자로. 학습 없음.
GT = per-frame instance-id PNG (semantic_instance류). 구조물 클래스는 분모에서 제외.

비교 목적: instance-dense scene(Waldo Kitchen / ScanNet 0000_00 등)에서
SAM3(auto-vocab 또는 지정 vocab) recall이 SAM2+graph baseline을 넘는지 — 문제가
실재하는 곳에서의 입증. (Replica room0처럼 깨끗한 곳이 아니라.)

규약: stage3(autocast bf16, mask squeeze).

실행 (sam3 env):
    conda activate sam3
    LD_LIBRARY_PATH= python axis1_discovery_recall.py \
        --frames <images_dir> --img_ext .jpg \
        --gt_dir <gt_instance_png_dir> --gt_fmt "semantic_instance_{idx}.png" \
        --vocab_json /home/elicer/sam3/vocab.json \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 1 --iou 0.1
  vocab 옵션: --vocab_json(파일) | --vocab "a,b,c" | --info_json(클래스명 자동)
"""
import argparse, glob, json, os, re
import numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

STRUCT_DEFAULT = {"wall","floor","ceiling","undefined","vent","switch","wall-plug","pillar",
                  "window","door","ceiling-light","blinds","curtain"}


def masks_np(out, hw):
    m = out.get("masks") if isinstance(out, dict) else None
    if m is None: return []
    if hasattr(m, "detach"): m = m.detach().float().cpu().numpy()
    m = np.squeeze(np.asarray(m))
    if m.ndim == 2: m = m[None]
    res = []
    for mm in m:
        b = mm > 0.5
        if b.shape != hw:
            b = np.array(Image.fromarray((b.astype(np.uint8)*255)).resize((hw[1],hw[0]),Image.NEAREST))>127
        res.append(b)
    return res


def load_id2cls(info_json):
    info = json.load(open(info_json))
    if "classes" in info and "objects" in info:
        classes = {c["id"]: c["name"] for c in info["classes"]}
        return {o["id"]: classes.get(o["class_id"], "?") for o in info["objects"]}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--img_ext", default=".jpg")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--gt_fmt", default="semantic_instance_{idx}.png")
    ap.add_argument("--info_json", default=None, help="id→class (구조물 제외용)")
    ap.add_argument("--vocab_json", default=None)
    ap.add_argument("--vocab", default=None, help="쉼표구분")
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--iou", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ignore", default="0", help="GT에서 무시할 id(쉼표)")
    args = ap.parse_args()

    id2cls = load_id2cls(args.info_json) if args.info_json else {}
    if args.vocab_json:
        VOCAB = json.load(open(args.vocab_json))["vocab"]
    elif args.vocab:
        VOCAB = [v.strip() for v in args.vocab.split(",")]
    elif id2cls:
        VOCAB = sorted({c for c in id2cls.values() if c not in STRUCT_DEFAULT})
    else:
        raise SystemExit("vocab 필요: --vocab_json | --vocab | --info_json")
    print(f"vocab={len(VOCAB)}: {VOCAB[:20]}{'...' if len(VOCAB)>20 else ''}")
    ignore = {int(x) for x in args.ignore.split(",")}

    model = build_sam3_image_model(**({"bpe_path": args.bpe} if args.bpe else {}))
    proc = Sam3Processor(model)

    frames = sorted(glob.glob(os.path.join(args.frames, f"*{args.img_ext}")))[::args.stride]
    print(f"frames={len(frames)}")
    gt_seen, gt_hit = set(), set()
    from collections import Counter
    for fp in frames:
        stem = os.path.splitext(os.path.basename(fp))[0]
        idx = int(re.sub(r"\D", "", stem))
        gtp = os.path.join(args.gt_dir, args.gt_fmt.format(idx=idx))
        if not os.path.exists(gtp): continue
        gt = np.array(Image.open(gtp)).astype(np.int64)
        ids = [i for i in np.unique(gt) if i not in ignore
               and id2cls.get(int(i), "?") not in STRUCT_DEFAULT]
        for i in ids: gt_seen.add(int(i))
        img = Image.open(fp).convert("RGB")
        dets = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = proc.set_image(img)
            for c in VOCAB:
                dets += masks_np(proc.set_text_prompt(state=st, prompt=c), gt.shape)
        for i in ids:
            gi = (gt == i)
            for mm in dets:
                inter = np.logical_and(mm, gi).sum(); union = np.logical_or(mm, gi).sum()
                if union > 0 and inter/union >= args.iou:
                    gt_hit.add(int(i)); break

    rec = len(gt_hit)/max(1, len(gt_seen))
    print(f"\nobject discovery recall(@IoU{args.iou}, stride{args.stride}): "
          f"{len(gt_hit)}/{len(gt_seen)} = {rec:.3f}")
    if id2cls:
        missed = sorted(gt_seen - gt_hit)
        print("missed:", dict(Counter(id2cls.get(i, "?") for i in missed)))
    print("\nbaseline(Split&Splat/SAM2+graph)와 같은 EXCLUDE·IoU·stride로 비교해야 공정.")


if __name__ == "__main__":
    main()
