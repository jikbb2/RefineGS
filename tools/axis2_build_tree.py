#!/usr/bin/env python3
"""
축2 Step 1 — 단일 뷰·단일 객체 → part 후보 (whole→parts, 1단계).

객체 마스크 안에 point grid → SAM3 multimask 후보 수집 → 객체 대비 size-filter
(너무 작은 노이즈·너무 큰 ≈whole 제거) → NMS dedup → 상위 K part.
deep tree 대신 part-whole 결정에 필요한 whole→parts 평면 구조만 출력.

규약: stage3(autocast bf16, mask squeeze), predict_inst(multimask_output=True).
객체 마스크 출처: 기존 per-view instance PNG (relabel 출력) 또는 --concept.

실행 (sam3 env):
    conda activate sam3
    LD_LIBRARY_PATH= python axis2_build_tree.py \
        --image .../images/frameXXXX.JPEG \
        --obj_mask .../masks/98/masks/frameXXXX.png \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --grid 6 --min_part 0.05 --max_part 0.7 --topk 12 --out_dir ~/axis2_tree_98
"""
import argparse
import json
import os
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_bool_masks(m):
    if hasattr(m, "detach"):
        m = m.detach().float().cpu().numpy()
    m = np.squeeze(np.asarray(m))
    if m.ndim == 2:
        m = m[None]
    return [mm > 0.5 if mm.dtype != bool else mm for mm in m]


def load_obj_mask(path, target_hw):
    m = np.asarray(Image.open(path).convert("L")) > 0
    if m.shape != target_hw:
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                     .resize((target_hw[1], target_hw[0]), Image.NEAREST)) > 127
    return m


def concept_obj_mask(proc, state, concept):
    out = proc.set_text_prompt(state=state, prompt=concept)
    m = out.get("masks") if isinstance(out, dict) else None
    if m is None:
        return None
    bm = to_bool_masks(m)
    return bm[int(np.argmax([x.sum() for x in bm]))] if bm else None


def grid_points_in_mask(mask, grid):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), int)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pts = []
    for gy in np.linspace(y0, y1, grid):
        for gx in np.linspace(x0, x1, grid):
            xi, yi = int(round(gx)), int(round(gy))
            if mask[yi, xi]:
                pts.append((xi, yi))
    return np.array(pts, int)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def coverage(small, big):
    s = small.sum()
    return np.logical_and(small, big).sum() / s if s else 0.0


def nms(cands, iou_th=0.7):
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["area"])
    kept = []
    for i in order:
        if all(iou(cands[i]["mask"], cands[k]["mask"]) < iou_th for k in kept):
            kept.append(i)
    return [cands[i] for i in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--obj_mask", default=None)
    ap.add_argument("--concept", default=None)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--keep_cov", type=float, default=0.8,
                    help="후보가 객체 안에 이만큼 들어와야 part로 채택")
    ap.add_argument("--min_part", type=float, default=0.05,
                    help="객체 대비 최소 면적(노이즈 제거)")
    ap.add_argument("--max_part", type=float, default=0.7,
                    help="객체 대비 최대 면적(≈whole 제거)")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--out_dir", default=os.path.expanduser("~/axis2_tree_out"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mk = dict(enable_inst_interactivity=True)
    if args.bpe:
        mk["bpe_path"] = args.bpe
    model = build_sam3_image_model(**mk)
    proc = Sam3Processor(model)
    image = Image.open(args.image).convert("RGB")
    W, H = image.size

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(image)
        if args.obj_mask:
            obj = load_obj_mask(args.obj_mask, (H, W))
        elif args.concept:
            obj = concept_obj_mask(proc, state, args.concept)
            proc.reset_all_prompts(state); state = proc.set_image(image)
        else:
            obj = np.ones((H, W), bool)
        if obj is None or obj.sum() == 0:
            print("[ERROR] 객체 마스크 비어있음"); return
        obj_area = int(obj.sum())
        obj_frac = obj_area / (H * W)
        print(f"객체 마스크 area={obj_frac*100:.2f}% ({obj_area}px)")
        if obj_frac > 0.9:
            print("  [WARN] 객체가 프레임의 90%+ — 잘못된(전체) 마스크일 가능성. "
                  "다른 프레임/마스크 확인 권장.")

        seeds = grid_points_in_mask(obj, args.grid)
        print(f"seed points: {len(seeds)}")
        cands = []
        for pt in seeds:
            masks, scores, _ = model.predict_inst(
                state, point_coords=np.array([pt]),
                point_labels=np.array([1]), multimask_output=True)
            for mm, s in zip(to_bool_masks(masks), np.asarray(scores).reshape(-1)):
                a = int(mm.sum())
                if a == 0:
                    continue
                fo = a / obj_area              # 객체 대비 면적
                if coverage(mm, obj) >= args.keep_cov and args.min_part <= fo <= args.max_part:
                    cands.append({"mask": mm, "score": float(s), "area": a, "frac_obj": fo})

    print(f"size-filter 통과 후보: {len(cands)}")
    parts = nms(cands, iou_th=0.7)
    parts = sorted(parts, key=lambda p: -p["area"])[:args.topk]
    print(f"NMS+topk 후 part: {len(parts)}\n")

    print(f"whole: obj ({obj_frac*100:.1f}% of frame)")
    for i, p in enumerate(parts):
        print(f"  part{i}: {p['frac_obj']*100:5.1f}% of obj  score={p['score']:.2f}")

    masks_arr = np.stack([p["mask"] for p in parts]) if parts else np.empty((0, H, W), bool)
    np.savez_compressed(os.path.join(args.out_dir, "parts.npz"), obj=obj, parts=masks_arr)
    json.dump([{"id": i, "frac_obj": round(p["frac_obj"], 3), "score": round(p["score"], 3)}
               for i, p in enumerate(parts)],
              open(os.path.join(args.out_dir, "parts.json"), "w"), indent=2)
    Image.fromarray((obj * 255).astype(np.uint8)).save(os.path.join(args.out_dir, "whole.png"))
    for i, p in enumerate(parts):
        Image.fromarray((p["mask"] * 255).astype(np.uint8)).save(
            os.path.join(args.out_dir, f"part{i}.png"))
    print(f"\n저장: {args.out_dir} (whole.png, part*.png, parts.json/npz)")
    print("판정: part 수가 적당(~3-10)하고 part*.png가 의미있는 부품이면 Step 1 OK → Step 2(multi-view voting).")


if __name__ == "__main__":
    main()
