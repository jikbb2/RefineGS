#!/usr/bin/env python3
"""
축2 Step 1 — 단일 뷰·단일 객체 → part tree 빌더 (atomic unit).

객체 마스크 안에 point grid를 뿌려 SAM3 multimask(subpart/part/whole) 후보를 모으고,
객체 밖으로 새는 마스크 제거 → NMS dedup → 포함관계 tree 구성.
이 단위를 Step 2에서 뷰·객체로 반복하며 3D voting으로 합친다.

객체 마스크 출처: 기존 per-view instance PNG (relabel 출력; 파이프라인 결합) 또는 --concept.

규약: stage3 (autocast bf16, mask squeeze), predict_inst(multimask_output=True).

실행 (sam3 env):
    conda activate sam3
    LD_LIBRARY_PATH= python axis2_build_tree.py \
        --image /home/elicer/RefineGS/data/replica_room0/images/frame000000.JPEG \
        --obj_mask /home/elicer/RefineGS/data/replica_room0/masks/97/masks/frame000000.png \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --grid 8 --out_dir ~/axis2_tree_out
    # 또는 객체를 concept로:
    #   --concept couch   (--obj_mask 대신)
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
    if not bm:
        return None
    return bm[int(np.argmax([x.sum() for x in bm]))]


def grid_points_in_mask(mask, grid):
    """grid x grid 균일 후보 중 마스크 내부 점만 (x,y)."""
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
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def coverage(small, big):
    s = small.sum()
    return np.logical_and(small, big).sum() / s if s else 0.0


def nms_dedup(cands, iou_th=0.85):
    """cands: list of dict(mask,score,area). IoU>th 중복 병합(면적 큰 것 유지)."""
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["area"])
    kept = []
    for i in order:
        if all(iou(cands[i]["mask"], cands[j]["mask"]) < iou_th for j in kept):
            kept.append(i)
    return [cands[i] for i in kept]


def build_tree(nodes):
    """parent = 자신을 가장 작게 포함(coverage>0.9)하는 더 큰 노드."""
    order = sorted(range(len(nodes)), key=lambda i: nodes[i]["area"])  # 작은→큰
    for a in order:
        parent, parea = -1, None
        for b in range(len(nodes)):
            if b == a or nodes[b]["area"] <= nodes[a]["area"]:
                continue
            if coverage(nodes[a]["mask"], nodes[b]["mask"]) > 0.9:
                if parea is None or nodes[b]["area"] < parea:
                    parent, parea = b, nodes[b]["area"]
        nodes[a]["parent"] = parent
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--obj_mask", default=None, help="객체 instance mask PNG")
    ap.add_argument("--concept", default=None, help="객체를 concept로 (obj_mask 대신)")
    ap.add_argument("--grid", type=int, default=8, help="객체 bbox 내 grid 해상도")
    ap.add_argument("--keep_cov", type=float, default=0.7,
                    help="후보 마스크가 객체 안에 이만큼 들어와야 채택(밖으로 새는 것 제거)")
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
        print(f"객체 마스크 area={obj.sum()/(H*W)*100:.2f}%")

        seeds = grid_points_in_mask(obj, args.grid)
        print(f"seed points: {len(seeds)}")

        cands = []
        for pt in seeds:
            masks, scores, _ = model.predict_inst(
                state, point_coords=np.array([pt]),
                point_labels=np.array([1]), multimask_output=True)
            bm = to_bool_masks(masks)
            sc = np.asarray(scores).reshape(-1).tolist()
            for mm, s in zip(bm, sc):
                if coverage(mm, obj) >= args.keep_cov and mm.sum() > 0:
                    cands.append({"mask": mm, "score": float(s), "area": int(mm.sum())})

    print(f"후보 마스크(객체 내부): {len(cands)}")
    nodes = nms_dedup(cands, iou_th=0.85)
    nodes = build_tree(nodes)
    print(f"dedup 후 노드: {len(nodes)}")

    # 저장 + 트리 요약
    masks_arr = np.stack([n["mask"] for n in nodes]) if nodes else np.empty((0, H, W), bool)
    np.savez_compressed(os.path.join(args.out_dir, "tree_masks.npz"),
                        obj=obj, masks=masks_arr)
    meta = [{"id": i, "area_frac": round(n["area"] / (H * W), 4),
             "score": round(n["score"], 3), "parent": n["parent"]}
            for i, n in enumerate(nodes)]
    json.dump(meta, open(os.path.join(args.out_dir, "tree.json"), "w"), indent=2)

    roots = [m for m in meta if m["parent"] == -1]
    print(f"\n트리: roots={len(roots)}  (parent=-1)")
    for m in sorted(meta, key=lambda x: -x["area_frac"]):
        depth = 0; p = m["parent"]
        while p != -1:
            depth += 1; p = meta[p]["parent"]
        print(f"  {'  '*depth}node{m['id']}: area={m['area_frac']*100:5.2f}% "
              f"score={m['score']:.2f} parent={m['parent']}")
    for i, n in enumerate(nodes):
        Image.fromarray((n["mask"] * 255).astype(np.uint8)).save(
            os.path.join(args.out_dir, f"node{i}.png"))
    print(f"\n저장: {args.out_dir} (tree.json, tree_masks.npz, node*.png)")
    print("판정: 객체가 의미있는 part로 분해되고(노드 수 적당), 트리 depth가 객체→part로 잡히면 Step 1 OK.")


if __name__ == "__main__":
    main()
