#!/usr/bin/env python3
"""
축2 Step 1 probe — SAM3 point multimask가 part 계층(whole>part>subpart)을 주는지 검증.

stage3 규약(autocast bf16, mask squeeze)을 그대로 사용.
SAM1-task interface: build_sam3_image_model(enable_inst_interactivity=True)
                     model.predict_inst(state, point_coords, point_labels, multimask_output=True)

동작:
  1) (옵션) concept 프롬프트로 객체 마스크를 잡아 그 내부에서 seed point 샘플
     (없으면 --points 로 직접, 또는 이미지 중앙 그리드).
  2) 각 seed point에서 predict_inst(multimask_output=True) → 3단계 마스크.
  3) 면적 오름차순 정렬 후, 작은→큰 마스크의 nesting coverage
     (area(small ∩ large)/area(small)) 출력 → ~1.0이면 계층 중첩 확인.
  4) 마스크 PNG 저장(눈으로 확인용).

실행 (sam3 env):
    conda activate sam3 && unset LD_LIBRARY_PATH
    python axis2_probe.py \
        --image /home/elicer/RefineGS/data/replica_room0/images/frame000000.JPEG \
        --concept couch --n_points 3 --out_dir ~/axis2_probe_out
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_bool_masks(m):
    """Normalize predict_inst masks to list of (H,W) bool arrays."""
    if hasattr(m, "detach"):
        m = m.detach().float().cpu().numpy()
    m = np.asarray(m)
    m = np.squeeze(m)
    if m.ndim == 2:
        m = m[None]
    return [(mm > 0.5) if mm.dtype != bool else mm for mm in m]


def concept_seed_points(model, proc, state, concept, n_points, rng):
    """Use a concept mask to sample interior seed points. Returns Nx2 (x,y) or None."""
    try:
        out = proc.set_text_prompt(state=state, prompt=concept)
    except Exception as e:
        print(f"  [concept seeding 실패: {e}] -> --points/grid 사용")
        return None
    m = out.get("masks") if isinstance(out, dict) else None
    if m is None:
        return None
    if hasattr(m, "detach"):
        m = m.detach().float().cpu().numpy()
    m = np.squeeze(np.asarray(m))
    if m.ndim == 2:
        m = m[None]
    # largest concept instance
    areas = [(mm > 0.5).sum() for mm in m]
    if not areas or max(areas) == 0:
        return None
    best = (m[int(np.argmax(areas))] > 0.5)
    ys, xs = np.where(best)
    if len(xs) == 0:
        return None
    sel = rng.choice(len(xs), min(n_points, len(xs)), replace=False)
    return np.stack([xs[sel], ys[sel]], axis=1)


def nesting_report(masks, scores):
    """masks: list of (H,W) bool. Print area frac + nesting coverage (small in large)."""
    order = np.argsort([m.sum() for m in masks])  # ascending area
    masks = [masks[i] for i in order]
    scores = [scores[i] for i in order]
    H, W = masks[0].shape
    print(f"    levels={len(masks)}  (면적 오름차순)")
    for i, (m, s) in enumerate(zip(masks, scores)):
        frac = m.sum() / (H * W)
        line = f"      L{i}: area={frac*100:5.2f}% score={s:.3f}"
        if i > 0:
            small = masks[i - 1]
            cov = np.logical_and(small, m).sum() / max(small.sum(), 1)
            line += f"  nest(L{i-1}⊂L{i})={cov:.2f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--concept", default=None, help="seed points from this concept mask")
    ap.add_argument("--points", default=None,
                    help='manual seeds "x1,y1 x2,y2" (concept 미사용 시)')
    ap.add_argument("--n_points", type=int, default=3)
    ap.add_argument("--bpe", default=None, help="bpe_simple_vocab_16e6.txt.gz 경로(자동탐지)")
    ap.add_argument("--out_dir", default=os.path.expanduser("~/axis2_probe_out"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    import sam3
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe = args.bpe or f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    model = build_sam3_image_model(bpe_path=bpe, enable_inst_interactivity=True)
    proc = Sam3Processor(model)
    image = Image.open(args.image).convert("RGB")
    W, H = image.size

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(image)

        # --- seed points ---
        seeds = None
        if args.concept:
            seeds = concept_seed_points(model, proc, state, args.concept,
                                        args.n_points, rng)
            if seeds is not None:
                print(f"concept '{args.concept}'에서 seed {len(seeds)}개 샘플")
                proc.reset_all_prompts(state)   # concept 지우고 inst로
                state = proc.set_image(image)
        if seeds is None and args.points:
            seeds = np.array([[int(c) for c in p.split(",")]
                              for p in args.points.split()])
        if seeds is None:
            seeds = np.array([[W // 2, H // 2], [W // 3, H // 2], [2 * W // 3, H // 2]])
            print("seed 미지정 → 이미지 중앙 그리드 사용")

        # --- per-seed multimask ---
        for k, pt in enumerate(seeds):
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=np.array([pt]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            bmasks = to_bool_masks(masks)
            sc = np.asarray(scores).reshape(-1).tolist()
            print(f"\nseed {k} @ ({pt[0]},{pt[1]}):")
            nesting_report(bmasks, sc)
            for li, mm in enumerate(bmasks):
                Image.fromarray((mm * 255).astype(np.uint8)).save(
                    os.path.join(args.out_dir, f"seed{k}_L{li}.png"))

    print(f"\n마스크 PNG 저장: {args.out_dir}")
    print("판정: levels=3 이고 nest≈0.9+ 이면 whole⊃part⊃subpart 계층 확인 → Step 1 진행.")


if __name__ == "__main__":
    main()
