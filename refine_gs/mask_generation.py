#!/usr/bin/env python3
"""
Step 1: Mask Generation using SAM2 (or SAM1 fallback).
Generates per-view instance masks with coarse-to-fine merging.

Usage:
    python -m refine_gs.mask_generation \
        -s data/hotdog \
        --output data/hotdog/masks \
        --sam_checkpoint /path/to/sam2_checkpoint
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def try_load_sam2(checkpoint_path):
    """Try loading SAM2. Returns (generator, version) or (None, None)."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # SAM2 model config - adjust based on checkpoint type
        if "large" in checkpoint_path.lower():
            model_cfg = "sam2_hiera_l.yaml"
        elif "base" in checkpoint_path.lower() or "b+" in checkpoint_path.lower():
            model_cfg = "sam2_hiera_b+.yaml"
        else:
            model_cfg = "sam2_hiera_s.yaml"

        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        print(f"[OK] SAM2 loaded from {checkpoint_path}")
        return generator, "sam2"
    except Exception as e:
        print(f"[WARN] SAM2 load failed: {e}")
        return None, None


def try_load_sam1(checkpoint_path):
    """Fallback: load SAM1."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam = sam.to(device)
        generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        print(f"[OK] SAM1 loaded from {checkpoint_path}")
        return generator, "sam1"
    except Exception as e:
        print(f"[ERROR] SAM1 load failed: {e}")
        return None, None


def merge_coarse_to_fine(masks, iou_threshold=0.8):
    """
    Coarse-to-fine mask merging (Section 3.1.1).
    Sort by area (largest first), merge smaller masks that overlap heavily.
    """
    if len(masks) == 0:
        return []

    # Sort by area descending
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    merged = []
    used = set()

    for i, mask_i in enumerate(masks):
        if i in used:
            continue

        current_seg = mask_i["segmentation"].copy()
        current_area = mask_i["area"]

        # Check smaller masks for overlap
        for j in range(i + 1, len(masks)):
            if j in used:
                continue
            mask_j = masks[j]
            intersection = np.logical_and(current_seg, mask_j["segmentation"]).sum()
            # If smaller mask is mostly inside current, absorb it
            if mask_j["area"] > 0 and intersection / mask_j["area"] > iou_threshold:
                current_seg = np.logical_or(current_seg, mask_j["segmentation"])
                used.add(j)

        merged.append({
            "segmentation": current_seg,
            "area": int(current_seg.sum()),
        })
        used.add(i)

    return merged


def generate_masks_for_dataset(source_path, output_dir, sam_checkpoint,
                                sam2_checkpoint=None):
    """
    Generate per-view instance masks for all training images.

    Args:
        source_path: dataset root (contains transforms_train.json)
        output_dir: where to save masks
        sam_checkpoint: path to SAM1 checkpoint (fallback)
        sam2_checkpoint: path to SAM2 checkpoint (preferred)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load mask generator
    generator, version = None, None
    if sam2_checkpoint and os.path.exists(sam2_checkpoint):
        generator, version = try_load_sam2(sam2_checkpoint)
    if generator is None and os.path.exists(sam_checkpoint):
        generator, version = try_load_sam1(sam_checkpoint)
    if generator is None:
        print("[ERROR] No SAM model available. Provide --sam_checkpoint or --sam2_checkpoint")
        sys.exit(1)

    # Load image list from transforms
    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    image_paths = []
    for frame in meta["frames"]:
        fp = frame["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(source_path, fp)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            if os.path.exists(fp + ext):
                image_paths.append(fp + ext)
                break

    print(f"Processing {len(image_paths)} images with {version}...")

    all_masks_meta = {}

    for k, img_path in enumerate(tqdm(image_paths, desc="Generating masks")):
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_rgb.shape[:2]

        # Generate masks
        raw_masks = generator.generate(image_rgb)

        # Coarse-to-fine merge
        merged = merge_coarse_to_fine(raw_masks)

        # Save individual masks as PNG
        view_dir = os.path.join(output_dir, f"view_{k:04d}")
        os.makedirs(view_dir, exist_ok=True)

        mask_info = []
        for j, m in enumerate(merged):
            seg = m["segmentation"].astype(np.uint8) * 255
            mask_path = os.path.join(view_dir, f"mask_{j:03d}.png")
            cv2.imwrite(mask_path, seg)
            mask_info.append({
                "mask_id": j,
                "area": m["area"],
                "mask_file": f"mask_{j:03d}.png",
            })

        # Save visualization
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        palette = np.random.RandomState(42).randint(50, 255, (len(merged), 3))
        for j, m in enumerate(merged):
            vis[m["segmentation"]] = palette[j]
        cv2.imwrite(os.path.join(view_dir, "vis_masks.png"), vis)

        all_masks_meta[k] = {
            "image_path": img_path,
            "num_masks": len(merged),
            "height": H,
            "width": W,
            "masks": mask_info,
        }

    # Save metadata
    meta_path = os.path.join(output_dir, "masks_meta.json")
    with open(meta_path, "w") as f:
        json.dump(all_masks_meta, f, indent=2)

    print(f"\nDone! Masks saved to {output_dir}")
    print(f"  Views: {len(all_masks_meta)}")
    print(f"  Avg masks/view: {np.mean([v['num_masks'] for v in all_masks_meta.values()]):.1f}")

    return all_masks_meta


def load_masks(mask_dir):
    """Load previously generated masks from disk."""
    meta_path = os.path.join(mask_dir, "masks_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    all_masks = {}
    for k_str, info in meta.items():
        k = int(k_str)
        view_dir = os.path.join(mask_dir, f"view_{k:04d}")
        masks = []
        for m_info in info["masks"]:
            seg = cv2.imread(os.path.join(view_dir, m_info["mask_file"]),
                             cv2.IMREAD_GRAYSCALE)
            masks.append({
                "segmentation": seg > 127,
                "area": m_info["area"],
                "mask_id": m_info["mask_id"],
            })
        all_masks[k] = masks

    return all_masks, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAM masks for dataset")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--output", default=None,
                        help="Output directory (default: source_path/masks)")
    parser.add_argument("--sam_checkpoint", default="/home/elicer/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2_checkpoint", default=None)
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.source_path, "masks")

    generate_masks_for_dataset(
        args.source_path,
        output_dir,
        args.sam_checkpoint,
        args.sam2_checkpoint,
    )
