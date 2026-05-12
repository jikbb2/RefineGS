#!/usr/bin/env python3
"""
Step 4: Per-Object 2DGS Reconstruction.

Trains independent 2DGS models per object instance using mask-weighted losses.
This wraps the standard 2DGS training loop, applying per-pixel mask weighting
so each object's Gaussians focus only on their assigned region.

Split&Splat approach (Section 3.2):
  - Independent reconstruction per object
  - Mask-weighted L1 + SSIM loss
  - Background gets its own model
  - After training: per-object meshes can be extracted via TSDF

Usage:
    python -m refine_gs.per_object_reconstruction \
        -s data/hotdog \
        --propagated data/hotdog/refine_gs_data/propagated \
        --output output/hotdog_per_object \
        --iterations 7000
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─── Mask-weighted loss functions ────────────────────────────────

def masked_l1_loss(rendered, gt, mask_weight):
    """
    L1 loss weighted by per-pixel instance mask.

    Args:
        rendered: [3, H, W] rendered image
        gt: [3, H, W] ground truth image
        mask_weight: [1, H, W] or [H, W] soft mask in [0, 1]
    """
    if mask_weight.dim() == 2:
        mask_weight = mask_weight.unsqueeze(0)  # [1, H, W]

    diff = torch.abs(rendered - gt) * mask_weight
    # Normalize by mask area to avoid bias toward larger objects
    area = mask_weight.sum().clamp(min=1.0)
    return diff.sum() / (area * rendered.shape[0])


def masked_ssim_loss(rendered, gt, mask_weight, window_size=11):
    """
    SSIM loss computed only in the masked region.
    Uses the mask as a soft weight on the SSIM map.

    Args:
        rendered: [1, 3, H, W] or [3, H, W]
        gt: [1, 3, H, W] or [3, H, W]
        mask_weight: [H, W] or [1, 1, H, W]
    """
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
        gt = gt.unsqueeze(0)
    if mask_weight.dim() == 2:
        mask_weight = mask_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif mask_weight.dim() == 3:
        mask_weight = mask_weight.unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=rendered.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
    window = window.expand(3, 1, -1, -1)  # [3, 1, ws, ws]

    pad = window_size // 2

    mu1 = F.conv2d(rendered, window, padding=pad, groups=3)
    mu2 = F.conv2d(gt, window, padding=pad, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(rendered ** 2, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt ** 2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(rendered * gt, window, padding=pad, groups=3) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Apply mask weighting
    ssim_weighted = (1 - ssim_map) * mask_weight
    area = mask_weight.sum().clamp(min=1.0)
    return ssim_weighted.sum() / (area * 3)


def combined_masked_loss(rendered, gt, mask_weight, lambda_ssim=0.2):
    """Combined L1 + SSIM loss with mask weighting."""
    l1 = masked_l1_loss(rendered, gt, mask_weight)
    ssim = masked_ssim_loss(rendered, gt, mask_weight)
    return (1 - lambda_ssim) * l1 + lambda_ssim * ssim


# ─── Data preparation ────────────────────────────────────────────

def load_propagated_masks(propagated_dir):
    """
    Load propagation results: labeled point cloud + per-view refined masks.

    Returns:
        meta: propagation metadata
        instance_masks: {view_idx: {label: bool_mask}}
    """
    meta_path = os.path.join(propagated_dir, "propagation_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    masks_dir = os.path.join(propagated_dir, "refined_masks")
    instance_masks = {}

    for view_dir in sorted(Path(masks_dir).glob("view_*")):
        k = int(view_dir.name.split("_")[1])
        instance_masks[k] = {}
        for mask_file in sorted(view_dir.glob("instance_*.png")):
            label = int(mask_file.stem.split("_")[1])
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            instance_masks[k][label] = (mask > 127)

    return meta, instance_masks


def prepare_per_object_dataset(source_path, instance_masks, target_label):
    """
    Create a per-object training dataset.

    For each view, the GT image is the original image, and
    the mask_weight is the binary mask for `target_label`.

    Returns:
        list of dicts: [{image_path, mask_weight (np.array)}, ...]
    """
    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    dataset = []
    for k, frame in enumerate(meta["frames"]):
        fp = frame["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(source_path, fp)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            if os.path.exists(fp + ext):
                fp = fp + ext
                break

        if k in instance_masks and target_label in instance_masks[k]:
            mask = instance_masks[k][target_label].astype(np.float32)
        else:
            # If this view doesn't have the object, use zero mask
            # (will contribute zero loss)
            img = cv2.imread(fp)
            if img is not None:
                mask = np.zeros(img.shape[:2], dtype=np.float32)
            else:
                continue

        dataset.append({
            "image_path": fp,
            "mask_weight": mask,
            "transform_matrix": frame["transform_matrix"],
        })

    return dataset


def prepare_background_dataset(source_path, instance_masks):
    """
    Create background training dataset.
    Background mask = 1 - union(all instance masks).
    """
    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    dataset = []
    for k, frame in enumerate(meta["frames"]):
        fp = frame["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(source_path, fp)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            if os.path.exists(fp + ext):
                fp = fp + ext
                break

        if k in instance_masks:
            # Union of all instance masks
            union = None
            for label, mask in instance_masks[k].items():
                if union is None:
                    union = mask.copy()
                else:
                    union = np.logical_or(union, mask)

            if union is not None:
                bg_mask = (~union).astype(np.float32)
            else:
                img = cv2.imread(fp)
                bg_mask = np.ones(img.shape[:2], dtype=np.float32)
        else:
            img = cv2.imread(fp)
            if img is not None:
                bg_mask = np.ones(img.shape[:2], dtype=np.float32)
            else:
                continue

        dataset.append({
            "image_path": fp,
            "mask_weight": bg_mask,
            "transform_matrix": frame["transform_matrix"],
        })

    return dataset


# ─── Training stub (to be integrated with actual 2DGS code) ─────

def create_per_object_config(base_config, object_label, output_dir, iterations):
    """
    Create a per-object training config for 2DGS.

    This generates the configuration needed to train one instance.
    In practice, this will modify the 2DGS train.py arguments.
    """
    config = {
        "object_label": object_label,
        "output_dir": os.path.join(output_dir, f"object_{object_label:03d}"),
        "iterations": iterations,
        "lambda_ssim": 0.2,
        "densify_until": min(iterations // 2, 15000),
        "densify_from": 500,
        "densify_interval": 100,
        "opacity_reset_interval": 3000,
    }
    config.update(base_config)
    return config


def write_per_object_transforms(dataset, source_path, output_path):
    """
    Write a per-object transforms file that includes mask paths.

    This creates a modified transforms_train.json that the 2DGS
    training script can read, with additional mask_weight_path fields.
    """
    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        original = json.load(f)

    mask_dir = os.path.dirname(output_path)
    os.makedirs(mask_dir, exist_ok=True)

    # Save masks alongside the transforms file
    masks_subdir = os.path.join(mask_dir, "train_masks")
    os.makedirs(masks_subdir, exist_ok=True)

    new_frames = []
    for i, entry in enumerate(dataset):
        # Save mask as image
        mask_path = os.path.join(masks_subdir, f"mask_{i:04d}.png")
        cv2.imwrite(mask_path, (entry["mask_weight"] * 255).astype(np.uint8))

        frame = {
            "file_path": entry["image_path"],
            "transform_matrix": entry["transform_matrix"],
            "mask_weight_path": mask_path,
        }
        new_frames.append(frame)

    new_transforms = {
        "camera_angle_x": original["camera_angle_x"],
        "frames": new_frames,
    }

    with open(output_path, "w") as f:
        json.dump(new_transforms, f, indent=2)

    return output_path


def train_per_object_stub(config, dataset, source_path):
    """
    Stub for per-object 2DGS training.

    In Session B, this will be replaced with actual 2DGS training
    that uses mask-weighted losses. For now, it:
    1. Prepares the per-object transforms file
    2. Prints the command that would be run
    3. Saves config for later use

    The actual 2DGS integration requires modifying:
    - gaussian_renderer/__init__.py (add mask_weight to render output)
    - train.py (load masks, apply masked_l1_loss + masked_ssim_loss)
    """
    obj_dir = config["output_dir"]
    os.makedirs(obj_dir, exist_ok=True)

    # Write per-object transforms
    transforms_path = write_per_object_transforms(
        dataset, source_path,
        os.path.join(obj_dir, "transforms_train.json")
    )

    # Save config
    config_path = os.path.join(obj_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Object {config['object_label']:03d}:")
    print(f"    Output: {obj_dir}")
    print(f"    Views with mask: {sum(1 for d in dataset if d['mask_weight'].sum() > 0)}")
    print(f"    Transforms: {transforms_path}")
    print(f"    Config: {config_path}")

    # Print equivalent 2DGS training command
    cmd = (
        f"python train.py "
        f"-s {source_path} "
        f"--model_path {obj_dir}/model "
        f"--iterations {config['iterations']} "
        f"--mask_weighted "
        f"--mask_transforms {transforms_path}"
    )
    print(f"    2DGS cmd: {cmd}")

    return {
        "object_label": config["object_label"],
        "output_dir": obj_dir,
        "transforms_path": transforms_path,
        "n_views": len(dataset),
        "n_views_with_mask": sum(1 for d in dataset if d["mask_weight"].sum() > 0),
    }


# ─── Main pipeline ───────────────────────────────────────────────

def prepare_all_objects(source_path, propagated_dir, output_dir, iterations=7000):
    """
    Prepare per-object reconstruction for all instances + background.

    Returns:
        list of result dicts from train_per_object_stub
    """
    print("Loading propagated masks...")
    meta, instance_masks = load_propagated_masks(propagated_dir)

    labels = meta["instance_labels"]
    print(f"Found {len(labels)} instances: {labels}")

    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Per-object models
    for label in labels:
        dataset = prepare_per_object_dataset(source_path, instance_masks, label)
        config = create_per_object_config({}, label, output_dir, iterations)
        result = train_per_object_stub(config, dataset, source_path)
        results.append(result)

    # Background model
    print("\n  Background:")
    bg_dataset = prepare_background_dataset(source_path, instance_masks)
    bg_config = create_per_object_config({}, -1, output_dir, iterations)
    bg_config["output_dir"] = os.path.join(output_dir, "background")
    bg_result = train_per_object_stub(bg_config, bg_dataset, source_path)
    bg_result["is_background"] = True
    results.append(bg_result)

    # Save summary
    summary = {
        "source_path": source_path,
        "propagated_dir": propagated_dir,
        "n_objects": len(labels),
        "n_total_models": len(labels) + 1,
        "iterations": iterations,
        "objects": results,
    }
    summary_path = os.path.join(output_dir, "reconstruction_plan.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nReconstruction plan saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Per-Object 2DGS Reconstruction (Session A prep + Session B training)"
    )
    parser.add_argument("-s", "--source_path", required=True,
                        help="Dataset root (contains transforms_train.json)")
    parser.add_argument("--propagated", required=True,
                        help="Path to propagation results dir")
    parser.add_argument("--output", required=True,
                        help="Output dir for per-object models")
    parser.add_argument("--iterations", type=int, default=7000,
                        help="Training iterations per object")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Only prepare configs/masks, don't train (default behavior for now)")
    args = parser.parse_args()

    print("=" * 60)
    print("PER-OBJECT 2DGS RECONSTRUCTION")
    print("=" * 60)

    results = prepare_all_objects(
        args.source_path,
        args.propagated,
        args.output,
        args.iterations,
    )

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Models to train: {len(results)}")
    print(f"  Output: {args.output}")
    print(f"\nNote: Actual 2DGS training will be integrated in Session B.")
    print(f"  The mask-weighted loss functions are defined in this module.")
    print(f"  See: masked_l1_loss(), masked_ssim_loss(), combined_masked_loss()")


if __name__ == "__main__":
    main()
