#!/usr/bin/env python3
"""
Visualization & verification script for RefineGS Session A outputs.

Generates visual comparisons at each stage:
  1. SAM masks overlay on original images
  2. Depth map side-by-side with images
  3. Propagated masks vs original SAM masks
  4. Per-object mask coverage heatmap
  5. Labeled point cloud statistics

Usage:
    python -m refine_gs.scripts.visualize_masks \
        -s data/hotdog \
        --base_dir data/hotdog/refine_gs_data \
        --output data/hotdog/refine_gs_data/visualizations

    # Specific stages only:
    python -m refine_gs.scripts.visualize_masks \
        -s data/hotdog \
        --base_dir data/hotdog/refine_gs_data \
        --stage masks depth propagation
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ─── Color palette ───────────────────────────────────────────────

PALETTE = np.array([
    [230, 25, 75],   [60, 180, 75],   [255, 225, 25],  [0, 130, 200],
    [245, 130, 48],  [145, 30, 180],  [70, 240, 240],  [240, 50, 230],
    [210, 245, 60],  [250, 190, 212], [0, 128, 128],   [220, 190, 255],
    [170, 110, 40],  [255, 250, 200], [128, 0, 0],     [170, 255, 195],
    [128, 128, 0],   [255, 215, 180], [0, 0, 128],     [128, 128, 128],
], dtype=np.uint8)


def get_color(idx):
    return PALETTE[idx % len(PALETTE)]


# ─── Stage 1: SAM Mask Visualization ────────────────────────────

def visualize_sam_masks(source_path, masks_dir, output_dir, max_views=8):
    """Overlay SAM masks on original images."""
    print("\n--- SAM Masks ---")
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(masks_dir, "masks_meta.json")
    if not os.path.exists(meta_path):
        print(f"  [SKIP] No masks_meta.json found at {masks_dir}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    views_to_show = sorted(meta.keys(), key=int)[:max_views]

    for k_str in views_to_show:
        k = int(k_str)
        info = meta[k_str]
        img_path = info["image_path"]

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        overlay = img.copy()
        mask_overlay = np.zeros((H, W, 3), dtype=np.uint8)

        view_dir = os.path.join(masks_dir, f"view_{k:04d}")
        n_masks = info["num_masks"]

        for j in range(n_masks):
            mask_file = os.path.join(view_dir, f"mask_{j:03d}.png")
            if not os.path.exists(mask_file):
                continue
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) > 127
            color = get_color(j)
            mask_overlay[mask] = color

        # Blend
        alpha = 0.5
        blended = cv2.addWeighted(img, 1 - alpha, mask_overlay, alpha, 0)

        # Side by side: original | overlay
        canvas = np.hstack([img, blended])

        # Add text
        cv2.putText(canvas, f"View {k}: {n_masks} masks",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = os.path.join(output_dir, f"sam_view_{k:04d}.png")
        cv2.imwrite(out_path, canvas)

    print(f"  Saved {len(views_to_show)} visualizations to {output_dir}")


# ─── Stage 2: Depth Visualization ───────────────────────────────

def visualize_depth(source_path, depth_dir, output_dir, max_views=8):
    """Side-by-side depth maps with original images."""
    print("\n--- Depth Maps ---")
    os.makedirs(output_dir, exist_ok=True)

    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    count = 0
    for k, frame in enumerate(meta["frames"]):
        if count >= max_views:
            break

        depth_path = os.path.join(depth_dir, f"depth_{k:04d}.npy")
        if not os.path.exists(depth_path):
            continue

        # Load image
        fp = frame["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(source_path, fp)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            if os.path.exists(fp + ext):
                fp = fp + ext
                break

        img = cv2.imread(fp)
        if img is None:
            continue

        H, W = img.shape[:2]

        # Load and colorize depth
        depth = np.load(depth_path)
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H))

        depth_vis = (depth / (depth.max() + 1e-8) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        canvas = np.hstack([img, depth_color])
        cv2.putText(canvas, f"View {k} | depth range: [{depth.min():.3f}, {depth.max():.3f}]",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_path = os.path.join(output_dir, f"depth_view_{k:04d}.png")
        cv2.imwrite(out_path, canvas)
        count += 1

    print(f"  Saved {count} visualizations to {output_dir}")


# ─── Stage 3: Propagated Masks Visualization ────────────────────

def visualize_propagation(source_path, masks_dir, propagated_dir, output_dir, max_views=8):
    """Compare original SAM masks vs propagated/refined masks."""
    print("\n--- Propagation Results ---")
    os.makedirs(output_dir, exist_ok=True)

    # Load propagation meta
    prop_meta_path = os.path.join(propagated_dir, "propagation_meta.json")
    if not os.path.exists(prop_meta_path):
        print(f"  [SKIP] No propagation_meta.json at {propagated_dir}")
        return

    with open(prop_meta_path) as f:
        prop_meta = json.load(f)

    print(f"  Labeled: {prop_meta['n_labeled']}/{prop_meta['n_points']} points")
    print(f"  Instances: {prop_meta['n_instances']}")

    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    refined_masks_dir = os.path.join(propagated_dir, "refined_masks")
    sam_meta_path = os.path.join(masks_dir, "masks_meta.json")

    has_sam = os.path.exists(sam_meta_path)
    if has_sam:
        with open(sam_meta_path) as f:
            sam_meta = json.load(f)

    count = 0
    for k, frame in enumerate(meta["frames"]):
        if count >= max_views:
            break

        view_dir = os.path.join(refined_masks_dir, f"view_{k:04d}")
        if not os.path.exists(view_dir):
            continue

        # Load image
        fp = frame["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(source_path, fp)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            if os.path.exists(fp + ext):
                fp = fp + ext
                break

        img = cv2.imread(fp)
        if img is None:
            continue
        H, W = img.shape[:2]

        # Load refined masks
        refined_overlay = np.zeros((H, W, 3), dtype=np.uint8)
        instance_files = sorted(Path(view_dir).glob("instance_*.png"))
        for mf in instance_files:
            label = int(mf.stem.split("_")[1])
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) > 127
            refined_overlay[mask] = get_color(label)

        refined_blended = cv2.addWeighted(img, 0.5, refined_overlay, 0.5, 0)

        # Load SAM masks for comparison
        if has_sam and str(k) in sam_meta:
            sam_overlay = np.zeros((H, W, 3), dtype=np.uint8)
            sam_view_dir = os.path.join(masks_dir, f"view_{k:04d}")
            for j in range(sam_meta[str(k)]["num_masks"]):
                mf = os.path.join(sam_view_dir, f"mask_{j:03d}.png")
                if os.path.exists(mf):
                    mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE) > 127
                    sam_overlay[mask] = get_color(j)
            sam_blended = cv2.addWeighted(img, 0.5, sam_overlay, 0.5, 0)

            # 3-panel: original | SAM | propagated
            canvas = np.hstack([img, sam_blended, refined_blended])
            cv2.putText(canvas, "Original", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, "SAM masks", (W + 10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, "Propagated", (2 * W + 10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # 2-panel: original | propagated
            canvas = np.hstack([img, refined_blended])
            cv2.putText(canvas, "Original", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, f"Propagated ({len(instance_files)} instances)", (W + 10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_path = os.path.join(output_dir, f"prop_view_{k:04d}.png")
        cv2.imwrite(out_path, canvas)
        count += 1

    print(f"  Saved {count} comparison images to {output_dir}")


# ─── Stage 4: Coverage & Statistics ──────────────────────────────

def visualize_coverage(propagated_dir, output_dir):
    """Generate per-instance coverage statistics and heatmap."""
    print("\n--- Coverage Statistics ---")
    os.makedirs(output_dir, exist_ok=True)

    # Load labeled point cloud
    npz_path = os.path.join(propagated_dir, "labeled_pointcloud.npz")
    if not os.path.exists(npz_path):
        print(f"  [SKIP] No labeled_pointcloud.npz at {propagated_dir}")
        return

    data = np.load(npz_path)
    points = data["points"]
    labels = data["labels"]

    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"  Total labeled points: {len(points)}")
    print(f"  Unique labels: {len(unique_labels)}")
    for label, count in zip(unique_labels, counts):
        pct = 100.0 * count / len(points)
        print(f"    Label {label}: {count} points ({pct:.1f}%)")

    # Save stats as JSON
    stats = {
        "total_points": int(len(points)),
        "unique_labels": int(len(unique_labels)),
        "per_label": {
            int(l): {"count": int(c), "percentage": float(100.0 * c / len(points))}
            for l, c in zip(unique_labels, counts)
        }
    }
    stats_path = os.path.join(output_dir, "coverage_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path}")

    # Point cloud bounding box per label
    print("\n  Bounding boxes (world coords):")
    for label in unique_labels:
        pts = points[labels == label]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        extent = bbox_max - bbox_min
        print(f"    Label {label}: center={pts.mean(axis=0).round(3)}, "
              f"extent={extent.round(3)}")


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize RefineGS Session A outputs"
    )
    parser.add_argument("-s", "--source_path", required=True,
                        help="Dataset root")
    parser.add_argument("--base_dir", required=True,
                        help="RefineGS data dir (contains masks/, depth_maps/, propagated/)")
    parser.add_argument("--output", default=None,
                        help="Output dir for visualizations (default: base_dir/visualizations)")
    parser.add_argument("--stage", nargs="*",
                        default=["masks", "depth", "propagation", "coverage"],
                        choices=["masks", "depth", "propagation", "coverage"],
                        help="Which stages to visualize")
    parser.add_argument("--max_views", type=int, default=8,
                        help="Max number of views to visualize per stage")
    args = parser.parse_args()

    masks_dir = os.path.join(args.base_dir, "masks")
    depth_dir = os.path.join(args.base_dir, "depth_maps")
    propagated_dir = os.path.join(args.base_dir, "propagated")
    output_dir = args.output or os.path.join(args.base_dir, "visualizations")

    print("=" * 60)
    print("RefineGS VISUALIZATION")
    print("=" * 60)
    print(f"  Source: {args.source_path}")
    print(f"  Base dir: {args.base_dir}")
    print(f"  Stages: {args.stage}")

    if "masks" in args.stage:
        visualize_sam_masks(
            args.source_path, masks_dir,
            os.path.join(output_dir, "masks"),
            args.max_views
        )

    if "depth" in args.stage:
        visualize_depth(
            args.source_path, depth_dir,
            os.path.join(output_dir, "depth"),
            args.max_views
        )

    if "propagation" in args.stage:
        visualize_propagation(
            args.source_path, masks_dir, propagated_dir,
            os.path.join(output_dir, "propagation"),
            args.max_views
        )

    if "coverage" in args.stage:
        visualize_coverage(
            propagated_dir,
            os.path.join(output_dir, "coverage")
        )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
