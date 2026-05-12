#!/usr/bin/env python3
"""
RefineGS Session A: Full pipeline runner.

Runs Steps 1-3 sequentially:
  Step 1: SAM mask generation
  Step 2: Depth estimation
  Step 3: Mask propagation

Usage:
    python -m refine_gs.scripts.run_pipeline \
        -s data/hotdog \
        --sam_checkpoint /home/elicer/sam_vit_b_01ec64.pth \
        --points output/hotdog_stage1_ckpt/point_cloud/iteration_30000/point_cloud.ply

    # With depth from existing 2DGS model:
    python -m refine_gs.scripts.run_pipeline \
        -s data/hotdog \
        --sam_checkpoint /home/elicer/sam_vit_b_01ec64.pth \
        --points output/hotdog_stage1_ckpt/point_cloud/iteration_30000/point_cloud.ply \
        --depth_from_2dgs
"""

import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description="RefineGS Session A Pipeline")
    parser.add_argument("-s", "--source_path", required=True,
                        help="Dataset root (contains transforms_train.json)")
    parser.add_argument("--sam_checkpoint", default="/home/elicer/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--points", required=True,
                        help="Path to dense point cloud PLY")
    parser.add_argument("--depth_from_2dgs", action="store_true",
                        help="Extract depth from PLY instead of monocular estimation")
    parser.add_argument("--output", default=None,
                        help="Base output dir (default: source_path/refine_gs_data)")

    # Propagation hyperparameters
    parser.add_argument("--tau_depth", type=float, default=0.02)
    parser.add_argument("--tau_label", type=float, default=0.7)
    parser.add_argument("--skip_masks", action="store_true",
                        help="Skip mask generation (use existing)")
    parser.add_argument("--skip_depth", action="store_true",
                        help="Skip depth estimation (use existing)")
    args = parser.parse_args()

    base_output = args.output or os.path.join(args.source_path, "refine_gs_data")
    masks_dir = os.path.join(base_output, "masks")
    depth_dir = os.path.join(base_output, "depth_maps")
    propagated_dir = os.path.join(base_output, "propagated")

    os.makedirs(base_output, exist_ok=True)
    total_start = time.time()

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Mask Generation
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_masks:
        print("\n" + "=" * 60)
        print("STEP 1: Mask Generation (SAM)")
        print("=" * 60)
        t0 = time.time()

        from refine_gs.mask_generation import generate_masks_for_dataset
        generate_masks_for_dataset(
            args.source_path, masks_dir,
            args.sam_checkpoint, args.sam2_checkpoint,
        )

        print(f"Step 1 done in {time.time()-t0:.1f}s")
    else:
        print("\n[SKIP] Step 1: Using existing masks from", masks_dir)

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Depth Estimation
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_depth:
        print("\n" + "=" * 60)
        print("STEP 2: Depth Estimation")
        print("=" * 60)
        t0 = time.time()

        if args.depth_from_2dgs:
            from refine_gs.depth_estimation import extract_depth_from_2dgs
            transforms_path = os.path.join(args.source_path, "transforms_train.json")
            os.makedirs(depth_dir, exist_ok=True)
            extract_depth_from_2dgs(args.points, transforms_path, depth_dir)
        else:
            from refine_gs.depth_estimation import (
                get_image_paths, estimate_depth_anything_v2, estimate_dpt
            )
            os.makedirs(depth_dir, exist_ok=True)
            image_paths = get_image_paths(args.source_path)
            result = estimate_depth_anything_v2(image_paths, depth_dir)
            if result is None:
                result = estimate_dpt(image_paths, depth_dir)
            if result is None:
                print("[FALLBACK] Using depth from 2DGS point cloud")
                from refine_gs.depth_estimation import extract_depth_from_2dgs
                transforms_path = os.path.join(args.source_path, "transforms_train.json")
                extract_depth_from_2dgs(args.points, transforms_path, depth_dir)

        print(f"Step 2 done in {time.time()-t0:.1f}s")
    else:
        print("\n[SKIP] Step 2: Using existing depth from", depth_dir)

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Mask Propagation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3: Mask Propagation")
    print("=" * 60)
    t0 = time.time()

    from refine_gs.mask_propagation import (
        propagate_masks, reproject_labels_to_masks,
        save_propagation_results,
    )
    from refine_gs.mask_generation import load_masks
    from refine_gs.depth_estimation import load_depth_maps
    from refine_gs.utils import load_cameras_from_transforms, load_ply_points
    import cv2

    transforms_path = os.path.join(args.source_path, "transforms_train.json")
    cameras = load_cameras_from_transforms(transforms_path)
    all_masks, masks_meta = load_masks(masks_dir)
    depth_maps = load_depth_maps(depth_dir, len(cameras))
    points_3d = load_ply_points(args.points)

    sample_img = cv2.imread(cameras[0]["image_path"])
    H, W = sample_img.shape[:2]

    P_labeled, weight_vectors = propagate_masks(
        points_3d, cameras, depth_maps, all_masks,
        tau_depth=args.tau_depth,
        tau_label=args.tau_label,
    )

    refined_masks = reproject_labels_to_masks(P_labeled, points_3d, cameras, H, W)
    save_propagation_results(propagated_dir, P_labeled, points_3d,
                              refined_masks, cameras)

    print(f"Step 3 done in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Masks: {masks_dir}")
    print(f"  Depth: {depth_dir}")
    print(f"  Propagated: {propagated_dir}")
    print(f"    - labeled_pointcloud.npz")
    print(f"    - refined_masks/view_XXXX/instance_NNN.png")
    print(f"\nNext step: Per-object 2DGS reconstruction (Session B)")


if __name__ == "__main__":
    main()
