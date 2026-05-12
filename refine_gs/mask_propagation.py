#!/usr/bin/env python3
"""
Step 3: Mask Propagation (Split&Splat Section 3.1.3 reimplementation).

Core algorithm:
  1. Project P_dense to each view, depth-filter for surface consistency
  2. Extract points inside eroded masks
  3. Sequential label propagation via virtual mask IoU matching
  4. Majority voting for final label assignment
  5. Reproject labeled point cloud to update 2D masks

Usage:
    python -m refine_gs.mask_propagation \
        -s data/hotdog \
        --masks data/hotdog/masks \
        --depth data/hotdog/depth_maps \
        --points output/hotdog/point_cloud/iteration_30000/point_cloud.ply \
        --output data/hotdog/propagated
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict

from refine_gs.utils import (
    load_cameras_from_transforms, get_intrinsic_matrix,
    project_points, load_ply_points, compute_iou,
    erode_mask, save_labeled_pointcloud, masks_to_colored_image,
)
from refine_gs.mask_generation import load_masks
from refine_gs.depth_estimation import load_depth_maps


def depth_filter_points(points_3d, camera, depth_map, K, tau_depth=0.02):
    """
    Eq. 1-2: Project points to view, keep only depth-consistent surface points.

    Returns:
        valid_indices: indices into points_3d that pass the filter
        pixel_coords: [N_valid, 2] pixel coordinates
        proj_depths: [N_valid] projected depths
    """
    H, W = depth_map.shape
    pixel_coords, proj_depths = project_points(points_3d, camera["w2c"], K)

    w = pixel_coords[:, 0]
    h = pixel_coords[:, 1]

    # In-bounds check
    in_bounds = (w >= 0) & (w < W) & (h >= 0) & (h < H) & (proj_depths > 0)
    indices = np.where(in_bounds)[0]

    if len(indices) == 0:
        return np.array([], dtype=int), np.empty((0, 2)), np.array([])

    wi = np.round(w[indices]).astype(int)
    hi = np.round(h[indices]).astype(int)
    di = proj_depths[indices]

    # Depth consistency check
    depth_at_pixel = depth_map[hi, wi]

    # Handle normalized vs absolute depth:
    # If depth_map is normalized (0-1), normalize proj_depths too
    if depth_map.max() <= 1.0 + 1e-3:
        d_max = proj_depths[in_bounds].max()
        if d_max > 0:
            di_norm = di / d_max
            depth_consistent = np.abs(depth_at_pixel - di_norm) < tau_depth
        else:
            depth_consistent = np.ones(len(di), dtype=bool)
    else:
        depth_consistent = np.abs(depth_at_pixel - di) < tau_depth

    valid = indices[depth_consistent]
    return valid, pixel_coords[valid], proj_depths[valid]


def extract_mask_points(valid_indices, pixel_coords, masks_k, erode_pixels=3):
    """
    Eq. 3: For each mask j in view k, find points inside the eroded mask.

    Returns:
        P_k_j: dict {mask_label: array of point indices}
    """
    P_k_j = {}

    for j, mask_data in enumerate(masks_k):
        seg = mask_data["segmentation"]
        eroded = erode_mask(seg, pixels=erode_pixels)

        w = np.round(pixel_coords[:, 0]).astype(int)
        h = np.round(pixel_coords[:, 1]).astype(int)

        # Clamp to image bounds
        H, W = eroded.shape
        w = np.clip(w, 0, W - 1)
        h = np.clip(h, 0, H - 1)

        inside = eroded[h, w]
        point_indices = valid_indices[inside]

        if len(point_indices) > 0:
            P_k_j[j] = point_indices

    return P_k_j


def remove_noise_dbscan(points_3d, point_indices, eps=0.05, min_samples=5):
    """Apply DBSCAN to remove isolated points."""
    if len(point_indices) < min_samples:
        return point_indices

    coords = points_3d[point_indices]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    clean = point_indices[labels >= 0]
    return clean


def create_virtual_mask(point_indices, points_3d, camera, K, H, W):
    """Reproject labeled 3D points to create a virtual mask in view k."""
    if len(point_indices) == 0:
        return np.zeros((H, W), dtype=bool)

    pixel_coords, depths = project_points(points_3d[point_indices], camera["w2c"], K)
    w = np.round(pixel_coords[:, 0]).astype(int)
    h = np.round(pixel_coords[:, 1]).astype(int)

    valid = (w >= 0) & (w < W) & (h >= 0) & (h < H) & (depths > 0)
    w, h = w[valid], h[valid]

    mask = np.zeros((H, W), dtype=bool)
    if len(w) > 0:
        mask[h, w] = True
        # Dilate to fill gaps
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7)), iterations=2).astype(bool)

    return mask


def match_labels_by_iou(prev_labels_points, curr_masks_points,
                         points_3d, camera_k, K, H, W):
    """
    Match current view's mask labels to previous view's labels via IoU.

    Creates virtual masks from prev_labels_points, computes IoU with
    curr_masks_points projected masks, and returns remapping.
    """
    remap = {}

    # Create virtual masks from previous labels
    virtual_masks = {}
    for prev_label, prev_indices in prev_labels_points.items():
        virtual_masks[prev_label] = create_virtual_mask(
            prev_indices, points_3d, camera_k, K, H, W
        )

    # Create masks for current view's labels
    curr_projected_masks = {}
    for curr_label, curr_indices in curr_masks_points.items():
        curr_projected_masks[curr_label] = create_virtual_mask(
            curr_indices, points_3d, camera_k, K, H, W
        )

    # IoU matching: for each current label, find best matching previous label
    for curr_label, curr_mask in curr_projected_masks.items():
        best_iou = 0
        best_prev = None
        for prev_label, virtual_mask in virtual_masks.items():
            iou = compute_iou(curr_mask, virtual_mask)
            if iou > best_iou:
                best_iou = iou
                best_prev = prev_label

        if best_prev is not None and best_iou > 0.1:
            remap[curr_label] = best_prev
        else:
            # New object not seen in previous views
            remap[curr_label] = curr_label  # keep as is, will get new global ID

    return remap


def propagate_masks(points_3d, cameras, depth_maps, all_masks,
                    tau_depth=0.02, tau_label=0.7, lambda_init=0.5,
                    dbscan_eps=0.05, dbscan_min=5, erode_pixels=3):
    """
    Full mask propagation algorithm (Section 3.1.3).

    Args:
        points_3d: [N, 3] dense point cloud
        cameras: list of camera dicts
        depth_maps: {k: [H, W] depth map}
        all_masks: {k: list of mask dicts}

    Returns:
        P_labeled: dict {point_index: global_label}
        weight_vectors: dict {point_index: {label: score}}
    """
    K_views = len(cameras)
    N = len(points_3d)

    # Read image dimensions
    sample_img = cv2.imread(cameras[0]["image_path"])
    H, W = sample_img.shape[:2]
    K_intrinsic = get_intrinsic_matrix(cameras[0]["fov_x"], W, H)

    # Global label counter and weight vectors
    global_label_map = {}  # local (view, mask_id) → global label
    next_global_label = 0
    weight_vectors = defaultdict(lambda: defaultdict(float))

    prev_global_P_k_j = None  # previous view's {global_label: point_indices}

    print(f"\nMask propagation: {K_views} views, {N} points")

    for k in tqdm(range(K_views), desc="Propagating labels"):
        if k not in all_masks or k not in depth_maps:
            continue

        cam = cameras[k]

        # Step 1: Project + depth filter
        valid_indices, pixel_coords, proj_depths = depth_filter_points(
            points_3d, cam, depth_maps[k], K_intrinsic, tau_depth
        )

        if len(valid_indices) == 0:
            continue

        # Step 2: Extract mask points (with erosion)
        P_k_j_local = extract_mask_points(
            valid_indices, pixel_coords, all_masks[k], erode_pixels
        )

        # Step 3: DBSCAN noise removal
        for j in list(P_k_j_local.keys()):
            P_k_j_local[j] = remove_noise_dbscan(
                points_3d, P_k_j_local[j], dbscan_eps, dbscan_min
            )
            if len(P_k_j_local[j]) == 0:
                del P_k_j_local[j]

        # Step 4: Label matching via IoU with previous view
        global_P_k_j = {}

        if prev_global_P_k_j is not None and len(P_k_j_local) > 0:
            remap = match_labels_by_iou(
                prev_global_P_k_j, P_k_j_local,
                points_3d, cam, K_intrinsic, H, W
            )

            used_globals = set()
            for local_label, point_indices in P_k_j_local.items():
                if local_label in remap:
                    gl = remap[local_label]
                    if gl in used_globals:
                        # Conflict: assign new global label
                        gl = next_global_label
                        next_global_label += 1
                    used_globals.add(gl)
                else:
                    gl = next_global_label
                    next_global_label += 1
                global_P_k_j[gl] = point_indices
        else:
            # First view: assign initial global labels
            for local_label, point_indices in P_k_j_local.items():
                gl = next_global_label
                next_global_label += 1
                global_P_k_j[gl] = point_indices

        # Step 5: Update weight vectors
        for gl, point_indices in global_P_k_j.items():
            for idx in point_indices:
                if weight_vectors[idx][gl] == 0:
                    weight_vectors[idx][gl] = 1 + lambda_init
                else:
                    weight_vectors[idx][gl] += 1

        prev_global_P_k_j = global_P_k_j

    # Step 6: Majority voting
    print("\nMajority voting...")
    P_labeled = {}
    for idx, scores in weight_vectors.items():
        total = sum(scores.values())
        if total == 0:
            continue
        best_label = max(scores, key=scores.get)
        confidence = scores[best_label] / total
        if confidence >= tau_label:
            P_labeled[idx] = best_label

    # Statistics
    labels = list(set(P_labeled.values()))
    print(f"  Labeled points: {len(P_labeled)} / {N} ({100*len(P_labeled)/N:.1f}%)")
    print(f"  Unique labels: {len(labels)}")
    for label in sorted(labels):
        count = sum(1 for v in P_labeled.values() if v == label)
        print(f"    Label {label}: {count} points")

    return P_labeled, dict(weight_vectors)


def reproject_labels_to_masks(P_labeled, points_3d, cameras, H, W):
    """
    Step 5 (post-voting): Reproject labeled point cloud to update 2D masks.
    Returns refined masks with global instance IDs.
    """
    K_intrinsic = get_intrinsic_matrix(cameras[0]["fov_x"], W, H)

    # Get all labeled point indices and their labels
    labeled_indices = np.array(list(P_labeled.keys()))
    labeled_points = points_3d[labeled_indices]
    labels_array = np.array([P_labeled[idx] for idx in labeled_indices])
    unique_labels = np.unique(labels_array)

    refined_masks = {}

    for k, cam in enumerate(tqdm(cameras, desc="Reprojecting masks")):
        pixel_coords, depths = project_points(labeled_points, cam["w2c"], K_intrinsic)

        w = np.round(pixel_coords[:, 0]).astype(int)
        h = np.round(pixel_coords[:, 1]).astype(int)
        valid = (w >= 0) & (w < W) & (h >= 0) & (h < H) & (depths > 0)

        view_masks = {}
        for label in unique_labels:
            mask = np.zeros((H, W), dtype=bool)
            label_valid = valid & (labels_array == label)
            if label_valid.any():
                ww, hh = w[label_valid], h[label_valid]
                mask[hh, ww] = True
                # Dilate to fill gaps from sparse projection
                mask = cv2.dilate(mask.astype(np.uint8),
                                  np.ones((5, 5)), iterations=3).astype(bool)
            view_masks[label] = mask

        refined_masks[k] = view_masks

    return refined_masks


def save_propagation_results(output_dir, P_labeled, points_3d,
                              refined_masks, cameras):
    """Save all propagation results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save labeled point cloud
    labeled_indices = list(P_labeled.keys())
    labeled_points = points_3d[labeled_indices]
    labels = [P_labeled[idx] for idx in labeled_indices]
    save_labeled_pointcloud(
        os.path.join(output_dir, "labeled_pointcloud.npz"),
        labeled_points, labels
    )

    # Save refined masks
    sample_img = cv2.imread(cameras[0]["image_path"])
    H, W = sample_img.shape[:2]

    masks_dir = os.path.join(output_dir, "refined_masks")
    os.makedirs(masks_dir, exist_ok=True)

    for k, view_masks in refined_masks.items():
        view_dir = os.path.join(masks_dir, f"view_{k:04d}")
        os.makedirs(view_dir, exist_ok=True)

        for label, mask in view_masks.items():
            cv2.imwrite(
                os.path.join(view_dir, f"instance_{label:03d}.png"),
                (mask.astype(np.uint8) * 255)
            )

        # Visualization
        vis = masks_to_colored_image(view_masks, H, W)
        cv2.imwrite(os.path.join(view_dir, "vis_refined.png"), vis)

    # Save metadata
    meta = {
        "n_points": len(points_3d),
        "n_labeled": len(P_labeled),
        "n_instances": len(set(P_labeled.values())),
        "instance_labels": sorted(set(P_labeled.values())),
    }
    with open(os.path.join(output_dir, "propagation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Mask propagation")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--masks", required=True, help="Path to generated masks dir")
    parser.add_argument("--depth", required=True, help="Path to depth maps dir")
    parser.add_argument("--points", required=True, help="Path to PLY point cloud")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--tau_depth", type=float, default=0.02)
    parser.add_argument("--tau_label", type=float, default=0.7)
    parser.add_argument("--lambda_init", type=float, default=0.5)
    parser.add_argument("--dbscan_eps", type=float, default=0.05)
    parser.add_argument("--erode_pixels", type=int, default=3)
    args = parser.parse_args()

    # Load inputs
    print("Loading inputs...")
    transforms_path = os.path.join(args.source_path, "transforms_train.json")
    cameras = load_cameras_from_transforms(transforms_path)

    all_masks, masks_meta = load_masks(args.masks)
    K_views = len(cameras)
    depth_maps = load_depth_maps(args.depth, K_views)

    points_3d = load_ply_points(args.points)

    sample_img = cv2.imread(cameras[0]["image_path"])
    H, W = sample_img.shape[:2]

    print(f"  Cameras: {len(cameras)}")
    print(f"  Masks: {len(all_masks)} views")
    print(f"  Depth maps: {len(depth_maps)} views")
    print(f"  Points: {len(points_3d)}")
    print(f"  Image size: {W}x{H}")

    # Run propagation
    P_labeled, weight_vectors = propagate_masks(
        points_3d, cameras, depth_maps, all_masks,
        tau_depth=args.tau_depth,
        tau_label=args.tau_label,
        lambda_init=args.lambda_init,
        dbscan_eps=args.dbscan_eps,
        erode_pixels=args.erode_pixels,
    )

    # Reproject to 2D masks
    refined_masks = reproject_labels_to_masks(P_labeled, points_3d, cameras, H, W)

    # Save results
    save_propagation_results(args.output, P_labeled, points_3d,
                              refined_masks, cameras)


if __name__ == "__main__":
    main()
