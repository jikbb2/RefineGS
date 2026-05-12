#!/usr/bin/env python3
"""
Step 2: Monocular depth estimation OR rendered depth extraction.

Strategy:
  1. Try Depth Anything V2 (best quality)
  2. Fallback: Extract depth from existing 2DGS model (already trained)
  3. Fallback: DPT-Large via torch.hub

Usage:
    python -m refine_gs.depth_estimation \
        -s data/hotdog \
        --output data/hotdog/depth_maps

    # Or extract from existing 2DGS:
    python -m refine_gs.depth_estimation \
        -s data/hotdog \
        --from_2dgs output/hotdog_stage1_ckpt/point_cloud/iteration_30000/point_cloud.ply
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch


def estimate_depth_anything_v2(image_paths, output_dir):
    """Use Depth Anything V2 for monocular depth estimation."""
    try:
        from transformers import pipeline
        pipe = pipeline(task="depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=0 if torch.cuda.is_available() else -1)
        print("[OK] Depth Anything V2 loaded")
    except Exception as e:
        print(f"[WARN] Depth Anything V2 failed: {e}")
        return None

    from PIL import Image

    depth_maps = {}
    for k, img_path in enumerate(tqdm(image_paths, desc="Depth estimation")):
        image = Image.open(img_path).convert("RGB")
        result = pipe(image)
        depth = np.array(result["depth"], dtype=np.float32)

        # Normalize to metric-like scale (relative depth)
        depth = depth / (depth.max() + 1e-8)

        depth_maps[k] = depth

        # Save
        depth_path = os.path.join(output_dir, f"depth_{k:04d}.npy")
        np.save(depth_path, depth)

        # Save visualization
        vis = (depth * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, f"depth_{k:04d}_vis.png"), vis)

    return depth_maps


def estimate_dpt(image_paths, output_dir):
    """Fallback: DPT-Large via torch.hub."""
    try:
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform
        print("[OK] DPT-Large loaded")
    except Exception as e:
        print(f"[WARN] DPT load failed: {e}")
        return None

    depth_maps = {}
    for k, img_path in enumerate(tqdm(image_paths, desc="DPT depth")):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        # MiDaS outputs inverse depth; convert
        depth = depth.max() - depth  # now closer = smaller
        depth = depth / (depth.max() + 1e-8)

        depth_maps[k] = depth
        np.save(os.path.join(output_dir, f"depth_{k:04d}.npy"), depth)

        vis = (depth * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, f"depth_{k:04d}_vis.png"), vis)

    return depth_maps


def extract_depth_from_2dgs(ply_path, transforms_path, output_dir):
    """
    Extract depth maps by rendering from an existing 2DGS model.
    Projects the Gaussian centers to each view as a depth proxy.
    """
    from refine_gs.utils import load_ply_points, load_cameras_from_transforms, \
        get_intrinsic_matrix, project_points

    print(f"Extracting depth from 2DGS point cloud: {ply_path}")
    points = load_ply_points(ply_path)
    cameras = load_cameras_from_transforms(transforms_path)

    # Read one image to get dimensions
    sample_img = cv2.imread(cameras[0]["image_path"])
    H, W = sample_img.shape[:2]

    depth_maps = {}
    for k, cam in enumerate(tqdm(cameras, desc="Rendering depth")):
        K = get_intrinsic_matrix(cam["fov_x"], W, H)
        pixel_coords, depths = project_points(points, cam["w2c"], K)

        # Create depth map via splatting (nearest neighbor)
        depth_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        w = np.round(pixel_coords[:, 0]).astype(int)
        h = np.round(pixel_coords[:, 1]).astype(int)

        valid = (w >= 0) & (w < W) & (h >= 0) & (h < H) & (depths > 0)
        w, h, d = w[valid], h[valid], depths[valid]

        # Use minimum depth (front surface)
        for i in range(len(w)):
            if depth_map[h[i], w[i]] == 0 or d[i] < depth_map[h[i], w[i]]:
                depth_map[h[i], w[i]] = d[i]

        # Fill holes with dilation
        mask = (depth_map > 0).astype(np.uint8)
        depth_filled = cv2.dilate(depth_map, np.ones((5, 5)), iterations=3)
        depth_map = np.where(depth_map > 0, depth_map, depth_filled)

        # Normalize
        if depth_map.max() > 0:
            depth_norm = depth_map / depth_map.max()
        else:
            depth_norm = depth_map

        depth_maps[k] = depth_norm

        np.save(os.path.join(output_dir, f"depth_{k:04d}.npy"), depth_norm)
        vis = (depth_norm * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(output_dir, f"depth_{k:04d}_vis.png"), vis)

    return depth_maps


def load_depth_maps(depth_dir, K):
    """Load pre-computed depth maps."""
    depth_maps = {}
    for k in range(K):
        path = os.path.join(depth_dir, f"depth_{k:04d}.npy")
        if os.path.exists(path):
            depth_maps[k] = np.load(path)
    return depth_maps


def get_image_paths(source_path):
    """Get sorted image paths from transforms_train.json."""
    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    paths = []
    for frame in meta["frames"]:
        fp = frame["file_path"]
        if not os.path.isabs(fp):
            fp = os.path.join(source_path, fp)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            if os.path.exists(fp + ext):
                paths.append(fp + ext)
                break
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--from_2dgs", default=None,
                        help="Extract depth from existing PLY instead of monocular estimation")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "depth_anything", "dpt", "2dgs"])
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.source_path, "depth_maps")
    os.makedirs(output_dir, exist_ok=True)

    if args.from_2dgs or args.method == "2dgs":
        ply_path = args.from_2dgs
        transforms_path = os.path.join(args.source_path, "transforms_train.json")
        extract_depth_from_2dgs(ply_path, transforms_path, output_dir)
    else:
        image_paths = get_image_paths(args.source_path)

        result = None
        if args.method in ("auto", "depth_anything"):
            result = estimate_depth_anything_v2(image_paths, output_dir)
        if result is None and args.method in ("auto", "dpt"):
            result = estimate_dpt(image_paths, output_dir)
        if result is None:
            print("[ERROR] No depth estimation method available.")
            print("  Install: pip install transformers  (for Depth Anything V2)")
            print("  Or use: --from_2dgs path/to/point_cloud.ply")
            sys.exit(1)

    print(f"\nDepth maps saved to {output_dir}")
