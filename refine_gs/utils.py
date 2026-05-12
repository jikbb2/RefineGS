#!/usr/bin/env python3
"""
RefineGS utilities: camera projection, IoU computation, visualization.
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path


# ─── Camera helpers ──────────────────────────────────────────────

def load_cameras_from_transforms(transforms_path):
    """
    Load camera intrinsics/extrinsics from NeRF-synthetic transforms_train.json.
    Returns list of dicts with keys: K, W2C, C2W, image_path, width, height.
    """
    with open(transforms_path) as f:
        meta = json.load(f)

    base_dir = os.path.dirname(transforms_path)
    fov_x = meta["camera_angle_x"]
    frames = meta["frames"]

    # Blender (OpenGL) → OpenCV coordinate conversion
    # Blender: +X right, +Y up, -Z forward
    # OpenCV:  +X right, +Y down, +Z forward
    blender_to_opencv = np.diag([1, -1, -1, 1]).astype(np.float64)

    cameras = []
    for frame in frames:
        c2w_blender = np.array(frame["transform_matrix"], dtype=np.float64)  # [4,4]
        c2w = c2w_blender @ blender_to_opencv  # convert to OpenCV convention
        w2c = np.linalg.inv(c2w)

        # Resolve image path
        file_path = frame["file_path"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(base_dir, file_path)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            if os.path.exists(file_path + ext):
                file_path = file_path + ext
                break

        cameras.append({
            "c2w": c2w,
            "w2c": w2c,
            "fov_x": fov_x,
            "image_path": file_path,
        })

    return cameras


def get_intrinsic_matrix(fov_x, W, H):
    """Compute 3x3 intrinsic matrix from horizontal FOV and image size."""
    fx = W / (2.0 * np.tan(fov_x / 2.0))
    fy = fx  # square pixels
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def project_points(points_3d, w2c, K):
    """
    Project 3D points to 2D image plane.

    Args:
        points_3d: [N, 3] world coordinates
        w2c: [4, 4] world-to-camera matrix
        K: [3, 3] intrinsic matrix

    Returns:
        pixel_coords: [N, 2] (w, h) float
        depths: [N] projected depth (camera z)
    """
    N = points_3d.shape[0]
    # To homogeneous
    pts_h = np.hstack([points_3d, np.ones((N, 1))])  # [N, 4]
    # World to camera
    pts_cam = (w2c @ pts_h.T).T[:, :3]  # [N, 3]
    depths = pts_cam[:, 2]  # z in camera frame

    # Project
    pts_2d = (K @ pts_cam.T).T  # [N, 3]
    pixel_coords = pts_2d[:, :2] / (pts_2d[:, 2:3] + 1e-8)  # [N, 2]

    return pixel_coords, depths


# ─── Mask helpers ────────────────────────────────────────────────

def compute_iou(mask_a, mask_b):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def erode_mask(mask, pixels=3):
    """Erode binary mask by given pixels."""
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def masks_to_colored_image(masks, H, W):
    """Convert dict of binary masks to colored visualization."""
    palette = np.array([
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
        [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
    ], dtype=np.uint8)

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for i, (label, mask) in enumerate(masks.items()):
        color = palette[i % len(palette)]
        vis[mask > 0] = color
    return vis


# ─── Point cloud helpers ─────────────────────────────────────────

def load_ply_points(ply_path):
    """Load point positions from a PLY file (simple parser)."""
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
        return xyz.astype(np.float64)
    except ImportError:
        # Fallback: manual parsing
        with open(ply_path, 'rb') as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
            # Parse header for vertex count
            for line in header.decode().split('\n'):
                if line.startswith('element vertex'):
                    n_vertices = int(line.split()[-1])
                    break
            # Read binary data (assuming float32 x,y,z as first 3 properties)
            data = np.frombuffer(f.read(), dtype=np.float32)
            # Determine stride from header
            n_props = header.decode().count('property float') + header.decode().count('property double')
            stride = n_props if n_props > 0 else data.shape[0] // n_vertices
            data = data[:n_vertices * stride].reshape(n_vertices, stride)
            return data[:, :3].astype(np.float64)


def save_labeled_pointcloud(path, points, labels):
    """Save labeled point cloud as NPZ."""
    np.savez(path,
             points=points.astype(np.float32),
             labels=np.array(labels, dtype=np.int32))
    print(f"Saved labeled point cloud: {path}")
    print(f"  Points: {len(points)}, Labels: {len(set(labels))}")
