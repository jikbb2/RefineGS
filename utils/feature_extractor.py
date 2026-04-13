#!/usr/bin/env python3
"""
Stage 2: Semantic Feature Extractor  (memory-efficient version)
Precomputes GT semantic feature maps for all training images.

Pipeline:
  1. SAM  -> instance masks per image
  2. CLIP ViT-B/32 -> per-instance embeddings (crop per SAM mask)
  3. DINOv2 ViT-S/14 -> dense patch features -> bilinear upsample
  4. Concatenate [CLIP 512D | DINO 384D] = 896D per pixel
  5. Fit PCA (896D -> 32D) on sampled pixels from all images
  6. Apply PCA + normalize -> save [32, H, W] per image

Memory strategy:
  - Pass 1: extract raw [896,H,W], save temp file to disk, sample pixels for PCA
  - PCA fit on sampled pixels only (~300K x 896 = ~1GB)
  - Pass 2: load each temp file, apply PCA, save final [32,H,W], delete temp

Usage:
  python utils/feature_extractor.py \
    -s data/hotdog \
    --sam_checkpoint /home/elicer/sam_vit_b_01ec64.pth \
    --feature_dim 32

Output (written to {source_path}/semantic_feature_cache/):
  {image_name}.pt   ->  torch.Tensor [32, H, W]  float32
  pca_model.pkl     ->  dict with 'pca' and 'proj_std'
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_models(sam_checkpoint: str, device: torch.device):
    print("Loading CLIP ViT-B/32 ...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    print("Loading DINOv2 ViT-S/14 ...")
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino_model = dino_model.to(device).eval()

    print(f"Loading SAM from {sam_checkpoint} ...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )

    return clip_model, clip_preprocess, dino_model, mask_generator


# ---------------------------------------------------------------------
# CLIP + SAM  ->  [512, H, W]
# ---------------------------------------------------------------------

def extract_clip_sam_features(
    image_np: np.ndarray,
    clip_model,
    clip_preprocess,
    mask_generator,
    device: torch.device,
) -> torch.Tensor:
    H, W = image_np.shape[:2]

    with torch.no_grad():
        global_inp = clip_preprocess(Image.fromarray(image_np)).unsqueeze(0).to(device)
        global_feat = clip_model.encode_image(global_inp).float()
        global_feat = F.normalize(global_feat, dim=-1).squeeze(0)

    feature_map = (
        global_feat.unsqueeze(-1).unsqueeze(-1)
        .expand(-1, H, W)
        .clone()
        .cpu()
    )

    masks = mask_generator.generate(image_np)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for mask_data in masks:
        seg = mask_data["segmentation"]
        x, y, w, h = (int(v) for v in mask_data["bbox"])
        if w < 8 or h < 8:
            continue

        crop = image_np[y : y + h, x : x + w]
        with torch.no_grad():
            crop_inp = clip_preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
            crop_feat = clip_model.encode_image(crop_inp).float()
            crop_feat = F.normalize(crop_feat, dim=-1).squeeze(0)

        mask_t = torch.tensor(seg, dtype=torch.bool)
        n_px = mask_t.sum()
        feature_map[:, mask_t] = crop_feat.cpu().unsqueeze(-1).expand(-1, n_px)

    return feature_map


# ---------------------------------------------------------------------
# DINOv2  ->  [384, H, W]
# ---------------------------------------------------------------------

@torch.no_grad()
def extract_dino_features(
    image_np: np.ndarray,
    dino_model,
    device: torch.device,
    patch_size: int = 14,
) -> torch.Tensor:
    H, W = image_np.shape[:2]

    H_p = (H // patch_size) * patch_size
    W_p = (W // patch_size) * patch_size
    img_pil = Image.fromarray(image_np).resize((W_p, H_p), Image.BILINEAR)
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = (img_arr - mean) / std

    img_t = (
        torch.from_numpy(img_arr)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    feat_dict  = dino_model.forward_features(img_t)
    patch_tok  = feat_dict["x_norm_patchtokens"]
    DINO_DIM   = patch_tok.shape[-1]

    n_h = H_p // patch_size
    n_w = W_p // patch_size
    spatial = (
        patch_tok.squeeze(0)
        .reshape(n_h, n_w, DINO_DIM)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    dino_feat = F.interpolate(spatial, size=(H, W), mode="bilinear", align_corners=False)
    dino_feat = F.normalize(dino_feat.squeeze(0), dim=0)

    return dino_feat.cpu()


# ---------------------------------------------------------------------
# PCA  (896D -> 32D)
# ---------------------------------------------------------------------

def apply_pca_normalize(feat_np: np.ndarray, pca, proj_std: np.ndarray) -> np.ndarray:
    D, H, W = feat_np.shape
    flat    = feat_np.reshape(D, -1).T
    reduced = pca.transform(flat)
    reduced = (reduced / proj_std).astype(np.float32)
    return reduced.T.reshape(-1, H, W)


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

def load_training_images(source_path: str):
    transforms_path = os.path.join(source_path, "transforms_train.json")
    with open(transforms_path) as f:
        meta = json.load(f)

    items = []
    for frame in meta["frames"]:
        file_path = frame["file_path"]
        abs_path = file_path
        if not os.path.isabs(file_path):
            abs_path = os.path.join(source_path, file_path)
        for ext in ("", ".png", ".jpg", ".jpeg"):
            candidate = abs_path + ext
            if os.path.exists(candidate):
                # Unique name: include parent dir  (e.g. "train_000_rgba")
                _p = Path(candidate)
                if _p.parent.name not in ('.', ''):
                    name = f"{_p.parent.name}_{_p.stem}"
                else:
                    name = _p.stem
                items.append((name, candidate))
                break
    return items


# ---------------------------------------------------------------------
# Main  (memory-efficient: disk-backed pipeline)
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path",    required=True)
    parser.add_argument("--sam_checkpoint",       default="/home/elicer/sam_vit_b_01ec64.pth")
    parser.add_argument("--feature_dim",          type=int, default=32)
    parser.add_argument("--samples_per_image",    type=int, default=3000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Output directory --
    cache_dir = os.path.join(args.source_path, "semantic_feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    pca_path  = os.path.join(cache_dir, "pca_model.pkl")
    tmp_dir   = os.path.join(cache_dir, "_tmp_raw")
    os.makedirs(tmp_dir, exist_ok=True)

    if os.path.exists(pca_path):
        print(f"[INFO] Cache already exists at {cache_dir}")
        print("       Delete it to recompute.")
        return

    # -- SAM checkpoint check --
    if not os.path.exists(args.sam_checkpoint):
        print(f"[ERROR] SAM checkpoint not found: {args.sam_checkpoint}")
        print("Download with:")
        print("  wget -P /home/elicer/ "
              "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        sys.exit(1)

    # -- Load models --
    clip_model, clip_preprocess, dino_model, mask_generator = load_models(
        args.sam_checkpoint, device
    )

    # -- Image list --
    image_items = load_training_images(args.source_path)
    print(f"Found {len(image_items)} training images")

    # ================================================================
    # Pass 1: Extract raw [896,H,W] → save to disk + sample for PCA
    # ================================================================
    print("\n=== Pass 1 : Extracting raw features (disk-backed) ===")
    pca_samples = []      # list of [samples_per_image, 896] arrays
    image_names = []      # ordered list of names

    for name, img_path in tqdm(image_items, desc="Images"):
        tmp_path = os.path.join(tmp_dir, f"{name}.npy")

        # Skip if temp file already exists (resume support)
        if os.path.exists(tmp_path):
            raw = np.load(tmp_path, mmap_mode='r')
        else:
            image_np = np.array(Image.open(img_path).convert("RGB"))

            clip_feat = extract_clip_sam_features(
                image_np, clip_model, clip_preprocess, mask_generator, device
            )
            dino_feat = extract_dino_features(
                image_np, dino_model, device
            )

            combined = torch.cat([clip_feat, dino_feat], dim=0).numpy()  # [896,H,W]

            np.save(tmp_path, combined)
            raw = combined

            del clip_feat, dino_feat, combined
            torch.cuda.empty_cache()

        # Sample pixels for PCA fitting
        D, H, W = raw.shape
        flat = raw.reshape(D, -1).T                                # [H*W, D]
        n = min(args.samples_per_image, H * W)
        idx = np.random.choice(H * W, n, replace=False)
        pca_samples.append(flat[idx].copy())   # .copy() for mmap_mode='r'

        image_names.append(name)

        del raw, flat       # free memory immediately

    # ================================================================
    # Fit PCA on sampled pixels
    # ================================================================
    print("\n=== Fitting PCA ===")
    all_samples = np.concatenate(pca_samples, axis=0)   # [N_total, 896]
    del pca_samples
    print(f"  Total samples: {all_samples.shape[0]}")

    D_raw = all_samples.shape[1]
    print(f"  Fitting PCA {D_raw}D -> {args.feature_dim}D ...")
    pca = PCA(n_components=args.feature_dim, whiten=False)
    pca.fit(all_samples)

    projected = pca.transform(all_samples)
    proj_std  = projected.std(axis=0) + 1e-6

    print(f"  Explained variance (sum): {pca.explained_variance_ratio_.sum():.3f}")

    with open(pca_path, "wb") as f:
        pickle.dump({"pca": pca, "proj_std": proj_std}, f)
    print(f"  Saved PCA model -> {pca_path}")

    del all_samples, projected

    # ================================================================
    # Pass 2: Load temp files → apply PCA → save final [32,H,W]
    # ================================================================
    print("\n=== Pass 2 : Applying PCA and saving ===")
    for name in tqdm(image_names, desc="Saving"):
        tmp_path = os.path.join(tmp_dir, f"{name}.npy")
        raw = np.load(tmp_path)                                    # [896, H, W]

        reduced = apply_pca_normalize(raw, pca, proj_std)          # [32, H, W]
        out_path = os.path.join(cache_dir, f"{name}.pt")
        torch.save(torch.tensor(reduced, dtype=torch.float32), out_path)

        # Delete temp file after processing
        os.remove(tmp_path)

    # Clean up temp directory
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"\nDone!  Feature maps -> {cache_dir}")
    print(f"Feature dim : {args.feature_dim}D   |   Images : {len(image_items)}")


if __name__ == "__main__":
    main()
