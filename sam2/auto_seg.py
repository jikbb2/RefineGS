import os
# if using Apple MPS, fall back to CPU for unsupported ops

import re                      # [RefineGS] robust 이름 정규화
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as IMG
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm
import gc
import json
import random as rnd
from pathlib import Path

import uuid
import argparse


device = "cuda"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

IMG_EXTS = {".jpg", ".jpeg", ".png"}   # [RefineGS] 지원 확장자(대소문자 무시)


# ===========================================================================
# [RefineGS] 범용 이름 정규화 (ScanNet 전용 파괴적 rename 대체)
# ===========================================================================
def normalize_image_dir(video_dir, strip_prefix="frame_", target_ext="JPEG",
                         mode="symlink", manifest_path=None):
    """
    원본 이미지 이름 ↔ 정규화 이름 매핑을 만든다.

    mode:
      "symlink" (기본·비파괴): images/ 원본은 그대로 두고, 정규화 이름을
                 images_norm/ 에 심볼릭 링크로 생성. → COLMAP(원본)·SAM(정규화) 공존.
      "rename"  (파괴적·비권장): images/ 안에서 직접 rename (COLMAP desync 주의).
      "none"    : 정규화 없이 원본 이름 그대로 사용 (COLMAP과 자동 일치).

    반환: (work_dir, manifest)  — work_dir = SAM 이 읽을 디렉토리
          manifest = {정규화이름: 원본이름}
    """
    files = sorted(os.listdir(video_dir))

    def norm_name(fn):
        stem, ext = os.path.splitext(fn)
        if strip_prefix:
            stem = re.sub(rf"^{re.escape(strip_prefix)}", "", stem)  # 접두사만, 확장자 안전
        return f"{stem}.{target_ext}"

    manifest = {}
    if mode == "none":
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                manifest[fn] = fn
        work_dir = video_dir
    elif mode == "rename":
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            nn = norm_name(fn)
            manifest[nn] = fn
            if nn != fn and not os.path.exists(os.path.join(video_dir, nn)):
                os.rename(os.path.join(video_dir, fn), os.path.join(video_dir, nn))
        work_dir = video_dir
    else:  # symlink (기본): images/ 안에 정규화 이름 심볼릭 링크 → 원본(.jpg)과 공존
        # mask_propagation.py 가 images/ 만 읽으므로 별도 dir 가 아니라 in-place 로 둔다.
        work_dir = video_dir
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            nn = norm_name(fn)
            manifest[nn] = fn
            if nn == fn:
                continue
            link = os.path.join(video_dir, nn)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(os.path.join(video_dir, fn)), link)

    if manifest_path:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[name_manifest] {len(manifest)} entries -> {manifest_path}")
    return work_dir, manifest


def show_anns(anns, borders=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def show_mask(mask, ax, frame, obj_id=None, random_color=False, out_dir="./output", dataset=""):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    img_to_save = (mask_image[:, :, :3] * 255).astype(np.uint8)
    os.makedirs(f"{out_dir}/{dataset}_autoseg_mask/{frame}", exist_ok=True)
    IMG.fromarray(img_to_save).save(f"{out_dir}/{dataset}_autoseg_mask/{frame}/{obj_id}.png")


def save_mask(masks, prefix):
    segms = {}
    for idx, m in enumerate(masks):
        segms[f"mask_{idx}"] = m["segmentation"].astype(np.uint8)
    np.savez_compressed(f"./data/{prefix}_segmentations_only.npz", **segms)
    return list(segms.values())


def load_mask(prefix, as_numpy=True):
    segms = np.load(f"./data/{prefix}_segmentations_only.npz")
    if as_numpy:
        masks = [segms[key] for key in sorted(segms.files, key=lambda x: int(x.split('_')[1]))]
    else:
        masks = [torch.from_numpy(segms[key]) for key in sorted(segms.files, key=lambda x: int(x.split('_')[1]))]
    return masks


def getMask(image, output_path, configs, preload=False):
    image = np.array(image.convert("RGB"))
    os.makedirs(output_path, exist_ok=True)

    generated_mask = []
    if (not preload):
        for cfg in configs:
            mask_generator = SAM2AutomaticMaskGenerator(**{k: v for k, v in cfg.items() if k != "name"})
            try:
                masks1 = mask_generator.generate(image)
                masks = save_mask(masks1, f"masks_{cfg['name']}")
                generated_mask.append(masks)
            except Exception as e:
                print("NO valid mask!")
                print(e)
                continue
    else:
        for cfg in configs:
            load = load_mask(f"masks_{cfg['name']}")
            generated_mask.append(load)

    final_masks = []
    if (len(generated_mask) == 0):
        return final_masks
    for i in range(len(generated_mask)):
        if (i == 0):
            sorted_masks_0 = sorted(list(generated_mask[i]), key=lambda x: np.sum(x), reverse=True)
            cleaned_mask_0 = useless_mask(sorted_masks_0)
            final_masks = cleaned_mask_0
        else:
            final_masks = MaskMerging(final_masks, generated_mask[i])
    plt.close("all")
    final_masks = useless_mask(final_masks)
    return final_masks


def MaskMerging(merged, mask):
    mask = sorted(mask, key=lambda x: np.sum(x), reverse=True)
    for m1 in mask:
        merged_flag = False
        for i, m2 in enumerate(merged):
            intersection = np.logical_and(m1, m2).sum()
            if intersection == 0:
                continue
            area1 = m1.sum()
            area2 = m2.sum()
            smaller_area = min(area1, area2)
            iou_score = intersection / smaller_area
            if (iou_score > 0.3):
                merged[i] = np.logical_or(m1, m2)
                merged_flag = True
        if not merged_flag:
            merged.append(m1)
    return merged


def overlap_bins(bin_rep, bins, masks):
    masks = useless_mask(masks)
    for m1 in masks:
        placed = False
        for bin_idx, b in enumerate(bins):
            bin_smaller = bin_rep[bin_idx]
            intersection = np.logical_and(m1, bin_smaller).sum()
            if intersection == 0:
                continue
            area1 = m1.sum()
            area2 = bin_smaller.sum()
            smaller_area = min(area1, area2)
            iou_score = intersection / smaller_area if smaller_area > 0 else 0
            if iou_score > 0.5:
                b.append(m1)
                placed = True
                if (area1 < area2):
                    bin_rep[bin_idx] = m1
        if not placed:
            bins.append([m1])
            bin_rep[len(bin_rep.keys())] = m1
    return bins, bin_rep


def majority_voting(bins):
    new_bins = []
    for b in bins:
        if len(b) == 1:
            new_bins.append(b)
            continue
        stack = np.stack(b)
        votes = stack.sum(axis=0)
        threshold = len(b) / 2
        majority_mask = votes > threshold
        agree_bin = []
        disagree_bin = []
        for mask in b:
            intersection = np.logical_and(mask, majority_mask).sum()
            union = np.logical_or(mask, majority_mask).sum()
            iou = intersection / union if union > 0 else 0
            if iou > 0.5:
                agree_bin.append(mask)
            else:
                disagree_bin.append(mask)
        if agree_bin:
            new_bins.append(agree_bin)
        for m in disagree_bin:
            new_bins.append([m])
    return new_bins


def useless_mask(masks, min_ratio=0.0025, iou_threshold=0.95):
    filtered_masks = []
    for m in masks:
        total_pixels = m.size
        mask_pixels = np.count_nonzero(m)
        if mask_pixels / total_pixels < min_ratio:
            continue
        keep = True
        for fm in filtered_masks:
            intersection = np.logical_and(m, fm).sum()
            area1 = m.sum()
            area2 = fm.sum()
            smaller_area = min(area1, area2)
            iou_score = intersection / smaller_area
            if iou_score > iou_threshold:
                keep = False
                break
        if keep:
            filtered_masks.append(m)
    return filtered_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation (generic)")
    parser.add_argument("--scene", type=str, default=None, help="scene to compute the masks on")
    parser.add_argument("--scene_path", type=str, default="./data", help="path to the scene")
    parser.add_argument("--out_dir", type=str, default="./output", help="output base dir")
    # [RefineGS] 범용 이름 정규화 옵션 (ScanNet 전용 가정 제거)
    parser.add_argument("--name_mode", choices=["symlink", "rename", "none"], default="symlink",
                        help="symlink(비파괴·기본) / rename(파괴적) / none(원본 그대로)")
    parser.add_argument("--strip_prefix", type=str, default="",
                        help="이미지 이름 앞에서 제거할 접두사. 기본 '' (제거 안 함). "
                             "mask_propagation.py 가 COLMAP 이름(frame_ 유지)을 기대하므로 비워두는 것이 일관적. "
                             "정말 짧은 이름을 쓰려면 mask_propagation 도 함께 패치 필요.")
    parser.add_argument("--target_ext", type=str, default="JPEG", help="정규화 확장자")
    parser.add_argument("--subsample", type=int, default=1,
                        help="N프레임마다 1개 사용 (이전 ScanNet 하드코딩 [::20] 대체)")
    args = parser.parse_args()

    DATASET = args.scene
    VIDEO_PATH = args.scene_path
    OUT_DIR = args.out_dir
    BASE_DIR = Path(__file__).resolve().parent

    sam2_checkpoint_init = (BASE_DIR / ".." / "checkpoints" / "sam2.1_hiera_large.pt").resolve()
    model_cfg_init = "./configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg_init, sam2_checkpoint_init, device=device, apply_postprocessing=False)
    sam2_post_proc = build_sam2(model_cfg_init, sam2_checkpoint_init, device=device, apply_postprocessing=True)

    configs = [
        {"name": "uber_huge", "model": sam2, "points_per_side": 1, "points_per_batch": 1,
         "pred_iou_thresh": 0.8, "stability_score_thresh": 0.85, "stability_score_offset": 0.85,
         "mask_threshold": 0.5, "crop_n_layers": 1, "box_nms_thresh": 0.3, "use_m2m": False},
        {"name": "very_coarse", "model": sam2, "points_per_side": 4, "points_per_batch": 4,
         "pred_iou_thresh": 0.8, "stability_score_thresh": 0.9, "stability_score_offset": 0.85,
         "mask_threshold": 0.5, "crop_n_layers": 1, "box_nms_thresh": 0.3, "use_m2m": False},
        {"name": "coarse", "model": sam2, "points_per_side": 8, "points_per_batch": 8,
         "pred_iou_thresh": 0.8, "stability_score_thresh": 0.85, "stability_score_offset": 0.85,
         "mask_threshold": 0.3, "crop_n_layers": 1, "box_nms_thresh": 0.5, "use_m2m": False},
        {"name": "fine", "model": sam2, "points_per_side": 16, "points_per_batch": 64,
         "pred_iou_thresh": 0.8, "stability_score_thresh": 0.9, "stability_score_offset": 1,
         "mask_threshold": 0.3, "crop_n_layers": 1, "box_nms_thresh": 0.3, "use_m2m": False},
    ]

    images_dir = os.path.join(VIDEO_PATH, DATASET, "images")

    # [RefineGS] 파괴적 rename 대신 범용 정규화 + 매니페스트
    manifest_path = os.path.join(VIDEO_PATH, DATASET, "name_manifest.json")
    work_dir, manifest = normalize_image_dir(
        images_dir, strip_prefix=args.strip_prefix, target_ext=args.target_ext,
        mode=args.name_mode, manifest_path=manifest_path)
    print(f"Name normalization done (mode={args.name_mode}). SAM reads from: {work_dir}")

    # 정규화 이름(manifest 키)만 순회 — images/ 안에 원본+심볼릭이 공존해도 중복 처리 방지
    best_views = sorted(manifest.keys())

    # [RefineGS] 범용 subsample (이전 '#### SUBSAMPLE SCANET ####  best_views[::20]' 대체)
    if args.subsample > 1:
        best_views = best_views[::args.subsample]

    for i, v in enumerate(tqdm(best_views)):
        frame = os.path.splitext(v)[0]          # 확장자 안전한 stem 추출
        f = IMG.open(os.path.join(work_dir, v))

        f_masks = getMask(f, f"{OUT_DIR}/{DATASET}/{frame}", configs=configs, preload=False)

        plt.figure(figsize=(20, 20))
        plt.imshow(f)
        for out_obj_id, out_mask in enumerate(f_masks):
            show_mask(out_mask, plt.gca(), frame, obj_id=out_obj_id, random_color=True,
                      out_dir=OUT_DIR, dataset=DATASET)
        plt.axis('off')
        plt.close("all")

        del f, f_masks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("auto_seg complete!")
