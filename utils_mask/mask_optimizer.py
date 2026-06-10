#
# RefineGS - utils_mask/mask_optimizer.py  (Stage 3.2.2 mask reprojection)
# ---------------------------------------------------------------------------
# ScanNet 전용 가정 제거 → 범용(LERF/임의 COLMAP) 버전.
#
# 핵심 변경 [RefineGS / generic]:
#   - SCENE 모듈 전역 하드코딩 제거 → --scene 인자
#   - 중복 './data/scanNet/...' 경로 블록 제거 → --data_root 단일 스킴
#   - ScanNet txt 포즈/intrinsic_depth.txt 의존 제거 → **COLMAP cam 에서 직접 유도**
#       (원하면 --pose_dir / --intrinsics_depth 로 ScanNet-style txt 사용: backward-compat)
#   - depth scale / 확장자 / iteration 인자화
#   - 이름 해석을 name_manifest.json + stem 글롭으로 robust 화
#   - render() 호출을 RefineGS 머지 시그니처로 수정 (use_trained_exp/separate_sh 제거),
#       2DGS 누적 alpha/mask 를 렌더 실루엣으로 사용
#
# ⚠️ 이 파일은 deferred 3.2.2 경로(point_projection CUDA + SAM 재정제)에 속함.
#    구문/로직만 검증됨 — 서버(CUDA/COLMAP/SAM2)에서 동작 테스트 필요.
# ---------------------------------------------------------------------------

import sys
sys.path.append('.')

import os
import glob
import json
import argparse
import torch
import open3d as o3d
import numpy as np
import sam2

import cv2
from PIL import Image as IMG
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from scipy.spatial import ConvexHull

from arguments import ModelParams, PipelineParams, ArgumentParser, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal
from scene import Scene, GaussianModel
from gaussian_renderer import render

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("./point_projection")
import point_projection_cuda as ppc


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

ACCURACY_LABELS = 0.7
DOWNSAMPLE = 100000
EPSILON = 0.01   # 기본값 (CLI --epsilon 로 덮어씀; depth 스케일에 맞게 조정)


# =========================== helpers (원본 유지) ===========================
def random_downsample(points, target_num):
    idx = np.random.choice(points.shape[0], size=target_num, replace=False)
    return points[idx]


def filter_points(ply, black_th=-1.75, alpha_th=4.5):
    vertices = ply['vertex']
    f_dc = np.stack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']], axis=1)
    is_black = np.all(f_dc < black_th, axis=1)
    keep_mask = ~is_black
    filtered_array = vertices[keep_mask]
    opacity = filtered_array['opacity']
    is_transparent = opacity > alpha_th
    keep_mask = ~is_transparent
    filtered_array = filtered_array[keep_mask]
    xyz = np.vstack([filtered_array['x'], filtered_array['y'], filtered_array['z']]).T
    return xyz


def greedy_coreset_2D(S, n, alpha=0.3, exclude_hull=True):
    if isinstance(S, np.ndarray):
        S = S.tolist()
    C = []
    original_S = S.copy()
    if exclude_hull and len(S) > 3:
        try:
            hull = ConvexHull(S)
        except Exception:
            return None
        hull_points = set(tuple(S[i]) for i in hull.vertices)
        S = [s for s in S if tuple(s) not in hull_points]
        if len(S) == 0:
            S = original_S.copy()
    mean_S = np.mean(S, axis=0)
    x0 = min(S, key=lambda s: np.linalg.norm(np.array(s) - mean_S))
    C.append(x0)
    S.remove(x0)
    while len(C) < n and S:
        def score(s):
            s = np.array(s)
            dist_to_C = min(np.linalg.norm(s - np.array(c)) for c in C)
            dist_from_mean = np.linalg.norm(s - mean_S)
            return dist_to_C - alpha * dist_from_mean
        y = max(S, key=score)
        C.append(y)
        S.remove(y)
    return C


def compute_IoU(mask_1, mask_2):
    mask_1 = mask_1.astype(bool)
    mask_2 = mask_2.astype(bool)
    intersection = np.logical_and(mask_1, mask_2).sum()
    union = np.logical_or(mask_1, mask_2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def mask_to_array(masks):
    if isinstance(masks, list):
        masks = np.array(masks)
    mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)
    return mask


def load_depth(path, scale=1000.0):
    """depth 이미지 로드 후 scale 로 나눔 (ScanNet mm→m: 1000 / 이미 metric: 1.0)."""
    return np.array(IMG.open(path)) / scale


def load_matrix_from_txt(path):
    with open(path) as f:
        vals = [float(v) for v in f.read().split()]
    return np.array(vals).reshape(4, 4)


# =========================== [RefineGS] generic geometry ===========================
def extrinsics_3x4_from_cam(cam):
    """COLMAP cam → world→camera 3x4 row-major (12,) CUDA tensor.
    cam.world_view_transform = getWorld2View2(R,T).transpose(0,1) 이므로 다시 transpose 해서
    수학 convention W2C 를 복원한 뒤 [:3,:] 사용."""
    W2C = cam.world_view_transform.transpose(0, 1)              # (4,4) world->cam
    return W2C[:3, :].contiguous().reshape(-1).to(torch.float32)  # (12,)


def intrinsics_from_cam(cam):
    """COLMAP cam → [[fx, fy, cx, cy]] (principal point는 중심 가정)."""
    W, H = cam.image_width, cam.image_height
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx, cy = W / 2.0, H / 2.0
    return torch.tensor([[fx, fy, cx, cy]], device="cuda", dtype=torch.float32)


def resolve_path(folder, stem, exts):
    """folder 안에서 stem.<ext> 를 exts 순서로 탐색, 첫 매치 반환 (없으면 None)."""
    for e in exts:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(folder, stem + ".*"))
    return hits[0] if hits else None


def get_render_mask(camera, gaussians, pipe):
    """RefineGS 머지 render() 호출. 2DGS: id 2-pass mask 또는 누적 alpha 를 실루엣으로."""
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pkg = render(camera, gaussians, pipe, bg)              # [fix] use_trained_exp/separate_sh 제거
    if pkg.get("mask", None) is not None:
        return pkg["mask"]
    return pkg["rend_alpha"]


if __name__ == "__main__":
    parser = ArgumentParser(description="Mask reprojection/refinement (generic)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--scene", required=True, type=str, help="scene name")
    parser.add_argument("--instance_test", default="0", type=str, help="instance ID")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ply_iteration", default=10000, type=int, help="per-instance PLY iteration")
    # [RefineGS] 범용 경로/스케일/네이밍 옵션
    parser.add_argument("--data_root", default="./data", type=str, help="데이터 루트 (ScanNet 의 ./data/scanNet 하드코딩 제거)")
    parser.add_argument("--output_root", default="./output", type=str)
    parser.add_argument("--depth_dir", default="depth", type=str)
    parser.add_argument("--depth_ext", default=".png", type=str)
    parser.add_argument("--depth_scale", default=1000.0, type=float, help="ScanNet mm→m=1000, metric=1.0")
    parser.add_argument("--epsilon", default=EPSILON, type=float, help="depth-projection 허용오차 (스케일에 맞게)")
    parser.add_argument("--img_exts", default=".JPEG,.jpg,.jpeg,.png", type=str)
    # ScanNet-style txt 포즈를 쓰고 싶을 때만 (기본은 COLMAP cam 에서 유도)
    parser.add_argument("--pose_dir", default="", type=str, help="비우면 COLMAP cam 사용; 채우면 txt 포즈 사용")
    parser.add_argument("--intrinsics_depth", default="", type=str, help="ScanNet intrinsic_depth.txt (pose_dir 사용 시)")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    pipeline = pipeline.extract(args)
    model = model.extract(args)
    iteration = args.iteration
    safe_state(args.quiet)

    SCENE = args.scene
    EPS = args.epsilon
    IMG_EXTS = [e if e.startswith(".") else "." + e for e in args.img_exts.split(",")]
    DEPTH_EXTS = [args.depth_ext, ".png", ".npy"]

    args.is_instance = False
    gaussians = GaussianModel(model.sh_degree)
    scene = Scene(model, gaussians, load_iteration=iteration, shuffle=False)
    id_color = scene.get_id()
    scene = scene.filter_gaussian()
    cameras = scene.getTrainCameras()

    INSTANCE_TEST = args.instance_test
    print(f"Processing Instance ID: {INSTANCE_TEST}")

    # ----- [RefineGS] 단일 경로 스킴 (ScanNet 중복 블록 제거) -----
    SCENE_DIR = os.path.join(args.data_root, SCENE)
    IMAGE_PATH = os.path.join(SCENE_DIR, "images")
    DEPTH_PATH = os.path.join(SCENE_DIR, args.depth_dir)
    INSTANCE_PATH = os.path.join(SCENE_DIR, SCENE + "_masks", INSTANCE_TEST)
    ply_path = os.path.join(args.output_root, SCENE, "raw", INSTANCE_TEST,
                            f"point_cloud/iteration_{args.ply_iteration}/point_cloud.ply")

    # name manifest (auto_seg 의 정규화 매핑) — 있으면 사용
    manifest_path = os.path.join(SCENE_DIR, "name_manifest.json")
    name_manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            name_manifest = json.load(f)   # {정규화이름: 원본이름}

    # ScanNet-style txt 포즈 모드 (선택)
    use_txt_pose = bool(args.pose_dir)
    if use_txt_pose:
        intr = load_matrix_from_txt(args.intrinsics_depth)
        intr_np_fixed = torch.tensor([[intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]]],
                                     device="cuda", dtype=torch.float32)

    # SAM
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # GS per-instance 재구성 로드
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd_points = np.asarray(pcd.points)
    if pcd_points.shape[0] > DOWNSAMPLE:
        pcd_points = random_downsample(pcd_points, DOWNSAMPLE)

    for cam in tqdm(cameras):
        img_name = getattr(cam, "image_name", None)
        if img_name is None:
            continue
        stem = os.path.splitext(img_name)[0]   # 확장자 안전 stem

        # 이미지 파일 해석 (manifest 우선 → stem 글롭)
        img_file = None
        if name_manifest:
            # manifest 는 {정규화: 원본}. cam.image_name 이 원본이면 정규화 키를 역참조
            for norm, orig in name_manifest.items():
                if os.path.splitext(orig)[0] == stem or os.path.splitext(norm)[0] == stem:
                    img_file = resolve_path(IMAGE_PATH, os.path.splitext(norm)[0], IMG_EXTS) \
                               or resolve_path(IMAGE_PATH, os.path.splitext(orig)[0], IMG_EXTS)
                    break
        if img_file is None:
            img_file = resolve_path(IMAGE_PATH, stem, IMG_EXTS)
        if img_file is None:
            continue

        img = IMG.open(img_file).convert("RGB")
        W, H = img.size
        img = np.array(img)

        # depth 해석
        depth_file = resolve_path(DEPTH_PATH, stem, DEPTH_EXTS)
        if depth_file is None:
            continue
        if depth_file.endswith(".npy"):
            depth = np.load(depth_file).astype(np.float32)
        else:
            depth = load_depth(depth_file, scale=args.depth_scale).astype(np.float32)

        # instance mask 해석
        mask = None
        mask_file = resolve_path(os.path.join(INSTANCE_PATH, "masks"), stem, [".png", ".jpg", ".JPEG"])
        if mask_file is not None:
            try:
                mask_img = IMG.open(mask_file).convert("RGBA")
                alpha = np.array(mask_img)[:, :, 3]
                mask = alpha > 0
            except Exception:
                mask = None

        # ----- [RefineGS] 카메라 기하: COLMAP cam 유도(기본) 또는 txt 포즈 -----
        if use_txt_pose:
            extr_path = os.path.join(args.pose_dir, stem + ".txt")
            if not os.path.exists(extr_path):
                continue
            extr = load_matrix_from_txt(extr_path)        # camera→world
            extr_inv = np.linalg.inv(extr)[:3, :].astype(np.float32).reshape(12)  # world→cam
            extrinsics = torch.tensor(extr_inv, device="cuda", dtype=torch.float32).contiguous()
            intrinsics = intr_np_fixed
        else:
            extrinsics = extrinsics_3x4_from_cam(cam).contiguous()
            intrinsics = intrinsics_from_cam(cam)

        # point projection (CUDA z-buffer) — backbone-무관
        t_pcd_points = torch.from_numpy(pcd_points).float().cuda()
        t_depth = torch.from_numpy(depth).float().cuda()
        t_points_2D = torch.full((H, W), -1, device="cuda", dtype=torch.float32)
        t_computed_depth = torch.full((H, W), float('inf'), device="cuda", dtype=torch.float32)
        ppc.pcd2D(t_pcd_points, t_depth, extrinsics, intrinsics, EPS, t_points_2D, t_computed_depth)

        ys, xs = torch.where(t_points_2D != -1)
        if len(xs) == 0:
            continue
        visible_point2D = torch.stack([xs, ys], dim=1).detach().cpu().numpy()

        p = greedy_coreset_2D(visible_point2D, 5)
        if p is None:
            continue
        input_point = np.array(p)
        input_label = np.ones(len(p))

        predictor.set_image(img)
        mask_original, scores, _ = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=False)
        new_mask = (mask_to_array(mask_original) > 0).astype(np.uint8)

        # 렌더 실루엣 (2DGS mask/alpha)
        rendered = get_render_mask(cam, gaussians, pipeline)
        if rendered.ndim == 3 and rendered.shape[0] in (3, 4):
            gray = rendered.mean(dim=0)
        else:
            gray = rendered.squeeze()
        rendered_mask = (gray > 0.2).to(torch.uint8).detach().cpu().numpy()

        iou_new = compute_IoU(rendered_mask, new_mask)
        if mask is not None:
            iou_old = compute_IoU(rendered_mask, mask)
            if iou_new < iou_old:
                continue
        if iou_new < 0.05:
            continue

        # 정제 마스크 저장 (id_color RGBA)
        alpha = (new_mask.astype(np.uint8)) * 255
        try:
            kernel = np.ones((3, 3), dtype=np.uint8)
            alpha = cv2.dilate(alpha, kernel, iterations=1)
        except Exception as e:
            print(f"Warning: dilation failed ({e})")
        rgba = np.zeros((new_mask.shape[0], new_mask.shape[1], 4), dtype=np.uint8)
        rgba[..., 0:3] = 255
        rgba[..., 3] = alpha
        os.makedirs(os.path.join(INSTANCE_PATH, "mask_extra"), exist_ok=True)
        IMG.fromarray(rgba, mode="RGBA").save(os.path.join(INSTANCE_PATH, "mask_extra", stem + ".png"))

    print("mask_optimizer complete!")
