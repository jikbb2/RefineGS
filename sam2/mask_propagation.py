#
# RefineGS - sam2/mask_propagation.py  (Axis 1: Robust Instance Re-labeling, 재설계)
# ---------------------------------------------------------------------------
# 설계: 3D kNN 후보 그래프 + 멀티뷰 마스크 co-occurrence 가중(거친 마스크 다운웨이트)
#       → Leiden 커뮤니티 분할 → 3D 라벨을 각 뷰로 재투영 + SAM 재프롬프트로 2D 마스크 생성.
#
# 이전 prototype 대비 수정:
#   [통합] 3D→2D 재투영 복구 → output/{scene}_masks/{label}/{frame}.png (Stage 2 필수)
#   [확장] W=B·Bᵀ(N² densify) 제거 → kNN O(Nk) + leidenalg(C백엔드)
#   [과병합] kNN 공간 제약(멀리 떨어진 객체 용접 차단) + 마스크 크기 가중(1/|S_m|^α)
#   [버그] matplotlib Agg, --scene(dest=dataset), 죽은 코드 제거, 벡터화
# ---------------------------------------------------------------------------

import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image as IMG
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion
import open3d as o3d
import pycolmap

import matplotlib
matplotlib.use("Agg")   # [fix] headless

from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sys.path.append("./point_projection")
import point_projection_cuda as ppc


# ----------------------------- device / SAM -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

BASE_DIR = Path(__file__).resolve().parent
_sam_ckpt = (BASE_DIR / ".." / "checkpoints" / "sam2.1_hiera_large.pt").resolve()
_sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(_sam_cfg, str(_sam_ckpt), device=device)
predictor = SAM2ImagePredictor(sam2_model)


# ----------------------------- helpers -----------------------------
def erode_mask(mask, size, a=0.05, b=0.5):
    if size < a:
        return binary_erosion(mask, structure=np.ones((3, 3)), iterations=3).astype(mask.dtype)
    elif size < b:
        return binary_erosion(mask, structure=np.ones((5, 5)), iterations=2).astype(mask.dtype)
    else:
        return binary_erosion(mask, structure=np.ones((5, 5)), iterations=2).astype(mask.dtype)


def mask_to_array(masks):
    if isinstance(masks, list):
        masks = np.array(masks)
    return (np.sum(masks, axis=0) > 0).astype(np.uint8)


def greedy_coreset_2D(S, n, alpha=0.3, exclude_hull=True):
    """경계를 피해 균일 분포 2D prompt 점 n개 선택."""
    if isinstance(S, np.ndarray):
        S = S.tolist()
    if len(S) == 0:
        return None
    C = []
    original_S = S.copy()
    if exclude_hull and len(S) > 3:
        try:
            hull = ConvexHull(S)
            hull_pts = set(tuple(S[i]) for i in hull.vertices)
            S = [s for s in S if tuple(s) not in hull_pts]
            if len(S) == 0:
                S = original_S.copy()
        except Exception:
            S = original_S.copy()
    mean_S = np.mean(S, axis=0)
    x0 = min(S, key=lambda s: np.linalg.norm(np.array(s) - mean_S))
    C.append(x0); S.remove(x0)
    while len(C) < n and S:
        def score(s):
            s = np.array(s)
            d_c = min(np.linalg.norm(s - np.array(c)) for c in C)
            d_m = np.linalg.norm(s - mean_S)
            return d_c - alpha * d_m
        y = max(S, key=score)
        C.append(y); S.remove(y)
    return C


def resolve_image_file(images_dir, stem, exts=(".JPEG", ".jpg", ".jpeg", ".png")):
    for e in exts:
        p = os.path.join(images_dir, stem + e)
        if os.path.exists(p):
            return p
    import glob
    hits = glob.glob(os.path.join(images_dir, stem + ".*"))
    return hits[0] if hits else None


def save_instance_mask(mask, scene, label, stem):
    """output/{scene}_masks/{label}/{stem}.png  (RGBA, alpha=mask — camera_utils 호환)"""
    alpha = (mask.astype(np.uint8)) * 255
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 0:3] = 255
    rgba[..., 3] = alpha
    out_dir = os.path.join("./output", f"{scene}_masks", str(label))
    os.makedirs(out_dir, exist_ok=True)
    IMG.fromarray(rgba, mode="RGBA").save(os.path.join(out_dir, stem + ".png"))


def project_view(pcd_points_t, depth, extrinsics, intrinsics, H, W, epsilon):
    """ppc.pcd2D z-buffer 투영. 반환: (xs, ys) LongTensor, visible_ids LongTensor."""
    t_depth = torch.from_numpy(depth).float().cuda()
    t_points_2D = torch.full((H, W), -1, device="cuda", dtype=torch.float32)
    t_computed_depth = torch.full((H, W), float('inf'), device="cuda", dtype=torch.float32)
    ppc.pcd2D(pcd_points_t, t_depth, extrinsics, intrinsics, epsilon, t_points_2D, t_computed_depth)
    ys, xs = torch.where(t_points_2D != -1)
    visible_ids = t_points_2D[ys, xs].long()
    return xs, ys, visible_ids


# ----------------------------- 그래프 구축 + Leiden -----------------------------
def cluster_instances(pt_to_masks, mask_sizes, pcd_points,
                      k=16, alpha=0.5, resolution=1.0, min_points=15):
    """
    pt_to_masks: dict point_id -> set(global_mask_id)
    mask_sizes : dict global_mask_id -> 가시점 개수 |S_m|
    return: labels dict {label(1..N) -> [point_id]}
    """
    observed = sorted(pt_to_masks.keys())
    n = len(observed)
    if n == 0:
        return {}
    coords = pcd_points[np.asarray(observed)]                  # (n,3) 벡터화 인덱싱

    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(coords)
    _, nbr = nn.kneighbors(coords)                             # (n, kk)

    # kNN 후보 ∩ co-occurrence(>0) → 가중 엣지
    edges, weights, seen = [], [], set()
    pm = [pt_to_masks[pid] for pid in observed]                # local idx -> set
    for a in range(n):
        ma = pm[a]
        for b in nbr[a][1:]:
            key = (a, int(b)) if a < b else (int(b), a)
            if key in seen:
                continue
            seen.add(key)
            shared = ma & pm[int(b)]
            if not shared:
                continue
            w = float(sum(1.0 / (mask_sizes[m] ** alpha) for m in shared))
            if w > 0:
                edges.append(key)
                weights.append(w)

    if not edges:
        print("⚠️ co-occurrence 엣지 0개 — 파라미터 확인 필요")
        return {}

    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = weights
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=42)

    comm = defaultdict(list)
    for local_i, c in enumerate(part.membership):
        comm[c].append(observed[local_i])

    labels, nxt = {}, 1
    for c, pids in sorted(comm.items(), key=lambda kv: -len(kv[1])):
        if len(pids) < min_points:
            continue
        labels[nxt] = pids
        nxt += 1
    return labels


# ----------------------------- main -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RefineGS Axis1 robust instance re-labeling")
    ap.add_argument("--scene", dest="dataset", type=str, required=True, help="scene name")
    ap.add_argument("--data_root", default="./data", type=str)
    ap.add_argument("--epsilon", type=float, default=0.02, help="depth z-buffer 허용오차(m)")
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.5, help="마스크 크기 다운웨이트 지수")
    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden 해상도(클수록 인스턴스↑)")
    ap.add_argument("--min_points", type=int, default=15)
    ap.add_argument("--n_prompt", type=int, default=8, help="SAM 재프롬프트 점 개수")
    ap.add_argument("--min_prompt", type=int, default=4, help="라벨-뷰 마스크 생성 최소 가시점")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    SCENE = args.dataset
    EPS = args.epsilon
    scene_dir = os.path.join(args.data_root, SCENE)
    image_path = os.path.join(scene_dir, "images")
    colmap_model_path = os.path.join(scene_dir, "sparse/0")
    depth_path = os.path.join(scene_dir, "depth")
    autoseg_mask_folder = os.path.join("./output", f"{SCENE}_autoseg_mask")
    ply_path = os.path.join(scene_dir, "sparse/0/points3D.ply")

    recon = pycolmap.Reconstruction(colmap_model_path)
    images = recon.images

    pcd = o3d.io.read_point_cloud(ply_path)
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    pcd_points_t = torch.from_numpy(pcd_points).float().cuda()

    # =========================================================
    # Pass 1: 관측 수집 (마스크별 가시점 집합 + 마스크 크기)
    # =========================================================
    pt_to_masks = defaultdict(set)     # point_id -> {global_mask_id}
    mask_sizes = {}                    # global_mask_id -> |S_m|
    mask_counter = 0

    print("Pass 1: 멀티뷰 관측 수집 ...")
    for image_id in tqdm(list(images)):
        img_meta = images[image_id]
        cam_meta = img_meta.camera
        H, W = cam_meta.height, cam_meta.width
        stem = os.path.splitext(img_meta.name)[0]

        depth_file = os.path.join(depth_path, stem + "_pred.npy")
        if not os.path.exists(depth_file):
            continue
        depth = np.load(depth_file).astype(np.float32)

        mask_dir = os.path.join(autoseg_mask_folder, stem)
        if not os.path.isdir(mask_dir):
            continue
        mask_files = sorted(os.listdir(mask_dir))
        if len(mask_files) == 0:
            continue

        extrinsics = torch.tensor(img_meta.cam_from_world.matrix(), device="cuda", dtype=torch.float32).contiguous()
        intrinsics = torch.tensor(cam_meta.params, device="cuda", dtype=torch.float32).reshape(2, 2)

        xs, ys, visible_ids = project_view(pcd_points_t, depth, extrinsics, intrinsics, H, W, EPS)
        if len(visible_ids) == 0:
            continue
        vis_ids_np = visible_ids.cpu().numpy()

        # 마스크 스택 (eroded)
        mts = []
        for m in mask_files:
            mimg = np.array(IMG.open(os.path.join(mask_dir, m)).convert("L")) > 0
            sz = mimg.sum() / (H * W)
            mts.append(torch.from_numpy(erode_mask(mimg, sz)))
        stacked = torch.stack(mts).to("cuda").bool()           # (M,H,W)

        point_in_masks = stacked[:, ys, xs]                    # (M, n_visible)
        m_idx, p_idx = torch.where(point_in_masks)
        m_idx = m_idx.cpu().numpy(); p_idx = p_idx.cpu().numpy()

        for local_m in range(len(mask_files)):
            sel = p_idx[m_idx == local_m]
            if sel.size == 0:
                continue
            gmid = mask_counter; mask_counter += 1
            mask_sizes[gmid] = int(sel.size)
            for pid in vis_ids_np[sel]:
                pt_to_masks[int(pid)].add(gmid)

    print(f"✅ 관측 완료: 점 {len(pt_to_masks)}개, 마스크 {mask_counter}개")

    # =========================================================
    # 그래프 + Leiden
    # =========================================================
    print("그래프 구축 + Leiden 분할 ...")
    labels = cluster_instances(pt_to_masks, mask_sizes, pcd_points,
                               k=args.knn, alpha=args.alpha,
                               resolution=args.resolution, min_points=args.min_points)
    print(f"🎉 인스턴스 {len(labels)}개")

    if len(labels) == 0:
        sys.exit("인스턴스 0개 — resolution/knn/epsilon 조정 필요")

    # point_id -> label
    pid_to_label = {}
    for l, pids in labels.items():
        for p in pids:
            pid_to_label[p] = l

    # 3D PLY 저장 (init/시각화용)
    for l, pids in labels.items():
        out_dir = os.path.join("./output", f"{SCENE}_masks", str(l))
        os.makedirs(out_dir, exist_ok=True)
        cl = o3d.geometry.PointCloud()
        idx = np.asarray(pids)
        cl.points = o3d.utility.Vector3dVector(pcd_points[idx])
        if pcd_colors is not None:
            cl.colors = o3d.utility.Vector3dVector(pcd_colors[idx])
        o3d.io.write_point_cloud(os.path.join(out_dir, f"label_{l}.ply"), cl)

    # =========================================================
    # Pass 2: 3D 라벨 → 각 뷰 2D 마스크 (SAM 재프롬프트)  [Stage 2 필수]
    # =========================================================
    print("Pass 2: 3D→2D 재투영 + SAM 재프롬프트로 2D 마스크 생성 ...")
    for image_id in tqdm(list(images)):
        img_meta = images[image_id]
        cam_meta = img_meta.camera
        H, W = cam_meta.height, cam_meta.width
        stem = os.path.splitext(img_meta.name)[0]

        depth_file = os.path.join(depth_path, stem + "_pred.npy")
        if not os.path.exists(depth_file):
            continue
        depth = np.load(depth_file).astype(np.float32)

        extrinsics = torch.tensor(img_meta.cam_from_world.matrix(), device="cuda", dtype=torch.float32).contiguous()
        intrinsics = torch.tensor(cam_meta.params, device="cuda", dtype=torch.float32).reshape(2, 2)
        xs, ys, visible_ids = project_view(pcd_points_t, depth, extrinsics, intrinsics, H, W, EPS)
        if len(visible_ids) == 0:
            continue

        pts2d = torch.stack([xs, ys], dim=1).cpu().numpy()
        vis_ids_np = visible_ids.cpu().numpy()

        # 라벨별 2D 가시점 모으기
        label_to_2d = defaultdict(list)
        for kk2, pid in enumerate(vis_ids_np):
            l = pid_to_label.get(int(pid))
            if l is not None:
                label_to_2d[l].append(pts2d[kk2])
        if not label_to_2d:
            continue

        img_file = resolve_image_file(image_path, stem)
        if img_file is None:
            continue
        img_rgb = np.array(IMG.open(img_file).convert("RGB"))
        predictor.set_image(img_rgb)   # 뷰당 1회 임베딩

        for l, p2d in label_to_2d.items():
            if len(p2d) < args.min_prompt:
                continue
            prompt = greedy_coreset_2D(np.array(p2d), args.n_prompt)
            if prompt is None:
                continue
            prompt = np.array(prompt)
            m, _, _ = predictor.predict(
                point_coords=prompt, point_labels=np.ones(len(prompt)),
                multimask_output=False)
            save_instance_mask(mask_to_array(m), SCENE, l, stem)

    print("✅ 완료: output/{scene}_masks/{label}/ 에 2D 마스크 PNG + label.ply 저장")
    print("   다음: mv output/{scene}_masks data/{scene}/masks → prepare_folder.sh")
