#!/usr/bin/env python3
"""RefineGS — depth 렌더 기반 SDF distillation (TSDF 대체, whole scene).

render.py의 TSDF fusion과 '동일한 입력·옵션'을 쓰되, Open3D TSDF 대신 implicit SDF(IGR류)로
watertight 메쉬를 뽑는다. 파이프라인:

  1) render()로 뷰별 surf_depth + rend_normal + rend_alpha + render(rgb) 를 얻음
  2) to_cam_open3d와 동일한 intrinsic/extrinsic으로 depth를 월드 점군으로 back-project
     → 법선은 카메라 방향으로 자동 정렬(부호 일관) : 기존 sdf_distill.py 스펀지의 근본원인 제거
  3) 정렬된 oriented point cloud에 IGR SDF(MLP) 피팅 (manifold+normal+eikonal+signed off-surface)
  4) 그리드 SDF 평가 → '관측된 복셀만' 마스킹(미관측 빈 공간의 박스 제거, 작은 구멍은 보간 채움)
     → marching cubes(zero level set)
  5) safe_post_process_mesh 로 num_cluster 후처리(기존 TSDF 경로와 동일 로직 + 클램프)

RefineGS repo 루트(render.py 옆)에 두고 실행:

  python sdf_distill_depth.py -m output/replica_room0_v2/scene_whole_orbit -s data/replica_room0_v2 \
    --iteration 7000 --depth_ratio 0 --depth_trunc 6.0 --voxel_size 0.01 \
    --sdf_trunc 0.04 --num_cluster 10000 \
    --sdf_iters 10000 --pts_per_view 40000

  # 위 render.py TSDF 명령과 동일 옵션 매핑:
  #   --depth_ratio, --depth_trunc  : render()/back-project에 그대로 사용
  #   --voxel_size                  : marching cubes 그리드 해상도 산출(2*scale/voxel_size)
  #   --sdf_trunc                   : (참고) — SDF 경로에선 미사용. 마스킹은 --mask_dist로 제어
  #   --num_cluster                 : safe_post_process_mesh 로 재사용(클러스터 수 클램프)
"""
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
import open3d as o3d


# ---------------------------------------------------------------------------
# 카메라 intrinsic/extrinsic — to_cam_open3d(mesh_utils.py)와 완전히 동일한 규약
# ---------------------------------------------------------------------------
def cam_intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    if hasattr(cam, "projection_matrix"):
        ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2],
                                [0, H / 2, 0, (H - 1) / 2],
                                [0, 0, 0, 1]]).float().cuda().T
        intrins = (cam.projection_matrix @ ndc2pix)[:3, :3].T
        fx, fy = intrins[0, 0].item(), intrins[1, 1].item()
        cx, cy = intrins[0, 2].item(), intrins[1, 2].item()
    else:  # MiniCam (extra_poses) — FoV에서 직접 산출
        fx = W / (2.0 * np.tan(cam.FoVx / 2))
        fy = H / (2.0 * np.tan(cam.FoVy / 2))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    extrinsic = cam.world_view_transform.T  # world->camera (w2c), CV 규약(+Z forward)
    return fx, fy, cx, cy, W, H, extrinsic


# ---------------------------------------------------------------------------
# utils.mesh_utils.post_process_mesh 의 안전 버전.
# 원본은 sorted[-cluster_to_keep] 인덱싱이라 연결성분 수 < num_cluster 이면
# IndexError 발생(깨끗한 SDF 메쉬에서 실제로 터짐). 클러스터 수로 클램프한다.
# ---------------------------------------------------------------------------
def safe_post_process_mesh(mesh, cluster_to_keep=1000):
    print(f"post processing the mesh to have {cluster_to_keep} clusters (clamped)")
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    keep = min(cluster_to_keep, len(cluster_n_triangles))  # ← 클램프 (원본 버그 수정)
    n_cluster = np.sort(cluster_n_triangles.copy())[-keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


# ---------------------------------------------------------------------------
# IGR-style SDF MLP (geometric init). PE는 기본 off — off-surface 진동/스펀지 방지.
# ---------------------------------------------------------------------------
class SDFNet(nn.Module):
    def __init__(self, d_hidden=256, n_layers=8, skip_in=(4,), pe_L=0, radius=0.5):
        super().__init__()
        self.pe_L = pe_L
        d_in = 3 + 3 * 2 * pe_L
        self.d_in = d_in
        dims = [d_in] + [d_hidden] * n_layers + [1]
        self.skip_in = set(skip_in)
        self.num_layers = len(dims)
        self.layers = nn.ModuleList()
        for l in range(self.num_layers - 1):
            out_dim = dims[l + 1] - d_in if (l + 1) in self.skip_in else dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if l == self.num_layers - 2:
                nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=1e-4)
                nn.init.constant_(lin.bias, -radius)
            else:
                nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                nn.init.constant_(lin.bias, 0.0)
            self.layers.append(lin)
        self.act = nn.Softplus(beta=100)

    def pe(self, x):
        if self.pe_L == 0:
            return x
        out = [x]
        for l in range(self.pe_L):
            for fn in (torch.sin, torch.cos):
                out.append(fn(2.0 ** l * np.pi * x))
        return torch.cat(out, -1)

    def forward(self, x):
        inp = self.pe(x)
        h = inp
        for l, lin in enumerate(self.layers):
            if l in self.skip_in:
                h = torch.cat([h, inp], -1) / np.sqrt(2)
            h = lin(h)
            if l < self.num_layers - 2:
                h = self.act(h)
        return h


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def load_carve_points(carve_dir, center, scale, n_max=2000000, margin=0.02,
                      px_per_view=20000, samples_per_ray=4):
    """전체 씬 depth 덤프(dump_scene_depth.py)에서 free-space 샘플 생성.
    각 픽셀 광선의 (카메라 → depth-margin) 구간에서 샘플 → 객체 bbox(정규화 |x|<1.2) 안만 유지."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(os.path.expanduser(carve_dir), "*.npz")))
    assert files, f"carve depth 없음: {carve_dir}"
    pts = []
    for f in files:
        z = np.load(f)
        depth = z["depth"].astype(np.float32)
        fx, fy, cx, cy = float(z["fx"]), float(z["fy"]), float(z["cx"]), float(z["cy"])
        c2w = z["c2w"].astype(np.float32)
        vs, us = np.nonzero(depth > 0)
        if len(vs) == 0:
            continue
        sel = np.random.choice(len(vs), min(px_per_view, len(vs)), replace=False)
        v, u = vs[sel], us[sel]
        d = depth[v, u]
        dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, np.float32)], -1) @ c2w[:3, :3].T
        dnorm = np.linalg.norm(dirs, axis=-1)
        dn = dirs / dnorm[:, None]
        on = (c2w[:3, 3] - center) / scale                       # 정규화 좌표 카메라 중심
        # 광선-구(반경 1.2) 교차 구간(chord)에서만 샘플 → 전 샘플이 bbox 내부·표면 앞
        b = (on[None] * dn).sum(-1)
        disc = b * b - ((on * on).sum() - 1.44)
        hit = disc > 0
        if not hit.any():
            continue
        sq = np.sqrt(disc[hit])
        t_in = np.maximum(-b[hit] - sq, 0.02)
        t_out = -b[hit] + sq
        tmax_n = (d[hit] * dnorm[hit] - margin) / scale          # depth까지(단위방향·정규화), margin 여유
        hi = np.minimum(t_out, tmax_n)
        ok = hi > t_in
        if not ok.any():
            continue
        t_in, hi, dh = t_in[ok], hi[ok], dn[hit][ok]
        for _ in range(samples_per_ray):
            t = t_in + np.random.rand(len(t_in)).astype(np.float32) * (hi - t_in)
            pts.append(on[None] + dh * t[:, None])
    if not pts:
        return np.zeros((0, 3))
    P = np.concatenate(pts)
    if len(P) > n_max:
        P = P[np.random.choice(len(P), n_max, replace=False)]
    return P


# ---------------------------------------------------------------------------
_C0 = 0.28209479177387814


def _set_label_color(gaussians, label):
    """gaussian 색을 라벨 값으로 교체(label-buffer 렌더용). render_hole_novel과 동일 트릭."""
    fdc, frest = gaussians._features_dc, gaussians._features_rest
    saved = (fdc.detach().clone(), frest.detach().clone(), int(gaussians.active_sh_degree))
    dc = (label - 0.5) / _C0
    with torch.no_grad():
        fdc[:, 0, 0] = dc; fdc[:, 0, 1] = dc; fdc[:, 0, 2] = dc
        frest.zero_()
    gaussians.active_sh_degree = 0
    return saved


def _restore_color(gaussians, saved):
    fdc_s, frest_s, sh = saved
    with torch.no_grad():
        gaussians._features_dc.copy_(fdc_s)
        gaussians._features_rest.copy_(frest_s)
    gaussians.active_sh_degree = sh


@torch.no_grad()
def render_extra_masks(extra_cams, gaussians, pipe, background, label, thr=0.3):
    """extra 포즈에서 객체 라벨을 렌더해 per-view 객체 마스크 생성 (base/바닥 유입 차단)."""
    from gaussian_renderer import render as _render
    saved = _set_label_color(gaussians, label)
    masks = {}
    for cam in extra_cams:
        lab = _render(cam, gaussians, pipe, background)["render"][0].clamp(0, 1)
        masks[cam.image_name] = lab > thr
    _restore_color(gaussians, saved)
    return masks


_mask_info_printed = False


def load_view_mask(mask_dir, image_name, H, W):
    """뷰별 객체 마스크 로드(없으면 None). 값 규약 자동 판별:
    - 0/1 바이너리      → >0 이 객체
    - amodal(188/0/255) → 188(visible)만 객체 (255=bg, 0=occluded 제외)
    - 0/255 바이너리    → >127 이 객체
    """
    global _mask_info_printed
    from PIL import Image
    stem = os.path.splitext(image_name)[0]
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            img = Image.open(p).resize((W, H), Image.NEAREST)
            a = np.array(img)
            if a.ndim == 3 and a.shape[2] == 4:
                a = a[..., 3]          # RGBA: 객체 마스크는 알파 채널 (RGB는 인스턴스 색 코드)
            elif a.ndim == 3:
                a = np.array(img.convert("L"))
            if a.max() <= 1:
                mm = a > 0
            elif (a == 188).any():
                mm = a == 188          # amodal 규약: 188=visible
            else:
                mm = a > 127
            if not _mask_info_printed:
                u, c = np.unique(a, return_counts=True)
                print(f"마스크 값 분포(첫 뷰 {os.path.basename(p)}, 채널 처리 후): "
                      f"{dict(zip(u.tolist()[:6], c.tolist()[:6]))} → 객체 픽셀 {int(mm.sum())}")
                _mask_info_printed = True
            return torch.from_numpy(mm).cuda()
    return None


def load_gt_depth(depth_dir, image_name, H, W, scale):
    """GT(데이터셋) depth 로드 — '씬 전체 기하' 기준 free-space carve 용.
    렌더 depth(다리 없는 가우시안)와 달리 실제 occlusion 을 반영 → 다리 절단 방지.
    stem 매칭: frame000918 → depth000918.png / frame000918.png / frame000918.npy"""
    from PIL import Image
    stem = os.path.splitext(image_name)[0]
    for c in (stem.replace("frame", "depth"), stem, stem + "_depth"):
        for ext in (".png", ".npy"):
            p = os.path.join(os.path.expanduser(depth_dir), c + ext)
            if not os.path.exists(p):
                continue
            if ext == ".npy":
                a = np.load(p).astype(np.float32)
            else:
                a = np.array(Image.open(p)).astype(np.float32) / scale
            if a.shape != (H, W):
                a = np.array(Image.fromarray(a).resize((W, H), Image.NEAREST))
            return a
    return None


@torch.no_grad()
def collect_oriented_points(scene, gaussians, pipe, background, args, mask_dir=None,
                            require_mask=False, extra_cams=None, extra_masks=None):
    """뷰별 depth를 월드 점군으로 back-project. 법선은 카메라 방향으로 정렬.
    mask_dir: 객체 마스크 밖 픽셀 제외(TSDF 경로와 동일 철학).
    require_mask: 마스크 없는 학습 뷰는 통째로 skip (composed 200뷰 모델 + 객체 마스크 8뷰 케이스).
    extra_cams: 추가 novel 카메라(MiniCam) — 마스크 미적용(ROI crop으로 제한 권장).
                See3D 정제된 unseen은 orbit 포즈에서만 보이므로 추출에 필수."""
    # [속도] 학습 뷰가 수천 장이면 back-project 가 지배적 비용이 된다. 관측 점군은
    # --pts_per_view/--n_pts 로 어차피 서브샘플되므로 뷰를 성기게 써도 손실이 작다.
    _stride = max(1, int(getattr(args, "view_stride", 1)))
    views = [(c, True) for c in scene.getTrainCameras()[::_stride]]
    if _stride > 1:
        print(f"[속도] view_stride={_stride} → 학습뷰 {len(views)}장 사용")
    if extra_cams:
        views += [(c, False) for c in extra_cams]
    P_all, N_all, C_all, O_all = [], [], [], []
    EO_all, ED_all = [], []
    VB = []          # [prior carve] 뷰 버퍼(다운스케일 depth+mask) — prior 샘플 visual-hull/freespace 검증용
    n_tr = sum(1 for _, u in views if u)
    n_carve_views = getattr(args, "prior_carve_views", 0)
    keep_every = max(1, n_tr // max(n_carve_views, 1)) if n_carve_views > 0 else 0
    ti = 0
    n_masked_views = 0
    n_skipped = 0
    for cam, use_mask in views:
        m_obj = None
        pkg = render(cam, gaussians, pipe, background)
        depth = pkg["surf_depth"][0]                      # [H,W]
        alpha = pkg["rend_alpha"][0]                      # [H,W]
        rgb = pkg["render"].permute(1, 2, 0)              # [H,W,3]
        nrm = torch.nn.functional.normalize(pkg["rend_normal"], dim=0).permute(1, 2, 0)  # [H,W,3] world

        fx, fy, cx, cy, W, H, extrinsic = cam_intrinsics(cam)
        c2w = torch.inverse(extrinsic)                    # camera->world
        cam_center = c2w[:3, 3]

        vv, uu = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32),
                                torch.arange(W, device="cuda", dtype=torch.float32),
                                indexing="ij")
        x = (uu - cx) * depth / fx
        y = (vv - cy) * depth / fy
        pts_cam = torch.stack([x, y, depth], -1)          # [H,W,3]
        pts_w = pts_cam @ c2w[:3, :3].T + cam_center      # [H,W,3] world

        valid = (depth > 0) & (depth < args.depth_trunc) & (alpha > args.alpha_thr)
        if mask_dir is not None and use_mask:
            m = load_view_mask(mask_dir, cam.image_name, H, W)
            if m is not None:
                valid &= m
                m_obj = m
                n_masked_views += 1
            elif require_mask:
                n_skipped += 1
                continue          # 마스크 없는 학습 뷰 제외 (씬 전체 점 유입 방지)

        # [prior carve] 균등 간격 뷰의 depth+mask 버퍼 저장
        if use_mask and keep_every and m_obj is not None:
            ti += 1
            if ti % keep_every == 0:
                ds = max(1, int(getattr(args, "prior_carve_ds", 2)))
                dbuf = torch.where(alpha > args.alpha_thr, depth,
                                   torch.zeros_like(depth))[::ds, ::ds].cpu().numpy()
                mbuf = m_obj[::ds, ::ds].cpu().numpy()
                w2c = extrinsic.cpu().numpy()
                b = {"R": w2c[:3, :3], "t": w2c[:3, 3],
                     "fx": fx / ds, "fy": fy / ds, "cx": cx / ds, "cy": cy / ds,
                     "W": dbuf.shape[1], "H": dbuf.shape[0],
                     "depth": dbuf, "mask": mbuf}
                if getattr(args, "gt_depth_dir", ""):
                    dg = load_gt_depth(args.gt_depth_dir, cam.image_name, H, W,
                                       args.gt_depth_scale)
                    if dg is not None:
                        b["dgt"] = dg[::ds, ::ds]
                VB.append(b)
        if not use_mask and extra_masks is not None and cam.image_name in extra_masks:
            valid &= extra_masks[cam.image_name]   # extra 포즈: label-buffer 객체 마스크
        pts_w = pts_w[valid]
        n = nrm[valid]
        c = rgb[valid].clamp(0, 1)

        # 법선을 카메라 쪽으로 정렬(부호 일관) — SDF sign이 전역적으로 정의됨
        view_dir = cam_center[None] - pts_w
        flip = (n * view_dir).sum(-1) < 0
        n[flip] = -n[flip]
        n = torch.nn.functional.normalize(n, dim=-1)

        if len(pts_w) > args.pts_per_view:
            sel = torch.randperm(len(pts_w), device="cuda")[:args.pts_per_view]
            pts_w, n, c = pts_w[sel], n[sel], c[sel]

        P_all.append(pts_w.cpu()); N_all.append(n.cpu()); C_all.append(c.cpu())
        O_all.append(cam_center[None].expand(len(pts_w), 3).cpu())  # 점별 관측 카메라 중심

        # 빈 광선 수집: alpha≈0 픽셀 = "이 광선 위엔 아무것도 없음"이 관측된 것
        # [FIX Step4-C] 실제 학습 뷰(use_mask=True)에서만 수집. extra/novel 뷰의 alpha≈0 은
        # "모델에 기하가 없음"이지 "빈 공간이 관측됨"이 아님 — 생성 다리를 carve로 지우는 원인.
        if args.empty_per_view > 0 and use_mask:
            em = alpha < args.empty_alpha
            eidx = em.nonzero(as_tuple=False)
            if len(eidx) > 0:
                sel_e = eidx[torch.randperm(len(eidx), device="cuda")[:args.empty_per_view]]
                ve, ue = sel_e[:, 0].float(), sel_e[:, 1].float()
                de = torch.stack([(ue - cx) / fx, (ve - cy) / fy, torch.ones_like(ue)], -1)
                de = torch.nn.functional.normalize(de @ c2w[:3, :3].T, dim=-1)
                EO_all.append(cam_center[None].expand(len(de), 3).cpu())
                ED_all.append(de.cpu())

    if mask_dir is not None:
        print(f"객체 마스크 적용: {n_masked_views}/{len(views)} 뷰, skip {n_skipped}뷰 (경로 {mask_dir})")
    if extra_cams:
        print(f"extra 포즈 렌더: {len(extra_cams)}뷰 (마스크 미적용)")
    P = torch.cat(P_all).numpy().astype(np.float64)
    N = torch.cat(N_all).numpy().astype(np.float64)
    C = torch.cat(C_all).numpy().astype(np.float64)
    O = torch.cat(O_all).numpy().astype(np.float64)
    EO = torch.cat(EO_all).numpy().astype(np.float64) if EO_all else np.zeros((0, 3))
    ED = torch.cat(ED_all).numpy().astype(np.float64) if ED_all else np.zeros((0, 3))
    print(f"빈 광선(empty ray) {len(EO)}개 수집, prior-carve 뷰 버퍼 {len(VB)}개")
    return P, N, C, O, EO, ED, VB


def train_sdf(P, N, O, EO, ED, args, CV=None, W=None, OBS=None, PV=None, PS=None):
    """정규화된 oriented point cloud에 IGR SDF 피팅.
    O = 점별 관측 카메라 중심(정규화 좌표) — 표면 근처 free-space carving.
    EO/ED = 빈 광선(alpha≈0 픽셀)의 카메라 중심/방향 — empty-ray carving.
    CV = 전체 씬 depth 기반 free-space 샘플 풀(carve_depth_dir) — 있으면 empty-ray 대신 사용.
    OBS = [FIX Step4-B] 점별 '관측점' 마스크(bool). l_free 는 관측점에서만 —
          prior 점의 가짜 원점(O=center)이 객체 내부를 carve하던 버그 제거.
    PV/PS = [Step2] prior mesh 볼륨 샘플 좌표 / target SDF(정규화). truncated L1 회귀 —
            얇은 다리 양쪽 빈 공간이 양수로 명시 감독되어 팽창을 원리적으로 차단."""
    dev = "cuda"
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    Nt = torch.tensor(N, dtype=torch.float32, device=dev)
    Ot = torch.tensor(O, dtype=torch.float32, device=dev)
    Wt = torch.tensor(W, dtype=torch.float32, device=dev) if W is not None else None
    EOt = torch.tensor(EO, dtype=torch.float32, device=dev) if EO is not None and len(EO) else None
    EDt = torch.tensor(ED, dtype=torch.float32, device=dev) if ED is not None and len(ED) else None
    CVt = torch.tensor(CV, dtype=torch.float32, device=dev) if CV is not None and len(CV) else None
    obs_idx = None
    if OBS is not None:
        obs_idx = torch.tensor(np.nonzero(OBS)[0], dtype=torch.long, device=dev)
        if len(obs_idx) == len(Pt):
            obs_idx = None                       # 전부 관측점이면 게이팅 불필요
    PVt = torch.tensor(PV, dtype=torch.float32, device=dev) if PV is not None and len(PV) else None
    PSt = torch.tensor(PS, dtype=torch.float32, device=dev) if PVt is not None else None
    net = SDFNet(pe_L=args.pe_L).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    delta = args.offsurf_delta
    for it in range(args.sdf_iters):
        bi = torch.randint(0, len(Pt), (args.batch,), device=dev)
        pts = Pt[bi].clone().requires_grad_(True)
        nrm = Nt[bi]
        sdf = net(pts)
        g = grad(sdf, pts)
        if Wt is not None:
            w = Wt[bi]; wsum = w.sum().clamp(min=1e-8)
            l_man = (sdf.abs().squeeze(-1) * w).sum() / wsum
            l_nrm = ((1 - torch.nn.functional.cosine_similarity(g, nrm, dim=-1)) * w).sum() / wsum
            pp = Pt[bi] + delta * nrm; pm = Pt[bi] - delta * nrm
            l_sign = (((net(pp) - delta).abs().squeeze(-1) * w).sum()
                      + ((net(pm) + delta).abs().squeeze(-1) * w).sum()) / wsum
        else:
            l_man = sdf.abs().mean()
            l_nrm = (1 - torch.nn.functional.cosine_similarity(g, nrm, dim=-1)).mean()
            pp = Pt[bi] + delta * nrm; pm = Pt[bi] - delta * nrm
            l_sign = (net(pp) - delta).abs().mean() + (net(pm) + delta).abs().mean()

        # free-space carving (표면 근처): 관측점 p에서 카메라 방향으로 s∈[2δ, free_range] 후퇴한
        # 점은 빈 공간 → SDF ≥ 0. (bbox 안 표면 근처를 집중 샘플 — 멀리 카메라 쪽은 정보 없음)
        # [FIX Step4-B] 관측점만 사용(obs_idx). prior 점은 O=center 가짜 원점이라
        # 객체 내부를 '빈 공간'으로 carve → l_sign 과 충돌 → 표면 뒤틀림의 원인이었음.
        l_free = torch.tensor(0.0, device=dev)
        if args.w_free > 0:
            bf = (obs_idx[torch.randint(0, len(obs_idx), (args.batch,), device=dev)]
                  if obs_idx is not None else bi)
            dirv = Pt[bf] - Ot[bf]
            dist = dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dirn = dirv / dist
            s = torch.rand(len(bf), 1, device=dev) * (args.free_range - 2 * delta) + 2 * delta
            s = torch.minimum(s, dist * 0.95)
            xf = Pt[bf] - dirn * s
            l_free = torch.relu(-net(xf)).mean()

        # empty-ray carving: 렌더 alpha≈0 픽셀의 광선은 '아무것도 없음'이 관측된 것 →
        # 광선이 반경 1.2 구(=bbox) 를 지나는 chord 구간 안에서만 샘플해 SDF ≥ 0 강제.
        l_empty = torch.tensor(0.0, device=dev)
        if args.w_empty > 0 and CVt is not None:
            bj = torch.randint(0, len(CVt), (args.batch,), device=dev)
            l_empty = torch.relu(-net(CVt[bj])).mean()
        elif args.w_empty > 0 and EOt is not None and len(EOt) > 0:
            bj = torch.randint(0, len(EOt), (args.batch,), device=dev)
            o, dn = EOt[bj], EDt[bj]
            t0 = -(o * dn).sum(-1, keepdim=True)                 # 원점 최근접 파라미터
            cp = o + dn * t0
            half = (1.44 - (cp * cp).sum(-1, keepdim=True)).clamp(min=0.0).sqrt()
            t = (t0 + (torch.rand_like(t0) * 2 - 1) * half).clamp(min=0.05)
            xe = o + dn * t
            l_empty = torch.relu(-net(xe)).mean()                         # 음수(내부)만 벌점

        # [Step2] prior mesh 볼륨 SDF distillation — truncated L1 회귀.
        # 표면점만 주입하던 기존 방식과 달리 다리 '양쪽' 빈 공간이 명시적으로 양수 감독됨.
        l_prior = torch.tensor(0.0, device=dev)
        if PVt is not None and args.w_prior_sdf > 0:
            bp = torch.randint(0, len(PVt), (args.batch,), device=dev)
            l_prior = (net(PVt[bp]).squeeze(-1) - PSt[bp]).abs().mean()

        # eikonal: 표면 근처 + 균등 랜덤
        rp = torch.cat([Pt[torch.randint(0, len(Pt), (args.batch,), device=dev)]
                        + 0.02 * torch.randn(args.batch, 3, device=dev),
                        torch.rand(args.batch, 3, device=dev) * 2 - 1], 0).requires_grad_(True)
        ge = grad(net(rp), rp)
        l_eik = ((ge.norm(dim=-1) - 1) ** 2).mean()

        loss = (l_man + args.w_normal * l_nrm + args.w_sign * l_sign
                + args.w_eik * l_eik + args.w_free * l_free + args.w_empty * l_empty
                + args.w_prior_sdf * l_prior)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"[{it}] man {l_man.item():.4f} nrm {l_nrm.item():.4f} "
                  f"sign {l_sign.item():.4f} eik {l_eik.item():.4f} "
                  f"free {l_free.item():.4f} empty {l_empty.item():.4f} "
                  f"prior {l_prior.item():.4f}")
    return net


def grid_fuse_tsdf(VB, sd_fn, center, scale, args, debug_pts=None):
    """[grid-fuse] MLP 없는 결정적 SDF 융합 — 모순을 '평균'이 아닌 '우선순위'로 해소:
        관측 TSDF > 관측된 빈 공간(occlusion-aware carve, +trunc) > 생성 SDF(미관측만).
    MLP 보간 병리(부풀림·스펀지)가 구조적으로 없음. 품질은 정합·생성 품질에만 의존.
    occlusion-aware: 관측 표면 '뒤'(다리 영역 등)는 판단 보류 → 생성이 채움."""
    from skimage.measure import marching_cubes
    from scipy import ndimage
    trunc = args.prior_trunc
    margin = args.prior_carve_margin
    G = args.grid if args.grid > 0 else int(round(2 * scale / args.voxel_size))
    G = int(min(G, args.max_grid))
    print(f"[grid-fuse] G={G} voxel≈{2*scale/(G-1):.4f}m trunc={trunc}m views={len(VB)}")
    n_gt = sum(1 for b in VB if "dgt" in b)
    print(f"[grid-fuse] GT depth 버퍼 {n_gt}/{len(VB)}뷰"
          + ("" if n_gt else "  ⚠ GT depth 없음 — (구)실루엣 carve 사용, 다리 절단 위험"))
    # [gt-check] GT depth ↔ 렌더 depth 정합 검증 — 값이 크면(수 cm↑) 스케일/프레임
    # 매칭 오류이며 free 판정 전체가 무효. 0 에 가까워야 정상.
    for b in VB[:5]:
        dg = b.get("dgt")
        if dg is not None:
            mm = b["mask"] & (b["depth"] > 0) & (dg > 0.01)
            if mm.sum() > 100:
                d = (dg - b["depth"])[mm]
                print(f"[gt-check] median(dgt-render)={np.median(d):+.4f}m  "
                      f"|d|중앙값={np.median(np.abs(d)):.4f}m  (마스크 {int(mm.sum())}px)")
    # [경계 번짐 가드] GT depth 불연속(실루엣 경계) 픽셀은 free 투표 무효 —
    # nearest 리사이즈로 먼 배경 depth 가 경계에 새어들어 얇은 구조를 갉는 것 방지.
    for b in VB:
        dg = b.get("dgt")
        if dg is not None and "dgt_ok" not in b:
            gy_, gx_ = np.gradient(dg.astype(np.float32))
            edge = (np.abs(gx_) + np.abs(gy_)) > args.gt_edge_thr
            edge = ndimage.binary_dilation(edge, iterations=1)
            b["dgt_ok"] = (dg > 0.01) & ~edge
    lin = np.linspace(-1, 1, G, dtype=np.float32)
    Fo = np.zeros((G, G, G), np.float32)      # 관측 TSDF 가중합
    Wo = np.zeros((G, G, G), np.float32)      # 관측 가중치(뷰 수)
    FRc = np.zeros((G, G, G), np.uint16)      # 관측된 빈 공간 '투표 수'(합의제)
    VOb = np.zeros((G, G, G), np.uint16)      # 씬 표면 근방 'obj6' 투표
    VOt = np.zeros((G, G, G), np.uint16)      # 씬 표면 근방 '타 객체' 투표
    NFR = np.zeros((G, G, G), np.uint16)      # [visual hull] 시야에 든 뷰 수
    NIN = np.zeros((G, G, G), np.uint16)      # [visual hull] 객체 마스크 안으로 투영된 뷰 수
    SG = np.empty((G, G, G), np.float32)      # 생성 SDF (truncated)
    for k0 in range(0, G, 8):
        k1 = min(k0 + 8, G)
        gx, gy, gz = np.meshgrid(lin, lin, lin[k0:k1], indexing="ij")
        Xw = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float64) * scale + center
        SG[:, :, k0:k1] = np.clip(sd_fn(Xw), -trunc, trunc).reshape(G, G, k1 - k0)
        f = np.zeros(len(Xw), np.float32); w = np.zeros(len(Xw), np.float32)
        fr = np.zeros(len(Xw), np.uint16)
        vo = np.zeros(len(Xw), np.uint16); vt = np.zeros(len(Xw), np.uint16)
        nfr = np.zeros(len(Xw), np.uint16); nin = np.zeros(len(Xw), np.uint16)
        for b in VB:
            Xc = Xw @ b["R"].T + b["t"]; z = Xc[:, 2]; zz = np.maximum(z, 1e-6)
            u = b["fx"] * Xc[:, 0] / zz + b["cx"]
            v = b["fy"] * Xc[:, 1] / zz + b["cy"]
            infr = (z > 0.05) & (u >= 0) & (u < b["W"]) & (v >= 0) & (v < b["H"])
            ui = np.clip(u, 0, b["W"] - 1).astype(int)
            vi = np.clip(v, 0, b["H"] - 1).astype(int)
            di = b["depth"][vi, ui]; mi = b["mask"][vi, ui]
            # [visual hull] 마스크 원뿔의 교집합 — 가림 판정 불필요.
            # 바닥/인접 객체 위 복셀은 대부분의 뷰에서 객체 마스크 밖으로 투영되어 배제되고,
            # 객체 뒷면(미관측)은 마스크 안이라 보존된다.
            nfr += infr.astype(np.uint16)
            nin += (infr & mi).astype(np.uint16)
            sdf = di - z                                   # 양수 = 관측 표면 앞(밖)
            hit = infr & mi & (di > 0) & (sdf > -trunc)    # 표면 뒤 trunc 초과는 미적용(occlusion)
            f[hit] += np.clip(sdf[hit], -trunc, trunc); w[hit] += 1
            dg = b.get("dgt")
            if dg is not None:
                # [GT-depth carve] 마스크 무관, 실제 씬 기하 기준:
                #   z < d_gt - margin → 관측된 빈 공간 '투표'(합의제, 경계 픽셀 제외)
                #   씬 표면 근방      → obj6/타 객체 소속 투표(마스크 노이즈에 강건)
                dgt = dg[vi, ui]
                vgt = infr & b["dgt_ok"][vi, ui]
                fr += (vgt & (z < dgt - margin)).astype(np.uint16)
                near = vgt & (np.abs(z - dgt) < 2 * margin)
                vo += (near & mi).astype(np.uint16)
                vt += (near & ~mi).astype(np.uint16)
            else:
                # (구) 실루엣 carve — GT depth 없을 때만. 렌더 depth 는 다리가 없어
                # occlusion 판단이 불완전하므로 다리 절단 위험 있음.
                fr += (infr & (~mi) & ((di <= 0) | (z < di - margin))).astype(np.uint16)
        sh = (G, G, k1 - k0)
        Fo[:, :, k0:k1] += f.reshape(sh); Wo[:, :, k0:k1] += w.reshape(sh)
        FRc[:, :, k0:k1] += fr.reshape(sh)
        VOb[:, :, k0:k1] += vo.reshape(sh); VOt[:, :, k0:k1] += vt.reshape(sh)
        NFR[:, :, k0:k1] += nfr.reshape(sh); NIN[:, :, k0:k1] += nin.reshape(sh)
        if (k0 // 8) % 8 == 0:
            print(f"  [grid-fuse] slab {k0}/{G}")
    Fobs = Fo / np.maximum(Wo, 1e-6)
    alpha = np.clip(Wo / args.grid_wcap, 0, 1)             # 관측 신뢰도(뷰 수 기반)
    # [seam] 관측/생성 전이를 부드럽게 — alpha 를 흐리면 경계에 blend band 가 생겨
    # '끊긴 듯한' 계단이 사라진다(F 를 나중에 흐리는 것과 달리 단차를 만들지 않음).
    if getattr(args, "alpha_smooth", 0) > 0:
        alpha = ndimage.gaussian_filter(alpha, sigma=args.alpha_smooth)
    FREE = FRc >= args.free_min_views                      # 합의제: N뷰 이상이 '빈 공간' 투표
    OTH = VOt > VOb                                        # 타 객체 표면(과반 투표) → obj6 밖
    step = 2.0 / (G - 1)

    # [sign-fix] non-watertight 생성 mesh(multi-material glb 등)는 winding-number 부호가
    # 깨져 내부가 음수가 아님 → 표면은 분류상 keep 인데 marching cubes 에 안 나오는 원인.
    # |SG| 를 unsigned 로 보고, 표면 셸을 경계로 그리드 외곽 flood-fill = 밖, 나머지 = 안.
    def _fix_sign(SGv):
        vox = 2 * scale / (G - 1)
        UD = np.abs(SGv)
        # (1) flood-fill 부호: 닫힌 부위의 내부 복원 (셸에 구멍 있으면 누수 가능)
        shell = UD <= 1.5 * vox
        openv = ~shell
        lab, _ = ndimage.label(openv)
        bl = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(),
                                       lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        outside = np.isin(lab, bl[bl > 0]) & openv
        inside = openv & ~outside
        flood = np.where(inside, -UD, UD)
        # (2) '적응형' 오프셋 셸 UD-δ(x): 제로 두께 시트도 두께 2δ 볼륨이 되되,
        #     δ 는 관측 표면 거리로 조절 — 관측 근처(테이블 테두리)는 δ_min 으로
        #     부풀음·이중표면 방지, 깊은 미관측(다리)은 δ_max 로 도톰하게.
        dmin = max(1.5 * vox, args.shell_delta_min)
        dmax = max(dmin, args.shell_delta)
        Dobs = ndimage.distance_transform_edt(~(alpha > 0.25)).astype(np.float32) * vox
        dmap = np.clip(dmin + (dmax - dmin) * (Dobs / max(args.shell_ramp, 1e-6)),
                       dmin, dmax)
        out = np.minimum(flood, UD - dmap).astype(np.float32)    # 볼륨 합집합
        print(f"[sign-fix] flood 내부 {inside.mean()*100:.2f}%  최종 SG<0 {(out < 0).mean()*100:.2f}%  "
              f"(수정 전 {(SGv < 0).mean()*100:.2f}%, δ {dmin*1000:.0f}→{dmax*1000:.0f}mm "
              f"ramp {args.shell_ramp}m)")
        return out

    need_sign_fix = getattr(args, "grid_sign_fix", False) or not getattr(args, "prior_watertight", True)
    if need_sign_fix:
        SG = _fix_sign(SG)

    # [carve-align] 정합 보정: 생성이 관측된 free 공간을 피해 unknown(미관측 그림자,
    # 실제 unseen 기하가 존재할 수 있는 유일한 영역) 속으로 들어가도록 9-DoF 재최적화.
    # 앵커: 관측 지배 영역에 붙은 생성 표면(상판)은 관측 TSDF zero-set 에 유지.
    if getattr(args, "carve_align", False) and debug_pts is not None:
        from scipy.optimize import minimize as _pmin
        from scipy.spatial.transform import Rotation as _Rot
        freef = np.minimum(FRc.astype(np.float32) / max(args.free_min_views, 1), 1.0)
        pn_all = ((debug_pts - center) / scale).astype(np.float64)
        sel = np.random.default_rng(1).choice(len(pn_all), min(20000, len(pn_all)), replace=False)
        pn = pn_all[sel]

        def _interp(vol, q):
            idx = np.clip(np.round((q + 1) / step), 0, G - 1).astype(int)
            return vol[idx[:, 0], idx[:, 1], idx[:, 2]]

        anch = _interp(alpha, pn) > 0.5
        c0 = pn.mean(0)

        def _unpack(x):
            return _Rot.from_rotvec(x[:3]).as_matrix(), x[3:6], np.exp(x[6:9])

        def _loss(x):
            R_, t_, s_ = _unpack(x)
            q = ((pn - c0) * s_) @ R_.T + c0 + t_
            L_free = _interp(freef, q).mean()                       # free 점유 벌점
            if anch.any():
                qa = q[anch]
                ai = _interp(alpha, qa) > 0.25
                fo = np.abs(_interp(Fobs, qa))
                L_anch = float(np.where(ai, fo, trunc).mean()) / trunc   # 관측 이탈 벌점
            else:
                L_anch = 0.0
            return L_free + args.carve_align_w * L_anch + 0.05 * float(np.abs(x).sum())

        x0 = np.zeros(9); l0 = _loss(x0)
        res = _pmin(_loss, x0, method="Powell", options={"maxiter": 250, "xtol": 1e-4})
        R_, t_, s_ = _unpack(res.x)
        print(f"[carve-align] loss {l0:.4f}→{res.fun:.4f}  dt={np.round(res.x[3:6]*scale, 3)}m  "
              f"scale={np.round(s_, 3)}  rot={np.rad2deg(np.linalg.norm(res.x[:3])):.1f}°")
        sm = float(s_.mean())
        for k0 in range(0, G, 8):                          # 보정 반영해 생성 SDF 재계산
            k1 = min(k0 + 8, G)
            gx, gy, gz = np.meshgrid(lin, lin, lin[k0:k1], indexing="ij")
            Xn = np.stack([gx, gy, gz], -1).reshape(-1, 3).astype(np.float64)
            q = ((Xn - c0 - res.x[3:6]) @ R_) / s_ + c0
            SG[:, :, k0:k1] = np.clip(sd_fn(q * scale + center) * sm,
                                      -trunc, trunc).reshape(G, G, k1 - k0)
        if need_sign_fix:
            SG = _fix_sign(SG)                             # 재계산된 SG 도 부호 복원
        debug_pts = (((pn_all - c0) * s_) @ R_.T + c0 + res.x[3:6]) * scale + center

    # [visual hull] prior 를 객체 자신의 마스크 원뿔 교집합 안으로 제한.
    # carve 는 '관측 표면보다 앞'만 지우므로, 생성 기하가 바닥이나 인접 객체 '속'으로
    # 파고들면 살아남아 seen accuracy 가 폭발한다(배치 실측: obj20 2.1→107mm).
    # hull 은 가림 판정 없이 이를 차단한다.
    HULL = np.ones((G, G, G), bool)
    if args.hull_min_frac > 0:
        HULL = (NIN >= args.hull_min_frac * np.maximum(NFR, 1)) & (NFR >= args.hull_min_views)
        print(f"[hull] 마스크 원뿔 통과 {HULL.mean()*100:.1f}%  "
              f"(frac≥{args.hull_min_frac}, 최소 {args.hull_min_views}뷰)  "
              f"→ 생성 표면 중 hull 밖 {(np.abs(SG) < 2*2*scale/(G-1))[~HULL].mean()*100 if (~HULL).any() else 0:.1f}% 제거")

    # [적용 게이트] 미관측이 거의 없는 객체에는 prior 가 얻을 게 없고 잃기만 한다
    # (배치 실측: baseline unseen completion 15mm 대 객체들에서 unseen F@2cm 이 0.68→0.21 로 붕괴).
    # 판정은 GT 없이 — '생성 **표면** 중 unknown 에 놓인 비율'.
    # ※ 내부 부피로 재면 안 된다: 물체 내부는 어떤 경우에도 미관측이라 완전 관측 객체도
    #   높게 나온다(구 검증: 완전 관측 67% vs 절반 미관측 91% — 변별력 없음).
    vox_g = 2 * scale / (G - 1)
    prior_surf = np.abs(SG) < 1.5 * vox_g
    unknown = (alpha < 0.25) & ~FREE & ~OTH
    ufrac = float((prior_surf & unknown).sum()) / max(int(prior_surf.sum()), 1)
    print(f"[gate] 생성 표면 중 unknown 비율 {ufrac*100:.1f}% "
          f"(임계 {args.min_unknown_frac*100:.0f}%)")
    if ufrac < args.min_unknown_frac:
        print("  → 미관측이 충분치 않음: prior 미적용(관측만으로 재구성)")
        SG = np.full_like(SG, trunc)

    # 빈공간/타객체/hull 밖 = +trunc, 미관측 ∩ hull = 생성
    base = np.where(FREE | OTH | ~HULL, trunc, SG)
    F = alpha * Fobs + (1 - alpha) * base                  # 우선순위 블렌드

    # [opening] 미관측 영역의 '뾰족한 돌출' 제거. 소파 뒷면처럼 어떤 카메라도 관통해
    # 보지 못한 곳은 carve 제약이 없어 생성 스파이크가 그대로 남는다. erosion→dilation
    # 은 구조요소보다 얇은 돌기만 지우고 굵은 몸통·다리는 보존한다(관측 영역은 미적용).
    # 반경은 '미터'로 지정 — 셸 팽창(2δ) 때문에 실제 돌기 두께 T 는 T+2δ 로 부푼다.
    # opening 반경 r 은 (T/2 + δ) 보다 커야 지워지므로 δ 를 자동 반영한다.
    if getattr(args, "unseen_open", 0) > 0:
        vox = 2 * scale / (G - 1)
        k = max(1, int(round(args.unseen_open / vox)))
        neg = F < 0
        st = ndimage.generate_binary_structure(3, 1)
        op = ndimage.binary_dilation(
            ndimage.binary_erosion(neg, st, iterations=k), st, iterations=k)
        rm = neg & ~op & (alpha < 0.5)
        F = np.where(rm, trunc, F)
        print(f"[opening] 미관측 뾰족 구조 제거 {int(rm.sum())}복셀 "
              f"(r={args.unseen_open*1000:.0f}mm → k={k}복셀, "
              f"두께 {2*args.unseen_open*1000:.0f}mm 이하 돌기)")

    if args.grid_smooth > 0:
        F = ndimage.gaussian_filter(F, sigma=args.grid_smooth)

    # [keep-connected] 최종 음수 볼륨 중 '이 객체의 관측 복셀과 연결된' 성분만 유지.
    # 통짜 생성(여러 객체 포함) prior 가 타 객체의 가려진 공간(unknown)에 남기는
    # 잔해를 구조적으로 차단 — 다리는 상판/하판을 통해 관측부와 연결되므로 보존.
    if getattr(args, "keep_connected", False):
        neg = F < 0
        lab, ncomp = ndimage.label(neg)
        seeds = np.unique(lab[neg & (alpha > 0.5)])
        seeds = seeds[seeds > 0]
        keepm = np.isin(lab, seeds)
        removed = int(neg.sum() - keepm.sum())
        F = np.where(neg & ~keepm, trunc, F)
        print(f"[keep-connected] 음수성분 {ncomp}개 → 관측 연결 {len(seeds)}개 유지, "
              f"{removed}복셀({removed/max(neg.sum(),1)*100:.1f}%) 제거")

    # [probe] 지정 박스(world 좌표) 안의 복셀 분류 통계 — "다리가 왜 없는가"를 국소 계측.
    # 사용: --probe_box "x0,y0,z0,x1,y1,z1" (사라진 다리 주변, 뷰어에서 좌표 읽기)
    if getattr(args, "probe_box", ""):
        try:
            v = [float(x) for x in args.probe_box.split(",")]
            assert len(v) == 6
        except (ValueError, AssertionError):
            print(f"[probe] ⚠ 잘못된 형식: '{args.probe_box}' — 숫자 6개 필요 "
                  f'(예: --probe_box "1.2,-0.5,0.0,1.5,-0.2,0.6"). probe 생략.')
            v = None
    else:
        v = None
    if v is not None:
        lo_n = (((np.array(v[:3]) - center) / scale) + 1) / step
        hi_n = (((np.array(v[3:]) - center) / scale) + 1) / step
        i0, j0, k0p = np.clip(np.floor(np.minimum(lo_n, hi_n)), 0, G - 1).astype(int)
        i1, j1, k1p = np.clip(np.ceil(np.maximum(lo_n, hi_n)), 0, G - 1).astype(int)
        sub = np.s_[i0:i1 + 1, j0:j1 + 1, k0p:k1p + 1]
        nvox = FREE[sub].size
        print(f"[probe] {v} ({nvox}복셀): FREE {FREE[sub].mean()*100:.0f}%  "
              f"FRc>0 {(FRc[sub] > 0).mean()*100:.0f}%  obs {(alpha[sub] > 0.5).mean()*100:.0f}%  "
              f"OTH {OTH[sub].mean()*100:.0f}%  SG<0 {(SG[sub] < 0).mean()*100:.0f}%  "
              f"최종 F<0 {(F[sub] < 0).mean()*100:.0f}%")
    print(f"[grid-fuse] 관측복셀 {(Wo > 0).mean()*100:.1f}%  "
          f"free {(FREE & (Wo == 0)).mean()*100:.1f}% (합의 {args.free_min_views}뷰, "
          f"1뷰라도 {((FRc > 0) & (Wo == 0)).mean()*100:.1f}%)  타객체 {OTH.mean()*100:.2f}%  "
          f"생성 내부(미관측) {((SG < 0) & (Wo == 0) & ~FREE & ~OTH).mean()*100:.2f}%")
    step = 2.0 / (G - 1)
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(step,) * 3)
    verts = (verts - 1.0) * scale + center

    # [debug] 생성 표면 샘플의 분류 시각화 — 다리가 왜 잘리는지 눈으로 판별:
    # 초록=unknown(생성 유지), 파랑=관측 지배, 노랑=타객체, 빨강=free-carve
    if getattr(args, "debug_class_ply", ""):
        # --prior_field 경로에는 메쉬가 없으므로 필드의 영교차 복셀에서 직접 점을 만든다
        if debug_pts is None:
            # 필드의 zero-level 을 marching cubes 로 직접 뽑는다.
            # (|SG|<임계 방식은 필드 스케일에 민감해, 값이 작은 필드에서 그리드 전체가
            #  선택되어 통짜 큐브가 나오는 문제가 있었다)
            if SG.min() < 0 < SG.max():
                dv, _, _, _ = marching_cubes(SG, level=0.0, spacing=(step,) * 3)
                debug_pts = (dv - 1.0) * scale + center
                if len(debug_pts) > 300000:
                    debug_pts = debug_pts[np.random.choice(len(debug_pts), 300000,
                                                           replace=False)]
                print(f"[debug] prior zero-level 에서 점군 {len(debug_pts)}개 "
                      f"(SG 범위 [{SG.min():.4f}, {SG.max():.4f}])")
            else:
                print(f"  ⚠ [debug] prior 필드에 영교차가 없음 "
                      f"(SG 범위 [{SG.min():.4f}, {SG.max():.4f}]) — "
                      f"필드 생성 실패 또는 게이트로 비활성화됨. 점군 생략")
                debug_pts = np.zeros((0, 3))
    if getattr(args, "debug_class_ply", "") and len(debug_pts):
        idx = np.clip(np.round(((debug_pts - center) / scale + 1) / step), 0, G - 1).astype(int)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        cls = np.zeros(len(debug_pts), int)
        cls[~HULL[i, j, k]] = 4                      # hull 밖(마스크 원뿔 위반)
        cls[FREE[i, j, k]] = 3
        cls[OTH[i, j, k] & ~FREE[i, j, k]] = 2
        cls[alpha[i, j, k] > 0.5] = 1
        pal = np.array([[0.1, 0.8, 0.1], [0.2, 0.4, 1.0], [1.0, 0.8, 0.1],
                        [1.0, 0.15, 0.15], [0.7, 0.1, 0.9]])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(debug_pts)
        pc.colors = o3d.utility.Vector3dVector(pal[cls])
        dp = os.path.expanduser(args.debug_class_ply)
        os.makedirs(os.path.dirname(dp) or ".", exist_ok=True)
        o3d.io.write_point_cloud(dp, pc)
        print(f"[debug] 분류 점군: {dp}  keep {(cls == 0).mean()*100:.0f}%  "
              f"obs {(cls == 1).mean()*100:.0f}%  oth {(cls == 2).mean()*100:.0f}%  "
              f"free {(cls == 3).mean()*100:.0f}%  hull밖 {(cls == 4).mean()*100:.0f}%  "
              f"(초록=유지, 파랑=관측, 노랑=타객체, 빨강=carve, 보라=hull밖)")
    return verts, faces


def main():
    parser = ArgumentParser(description="Depth-based SDF distillation (TSDF replacement)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # render.py TSDF 옵션 매핑
    parser.add_argument("--depth_trunc", default=6.0, type=float, help="최대 depth (back-project cutoff)")
    parser.add_argument("--voxel_size", default=0.01, type=float, help="marching cubes 복셀 크기(그리드 산출)")
    parser.add_argument("--sdf_trunc", default=0.04, type=float, help="(참고, SDF 경로 미사용)")
    parser.add_argument("--num_cluster", default=10000, type=int, help="post-process 유지 클러스터 수(클램프됨)")
    # SDF 관련
    parser.add_argument("--alpha_thr", default=0.5, type=float, help="이하 alpha 픽셀 제거(배경/floater)")
    parser.add_argument("--pts_per_view", default=40000, type=int)
    parser.add_argument("--n_pts", default=1500000, type=int, help="학습용 표면점 상한(subsample)")
    parser.add_argument("--sdf_iters", default=10000, type=int)
    parser.add_argument("--batch", default=16384, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--pe_L", default=0, type=int, help="positional encoding 레벨(0=off, 권장)")
    parser.add_argument("--w_normal", default=1.0, type=float)
    parser.add_argument("--w_sign", default=1.0, type=float)
    parser.add_argument("--w_eik", default=0.5, type=float)
    parser.add_argument("--w_free", default=1.0, type=float,
                        help="표면 근처 free-space carving 가중치(0=off)")
    parser.add_argument("--free_range", default=0.5, type=float,
                        help="관측점에서 카메라 쪽으로 carve하는 최대 거리(정규화 좌표)")
    parser.add_argument("--w_empty", default=1.0, type=float,
                        help="empty-ray carving 가중치(0=off). alpha≈0 광선의 bbox 통과 구간 SDF≥0 — 진짜 구멍 보존/부풀림 제거")
    parser.add_argument("--empty_per_view", default=4096, type=int)
    parser.add_argument("--empty_alpha", default=0.1, type=float,
                        help="이 alpha 미만 픽셀을 빈 광선으로 간주. junk가 alpha를 깔면 0.3~0.5로 완화")
    parser.add_argument("--tile", default=0, type=int,
                        help="타일 marching 블록 크기(예: 128). 0=단일 볼륨. whole-scene 고해상도에 필수")
    parser.add_argument("--extra_points", default="", type=str,
                        help="생성 뷰 점군 ply(법선 포함) — unseen 표면 증거 주입 (make_gen_points.py)")
    parser.add_argument("--prior_repeat", default=1.0, type=float,
                        help="<1 이면 prior 점을 그 비율로 서브샘플(신뢰도 하향)")
    parser.add_argument("--prior_weight", default=1.0, type=float,
                        help="extra_points(prior) 표면 손실 가중치(<1: seen 이 지배하는 soft prior)")
    parser.add_argument("--prior_field", default="", type=str,
                        help="[권장] ShapeR 부호 SDF 그리드 npz(shaper_field.py 출력). "
                             "메쉬를 거치지 않아 sign-fix/shell_delta/정합이 모두 불필요")
    parser.add_argument("--prior_field_rescale", default=1, type=int,
                        help="prior 필드의 포화값을 --prior_trunc 에 맞춰 정규화(1=on). "
                             "carve 경계 급점프로 생기는 가짜 표면 방지. 영교차는 불변")
    parser.add_argument("--prior_sigma_w", default=1.0, type=float,
                        help="앙상블 σ 가중 강도(0=off). 클수록 시드가 갈리는 곳의 생성을 "
                             "적극 억제 — precision↑ recall↓")
    parser.add_argument("--prior_sigma_ref", default=0.0, type=float,
                        help="σ 기준값(m). 0=영교차 근방 σ 중앙값으로 자동")
    parser.add_argument("--prior_mesh", default="", type=str,
                        help="[Step2] 정합된 watertight 생성 메쉬(fuse_generated_mesh --save_aligned "
                             "출력 *_gen_aligned.ply). 표면점+볼륨 SDF distillation — extra_points 상위호환")
    parser.add_argument("--w_prior_sdf", default=0.5, type=float,
                        help="볼륨 SDF distillation 손실 가중치(0=off)")
    parser.add_argument("--prior_surf_n", default=150000, type=int,
                        help="prior mesh 표면 샘플 수")
    parser.add_argument("--prior_band", default=0.08, type=float,
                        help="볼륨 셸 샘플 half-width (world m). 다리 최소 두께의 2~3배 권장")
    parser.add_argument("--prior_trunc", default=0.05, type=float,
                        help="target SDF truncation (world m)")
    parser.add_argument("--prior_unseen_dist", default=0.03, type=float,
                        help="[Step1] 관측점에서 이 거리(m) 이내의 prior 샘플 제거 — 관측이 항상 승리")
    parser.add_argument("--prior_gate", default=0.02, type=float,
                        help="[Step4] carve/empty 샘플이 prior mesh 표면 s<=gate(m) 이내면 제외")
    parser.add_argument("--prior_uniform_n", default=200000, type=int,
                        help="uniform far-field 볼륨 샘플 수 — 셸 밖 unseen 공간 부풀림 차단")
    parser.add_argument("--prior_carve_views", default=40, type=int,
                        help="prior 할루시네이션 carve 에 쓸 균등 간격 뷰 수(0=off)")
    parser.add_argument("--prior_carve_margin", default=0.015, type=float,
                        help="carve depth 여유(m) — 관측 표면보다 이만큼 앞이면 freespace 위반")
    parser.add_argument("--grid_fuse", action="store_true",
                        help="MLP 대신 결정적 grid TSDF 융합(관측>carve>생성 우선순위). "
                             "--prior_mesh 필수, --prior_carve_views 120+ 권장. 부풀림·스펀지 원천 차단")
    parser.add_argument("--grid_wcap", default=5.0, type=float,
                        help="관측 신뢰도 포화 뷰 수 — 이 이상 관측된 복셀은 관측 TSDF 100%")
    parser.add_argument("--grid_smooth", default=0.7, type=float,
                        help="융합 grid 가우시안 스무딩 sigma(voxel). 0=off")
    parser.add_argument("--gt_depth_dir", default="", type=str,
                        help="GT depth 폴더(nice-slam results 등). 지정 시 carve 를 마스크가 아닌 "
                             "'실제 씬 depth' 기준으로 수행 — 다리 절단·경계 거침 해결")
    parser.add_argument("--gt_depth_scale", default=6553.5, type=float,
                        help="GT depth PNG 스케일(픽셀값/스케일=미터). Replica nice-slam=6553.5")
    parser.add_argument("--prior_carve_ds", default=2, type=int,
                        help="뷰 버퍼 다운스케일(작을수록 경계 정밀, 메모리↑)")
    parser.add_argument("--free_min_views", default=3, type=int,
                        help="free 판정 최소 합의 뷰 수 — 1이면 OR(공격적), 3+ 권장. "
                             "얇은 구조가 depth 경계 노이즈로 갉히는 것 방지")
    parser.add_argument("--gt_edge_thr", default=0.1, type=float,
                        help="GT depth 불연속 경계 임계(m) — 경계 픽셀 free 투표 무효화")
    parser.add_argument("--debug_class_ply", default="", type=str,
                        help="생성 표면 샘플 분류 점군 저장 경로 — 잘림 원인 시각 진단용")
    parser.add_argument("--carve_align", action="store_true",
                        help="[정합 보정] 생성이 free 공간을 피해 unknown 영역으로 들어가도록 "
                             "9-DoF 재최적화(관측 앵커 유지) — 생성-실측 형상 차이 흡수")
    parser.add_argument("--carve_align_w", default=1.0, type=float,
                        help="carve-align 관측 앵커 가중치(클수록 상판 정렬 엄격)")
    parser.add_argument("--probe_box", default="", type=str,
                        help='진단용 world 박스 "x0,y0,z0,x1,y1,z1" — 내부 복셀 분류 통계 출력')
    parser.add_argument("--grid_sign_fix", action="store_true",
                        help="생성 SDF 부호를 flood-fill 로 강제 복원(watertight=False 면 자동)")
    parser.add_argument("--shell_delta", default=0.02, type=float,
                        help="오프셋 셸 최대 반두께 δ_max(m) — 깊은 미관측 영역(다리)에 적용")
    parser.add_argument("--shell_delta_min", default=0.006, type=float,
                        help="오프셋 셸 최소 반두께 δ_min(m) — 관측 표면 인접부(테두리)에 적용")
    parser.add_argument("--shell_ramp", default=0.10, type=float,
                        help="δ_min→δ_max 전이 거리(m, 관측 표면으로부터)")
    parser.add_argument("--alpha_smooth", default=1.0, type=float,
                        help="[seam] 관측 신뢰도 alpha 가우시안 sigma(voxel) — 관측/생성 "
                             "전이대를 만들어 접합부 계단 제거. 0=off")
    parser.add_argument("--unseen_open", default=0.0, type=float,
                        help="[스파이크] 미관측 영역 모폴로지 opening 반경(m). 두께 2r 이하 "
                             "돌기를 제거한다. ⚠ 얇은 구조(테이블 다리 등)도 같이 지우므로 "
                             "기본 off. 켤 때는 반드시 다리 생존을 확인할 것")
    parser.add_argument("--no_color_match", action="store_true",
                        help="생성 색을 관측 색 통계에 맞추는 보정 비활성")
    parser.add_argument("--color_blend_ramp", default=0.05, type=float,
                        help="접합부 색 블렌드 거리(m) — 관측면에서 이 거리까지 관측색으로 "
                             "가중 혼합. 0=off")
    parser.add_argument("--hull_min_frac", default=0.6, type=float,
                        help="[visual hull] 시야에 든 뷰 중 객체 마스크 안으로 투영된 비율이 "
                             "이 값 이상인 복셀에만 prior 허용. 0=off. 생성 기하가 바닥·인접 "
                             "객체로 새는 것을 차단")
    parser.add_argument("--hull_min_views", default=5, type=int,
                        help="hull 판정에 필요한 최소 시야 뷰 수(증거 부족 복셀 배제)")
    parser.add_argument("--view_stride", default=1, type=int,
                        help="[속도] 학습 뷰를 이 간격으로만 사용(back-project/carve 비용 ∝ 뷰 수). "
                             "관측 점군은 어차피 서브샘플되므로 2~4 는 손실이 작다")
    parser.add_argument("--min_unknown_frac", default=0.10, type=float,
                        help="[적용 게이트] 생성 내부 부피 중 unknown 비율이 이 값 미만이면 "
                             "prior 를 쓰지 않는다(이미 충분히 관측된 객체). 0=항상 적용")
    parser.add_argument("--keep_connected", action="store_true",
                        help="최종 음수 볼륨 중 관측 복셀과 연결된 성분만 유지 — "
                             "통짜 생성 prior 의 타 객체 잔해 제거(배치에서 권장)")
    parser.add_argument("--carve_depth_dir", default="", type=str,
                        help="dump_scene_depth.py 출력 폴더. 전체 씬 200뷰 depth로 free-space carving (empty-ray보다 우선)")
    parser.add_argument("--offsurf_delta", default=0.01, type=float, help="정규화 좌표 기준 off-surface 오프셋")
    parser.add_argument("--grid", default=0, type=int, help="marching cubes 해상도(0=voxel_size로 산출)")
    parser.add_argument("--max_grid", default=512, type=int)
    parser.add_argument("--mask_dist", default=0.10, type=float,
                        help="메쉬 정점이 관측 점군에서 이 거리(world) 초과면 제거(0=off) — 박스 제거 vs 구멍채움 균형. "
                             "unseen 완성(측면/뒷면 보간)을 보존하려면 ROI crop과 함께 0 또는 크게")
    parser.add_argument("--roi_mesh", default="", type=str,
                        help="관측 anchor mesh(예: TSDF fuse_post.ply). 이 mesh에서 roi_dist 밖 점은 SDF 입력에서 제외")
    parser.add_argument("--roi_dist", default=0.15, type=float)
    parser.add_argument("--mask_dir", default="auto", type=str,
                        help="뷰별 객체 마스크 폴더. 'auto'=<source_path>/masks (있으면 사용), ''=사용 안 함")
    parser.add_argument("--require_mask", action="store_true",
                        help="마스크 없는 학습 뷰는 통째로 skip (composed 모델에서 객체만 추출할 때 필수)")
    parser.add_argument("--extra_poses", default="", type=str,
                        help="추가 novel 포즈 npz(render_hole_novel soft_out poses.npz). "
                             "See3D 정제된 unseen 밴드를 추출에 포함")
    parser.add_argument("--extra_mask_npy", default="", type=str,
                        help="per-Gaussian 객체 라벨 npy — extra 포즈에서 label-buffer 렌더로 "
                             "객체 마스크 생성(base/바닥 유입 차단). 미지정 시 extra 뷰는 마스크 없음")
    parser.add_argument("--extra_mask_thr", default=0.3, type=float)
    parser.add_argument("--out", default="", type=str)
    args = get_combined_args(parser)

    _T0 = time.time(); _tk = _T0

    def _lap(msg):
        now = time.time()
        print(f"[time] {msg}: {now - _lap.prev:.1f}s (누적 {now - _T0:.1f}s)", flush=True)
        _lap.prev = now
    _lap.prev = _tk

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    _lap(f"Scene 로드 (학습뷰 {len(scene.getTrainCameras())}장)")
    gaussians.active_sh_degree = 0  # diffuse만(테xture) — render.py mesh 경로와 동일
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    # 1) oriented point cloud (객체 마스크 밖 픽셀 제외 — TSDF 경로와 동일)
    mask_dir = None
    if args.mask_dir == "auto":
        cand = os.path.join(dataset.source_path, "masks")
        mask_dir = cand if os.path.isdir(cand) else None
    elif args.mask_dir:
        mask_dir = args.mask_dir

    extra_cams = []
    if args.extra_poses:
        from scene.cameras import MiniCam
        recs = np.load(os.path.expanduser(args.extra_poses), allow_pickle=True)["records"]
        for r in recs:
            wvt = torch.tensor(np.asarray(r["world_view_transform"]), dtype=torch.float32).cuda()
            fpt = torch.tensor(np.asarray(r["full_proj_transform"]), dtype=torch.float32).cuda()
            mc = MiniCam(int(r["width"]), int(r["height"]),
                         float(r["FoVy"]), float(r["FoVx"]), 0.01, 100.0, wvt, fpt)
            mc.image_name = f"extra{int(r['idx']):04d}"
            extra_cams.append(mc)

    extra_masks = None
    if extra_cams and args.extra_mask_npy:
        lab_np = np.load(os.path.expanduser(args.extra_mask_npy)).astype(np.float32)
        assert len(lab_np) == gaussians.get_xyz.shape[0], \
            f"extra_mask_npy {len(lab_np)} != gaussians {gaussians.get_xyz.shape[0]}"
        extra_masks = render_extra_masks(extra_cams, gaussians, pipe, background,
                                         torch.from_numpy(lab_np).cuda(), thr=args.extra_mask_thr)
        cov = np.mean([m.float().mean().item() for m in extra_masks.values()])
        print(f"extra 객체 마스크 생성: {len(extra_masks)}뷰 (평균 커버 {cov*100:.1f}%)")

    print("뷰별 depth back-project + 법선 정렬 ...")
    P, N, C, O, EO, ED, VB = collect_oriented_points(scene, gaussians, pipe, background, args,
                                                     mask_dir=mask_dir,
                                                     require_mask=args.require_mask,
                                                     extra_cams=extra_cams,
                                                     extra_masks=extra_masks)
    _lap("뷰별 depth back-project")
    print(f"표면점 {len(P)} (관측 back-projected)")
    if len(P) < 1000:
        raise SystemExit(f"[중단] 유효 표면점 {len(P)}개 — 마스크 값 규약 또는 alpha_thr 확인 필요. "
                         f"(마스크 없이 테스트: --mask_dir '' , alpha 완화: --alpha_thr 0.5)")

    # 1b) ROI crop — 신뢰 가능한 관측 mesh(TSDF fuse_post 등) 근방 점만 유지.
    #     instance 모델의 배경/마스크 경계 junk(검은 노이즈)를 SDF 피팅 전에 제거.
    if args.roi_mesh:
        from scipy.spatial import cKDTree as _KD
        rm = o3d.io.read_triangle_mesh(args.roi_mesh)
        rv = np.asarray(rm.vertices)
        assert len(rv) > 0, f"ROI mesh 비어있음: {args.roi_mesh}"
        d, _ = _KD(rv).query(P, workers=-1)
        keep = d < args.roi_dist
        print(f"ROI crop: {int(keep.sum())}/{len(P)} 유지 (dist<{args.roi_dist}, mesh={args.roi_mesh})")
        P, N, C, O = P[keep], N[keep], C[keep], O[keep]

    if len(P) > args.n_pts:
        idx = np.random.choice(len(P), args.n_pts, replace=False)
        P, N, C, O = P[idx], N[idx], C[idx], O[idx]

    # robust 정규화 [-1,1] — floater가 scale을 부풀리지 않도록 percentile bbox 밖 점 제거
    lo = np.percentile(P, 0.5, axis=0); hi = np.percentile(P, 99.5, axis=0)
    pad = 0.05 * (hi - lo)
    keep = np.all((P >= lo - pad) & (P <= hi + pad), axis=1)
    n_drop = int((~keep).sum())
    P, N, C, O = P[keep], N[keep], C[keep], O[keep]
    center = (lo + hi) / 2
    scale = np.abs(P - center).max() * 1.1
    print(f"robust bbox: outlier {n_drop}점 제거, scale={scale:.3f} world (bbox {np.round(hi-lo,3)})")
    # [FIX] Pn/On 은 prior 주입 '후'에 계산 (기존엔 여기서 계산해 stale Pn 이 train 에 들어가
    # prior 점이 표면 손실에 전혀 반영되지 않던 치명 버그 — 다리 무감독→팽창의 직접 원인)
    EOn = (EO - center) / scale if len(EO) else EO   # 방향(ED)은 정규화 불변
    if len(EOn):
        t0 = -(EOn * ED).sum(-1)
        dmin = np.linalg.norm(EOn + ED * t0[:, None], axis=-1)
        print(f"empty ray 진단: t0(전방거리) 중앙값 {np.median(t0):.2f} (양수여야 정상), "
              f"dmin(중심 최근접) min/중앙값 {dmin.min():.2f}/{np.median(dmin):.2f} (단위=정규화, bbox≈1)")
        keep_e = (t0 > 0) & (dmin < 1.2)             # 객체 bbox 근처를 실제로 지나는 광선만
        EOn, ED = EOn[keep_e], ED[keep_e]
        print(f"empty ray 필터: {int(keep_e.sum())}/{len(keep_e)} 유지 (bbox 관통 광선)")

    n_obs = len(P)
    OBS = np.ones(n_obs, bool)          # True=관측점. prior 점은 False → l_free 제외

    # 1b-2) [prior·점군 경로] 생성 뷰 점군 주입 (make_gen_points.py / fuse --export_points).
    #       [FIX Step4-B] OBS=False 로 표시 — 가짜 원점(O=center) free-carve 버그 제거.
    if args.extra_points:
        pp = o3d.io.read_point_cloud(os.path.expanduser(args.extra_points))
        Pe = np.asarray(pp.points)
        Ne = np.asarray(pp.normals) if pp.has_normals() else None
        Ce = np.asarray(pp.colors) if pp.has_colors() else np.tile([0.6, 0.6, 0.6], (len(Pe), 1))
        assert Ne is not None and len(Ne) == len(Pe), "extra_points 에 법선 필요 (make_gen_points.py 사용)"
        if 0 < args.prior_repeat < 1.0:
            sel = np.random.choice(len(Pe), max(int(len(Pe) * args.prior_repeat), 1), replace=False)
            Pe, Ne, Ce = Pe[sel], Ne[sel], Ce[sel]
        P = np.concatenate([P, Pe]); N = np.concatenate([N, Ne])
        C = np.concatenate([C, Ce]); O = np.concatenate([O, np.tile(center, (len(Pe), 1))])
        OBS = np.concatenate([OBS, np.zeros(len(Pe), bool)])
        print(f"prior 점군 주입: {len(Pe)}점 (총 {len(P)})")
        n_extra = len(Pe)
    else:
        n_extra = 0

    # 1b-3) [prior·메쉬 경로, Step2] 정합된 watertight 생성 메쉬
    #       (fuse_generated_mesh --save_aligned 출력 *_gen_aligned.ply).
    #       표면 샘플(face normal, unseen 게이트) + 볼륨 셸 샘플의 target SDF 회귀.
    PV = PS = None
    rs_prior = None
    _sd = None
    prior_dbg = None

    # 1b-3') [prior·필드 경로] ShapeR 디코더의 '부호 있는' SDF 그리드를 직접 주입.
    #   메쉬를 거치지 않으므로:
    #     - sign-fix 불필요 (필드가 이미 signed)
    #     - shell_delta 불필요 (제로 두께 시트 문제 자체가 없음. ShapeR 의 UDF 메쉬는
    #       |f|=iso 로 뽑혀 표면 양쪽에 껍질이 생기는데, 여기선 그 단계를 건너뜀)
    #     - 정합 불필요 (ShapeR 은 metric + world 변환)
    #   → grid_fuse 의 SG 를 이 필드에서 삼선형 보간으로 채운다.
    if args.prior_field:
        z = np.load(os.path.expanduser(args.prior_field))
        Ffield = z["field"].astype(np.float32)
        f_center = z["center"]; f_R = z["R_align"]; f_scale = float(z["scale"])
        Gf = Ffield.shape[0]
        print(f"[prior-field] {os.path.basename(args.prior_field)}  G={Gf}  "
              f"voxel={float(z['vox_world'])*1000:.2f}mm  "
              f"내부 {(Ffield < 0).mean()*100:.2f}%  "
              f"범위 [{Ffield.min():.4f}, {Ffield.max():.4f}]m")

        # [truncation 정규화] 디코더 출력은 객체마다 다른 값에서 포화한다(obj1 ±27mm,
        # obj20 ±6mm). 그런데 융합은 carve 복셀을 +prior_trunc(50mm)로 채우므로,
        # 포화값이 작을수록 carve 경계에서 필드가 급점프하고 스무딩이 그 구간에
        # **인위적 영교차**(가짜 표면)를 만든다. 영교차 위치는 스케일링에 불변이므로
        # 포화값을 prior_trunc 에 맞춰 두 필드를 commensurate 하게 만든다.
        if args.prior_field_rescale:
            sat = float(np.percentile(np.abs(Ffield), 99.5))
            if sat > 1e-9 and abs(sat - args.prior_trunc) / args.prior_trunc > 0.2:
                Ffield = (Ffield * (args.prior_trunc / sat)).astype(np.float32)
                print(f"  → truncation 정규화: 포화 {sat*1000:.1f}mm → "
                      f"{args.prior_trunc*1000:.0f}mm (×{args.prior_trunc/sat:.2f})")
        args.prior_watertight = True          # sign-fix 비활성(이미 signed)

        # [앙상블 σ 가중] 50% 이상 미관측이면 정답이 하나가 아니다. 여러 시드가 동의하는
        # 곳만 prior 를 믿고, 갈리는 곳(σ 큼)은 '표면 없음(+trunc)' 쪽으로 후퇴시켜
        # 관측 표면에서의 자연스러운 연장(eikonal·평활화)에 맡긴다.
        Fsig = None
        if "field_std" in z.files and args.prior_sigma_w > 0:
            Fsig = z["field_std"].astype(np.float32)
            near = np.abs(Ffield) < 3 * float(z["vox_world"])
            # 필드가 metric 이므로 σ0 는 '허용 가능한 표면 위치 불확실성'의 절대 기준.
            # 기본 = prior_trunc → σ≪σ0 이면 w≈1(그대로 신뢰), σ≈σ0 이면 w=0.5.
            # ※ σ0 를 σ 중앙값으로 잡으면 정의상 복셀 절반이 억제되므로 쓰지 않는다.
            s0 = max(args.prior_sigma_ref if args.prior_sigma_ref > 0
                     else args.prior_trunc, 1e-6)
            Wsig = 1.0 / (1.0 + args.prior_sigma_w * (Fsig / s0) ** 2)
            sm = float(np.median(Fsig[near])) if near.any() else float("nan")
            print(f"[prior-field] σ 가중 활성: σ0={s0*1000:.1f}mm  "
                  f"표면근방 σ 중앙값 {sm*1000:.2f}mm  "
                  f"w 중앙값 {float(np.median(Wsig[near])) if near.any() else float('nan'):.3f}  "
                  f"(w<0.5 복셀 {(Wsig < 0.5).mean()*100:.1f}%)")

        def _sd(q):
            """world 점 → 근사 metric SDF(음수=내부). 그리드 밖은 +trunc."""
            n = ((np.asarray(q, np.float64) - f_center) @ f_R.T) * f_scale
            idx = (n + 1.0) * (Gf - 1) / 2.0
            out = np.full(len(idx), args.prior_trunc, np.float64)
            ok = np.all((idx >= 0) & (idx <= Gf - 1.001), axis=1)
            if not ok.any():
                return out
            p = idx[ok]
            i0 = np.floor(p).astype(np.int64); w = p - i0
            i1 = i0 + 1

            def _interp(vol):                 # 삼선형 보간
                v = np.zeros(len(p), np.float64)
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            ww = ((w[:, 0] if dx else 1 - w[:, 0])
                                  * (w[:, 1] if dy else 1 - w[:, 1])
                                  * (w[:, 2] if dz else 1 - w[:, 2]))
                            v += ww * vol[(i1 if dx else i0)[:, 0],
                                          (i1 if dy else i0)[:, 1],
                                          (i1 if dz else i0)[:, 2]]
                return v

            v = _interp(Ffield)
            if Fsig is not None:              # 합의도 가중: 불확실하면 '표면 없음'으로 후퇴
                wg = _interp(Wsig)
                v = wg * v + (1 - wg) * args.prior_trunc
            out[ok] = v
            return out
    if args.prior_mesh:
        import open3d.core as o3c
        from scipy.spatial import cKDTree as _KDp
        pm_path = os.path.expanduser(args.prior_mesh)
        gm = o3d.io.read_triangle_mesh(pm_path)
        assert len(gm.vertices), f"prior mesh 로드 실패: {args.prior_mesh}"
        # glb 등 UV 텍스처 입력이면 vertex color 로 bake (없으면 샘플 점이 흰색이 됨).
        # ※ "more than 1 material" Open3D 경고는 RaycastingScene 변환 시 재질을 버린다는
        #   의미일 뿐 — SDF(geometry) 계산에는 무해.
        if not (gm.has_vertex_colors() and len(gm.vertex_colors) == len(gm.vertices)):
            try:
                import trimesh
                tm = trimesh.load(pm_path, process=False, force="mesh")
                vc = np.asarray(tm.visual.to_color().vertex_colors)[:, :3] / 255.0
                if len(vc) == len(tm.vertices):
                    gm = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(np.asarray(tm.vertices, np.float64)),
                        o3d.utility.Vector3iVector(np.asarray(tm.faces, np.int32)))
                    gm.vertex_colors = o3d.utility.Vector3dVector(np.clip(vc, 0, 1))
                    print(f"[prior] 텍스처→정점색 bake ({len(vc)} verts)")
            except Exception as e:
                print(f"[prior] 색 bake 실패({e}) — 회색 유지")
        wt = gm.is_watertight()
        args.prior_watertight = bool(wt)
        print(f"[prior] mesh verts {len(gm.vertices)} watertight={wt}"
              + ("" if wt else "  ⚠ signed distance 부호 불안정 — grid_fuse 에서 sign-fix 자동 적용"))
        rs_prior = o3d.t.geometry.RaycastingScene()
        rs_prior.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gm))

        def _sd(q):     # world 좌표 signed distance (양수=밖)
            return rs_prior.compute_signed_distance(
                o3c.Tensor(q.astype(np.float32))).numpy().astype(np.float64)

        obs_tree_w = _KDp(P[:n_obs])
        band, trunc = args.prior_band, args.prior_trunc
        # 표면 샘플 — face normal 사용(생성 mesh vertex normal 오염 회피) + 부호 검증
        sp = gm.sample_points_uniformly(args.prior_surf_n, use_triangle_normal=True)
        Ps_w = np.asarray(sp.points); Ns_w = np.asarray(sp.normals).copy()
        prior_dbg = Ps_w.copy()                      # [debug] 분류 시각화용(게이트 전 원본)
        flip = _sd(Ps_w + 0.5 * band * Ns_w) < 0     # p+εn 이 내부면 법선 뒤집힘
        Ns_w[flip] = -Ns_w[flip]
        print(f"[prior] 표면 법선 부호 flip {int(flip.sum())}/{len(Ns_w)}")
        # [할루시네이션 carve] 렌더 depth+mask 뷰 버퍼(VB)로 prior 샘플 검증.
        #   freespace 위반: 마스크 안인데 관측 표면보다 '앞'에 뜸 (상판 위 덩어리 등)
        #   silhouette 위반: 마스크 밖 픽셀에 비가림 상태로 투영 (꼬리 등)
        def _carve_viol(Xw, margin=args.prior_carve_margin):
            viol = np.zeros(len(Xw), bool)
            for b in VB:
                Xc = Xw @ b["R"].T + b["t"]
                z = Xc[:, 2]
                zz = np.maximum(z, 1e-6)
                u = b["fx"] * Xc[:, 0] / zz + b["cx"]
                v = b["fy"] * Xc[:, 1] / zz + b["cy"]
                infr = (z > 0.05) & (u >= 0) & (u < b["W"]) & (v >= 0) & (v < b["H"])
                ui = np.clip(u, 0, b["W"] - 1).astype(int)
                vi = np.clip(v, 0, b["H"] - 1).astype(int)
                di = b["depth"][vi, ui]; mi = b["mask"][vi, ui]
                front = (di > 0) & (z < di - margin)             # 관측 표면보다 앞
                sil = (~mi) & ((di <= 0) | front)                # 실루엣 밖 & 비가림
                viol |= infr & ((mi & front) | sil)
            return viol

        # [Step1] unseen 게이트: 관측점 τ 이내 표면 샘플 제거 — 관측이 항상 승리(이중 표면 방지)
        du, _ = obs_tree_w.query(Ps_w, workers=-1)
        ku = du > args.prior_unseen_dist
        Cs = (np.asarray(sp.colors)[ku] if len(sp.colors) == len(ku)
              else np.tile([0.6, 0.6, 0.6], (int(ku.sum()), 1)))
        Ps_w, Ns_w = Ps_w[ku], Ns_w[ku]
        # 표면 샘플 carve — 관측과 모순되는 생성 표면(할루시네이션) 제거
        vv = _carve_viol(Ps_w)
        Ps_w, Ns_w, Cs = Ps_w[~vv], Ns_w[~vv], Cs[~vv]
        print(f"[prior] unseen 표면 샘플 {len(Ps_w)}/{len(ku)} "
              f"(τ={args.prior_unseen_dist}m, carve 제거 {int(vv.sum())})")
        P = np.concatenate([P, Ps_w]); N = np.concatenate([N, Ns_w])
        C = np.concatenate([C, Cs]);  O = np.concatenate([O, np.tile(center, (len(Ps_w), 1))])
        OBS = np.concatenate([OBS, np.zeros(len(Ps_w), bool)])
        n_extra += len(Ps_w)
        # [Step2] 볼륨 샘플 = 셸(표면 ±band) + uniform far-field.
        #   셸: 얇은 구조 '양쪽' 근접 빈 공간을 양수로 감독.
        #   uniform: 셸 밖 unseen ROI 전체를 truncated SDF 로 감독 —
        #            셸 사이 무감독 공간에서 생기던 잔여 부풀림 차단.
        rng = np.random.default_rng(0)
        Xs = [Ps_w + rng.standard_normal(Ps_w.shape) * band * f for f in (0.25, 1.0)]
        Xu = rng.uniform(-1, 1, (args.prior_uniform_n, 3)) * scale + center
        X = np.concatenate(Xs + [Xu])
        sd_x = _sd(X)
        tgt = np.clip(sd_x, -trunc, trunc)
        # carve 위반 볼륨 샘플: 제거 대신 target=+trunc 강제(관측된 빈 공간 = 확실한 '밖')
        vx = _carve_viol(X)
        tgt[vx] = trunc
        dxo, _ = obs_tree_w.query(X, workers=-1)
        kx = dxo > args.prior_unseen_dist            # 관측 근방 샘플 제외(관측 항이 담당)
        Xn_ = (X[kx] - center) / scale
        kin = np.all(np.abs(Xn_) < 1.0, axis=1)      # 정규화 큐브 내부만
        PV = Xn_[kin].astype(np.float64)
        PS = (tgt[kx][kin] / scale).astype(np.float64)
        print(f"[prior] 볼륨 distill 샘플 {len(PV)} (셸 {len(X)-len(Xu)} + uniform {len(Xu)}, "
              f"carve override {int(vx.sum())}, band={band}m trunc={trunc}m)")

    # 정규화 좌표 — prior 주입 '후' 계산 ([FIX] stale Pn 버그 수정의 핵심)
    Pn = (P - center) / scale
    On = (O - center) / scale

    Wp = np.ones(len(P), np.float32)
    if n_extra and args.prior_weight != 1.0:
        Wp[len(P) - n_extra:] = args.prior_weight
        print(f"prior 가중치 {args.prior_weight}: seen {len(P)-n_extra} : prior {n_extra}")

    # 1c) 전체 씬 depth 기반 carve 샘플 (있으면 empty-ray보다 우선)
    CV = None
    if args.carve_depth_dir:
        CV = load_carve_points(args.carve_depth_dir, center, scale)
        print(f"carve 샘플 {len(CV)}개 (전체 씬 depth, bbox 내부)")

    # 1d) [FIX Step4-D] carve/empty 샘플이 prior mesh 근방(s<=prior_gate)이면 제외 —
    #     'SDF>=0 강제'와 'l_prior 음수 회귀'가 같은 voxel에서 싸우는 것을 차단.
    if _sd is not None:
        gate = args.prior_gate
        if CV is not None and len(CV):
            keep_cv = _sd(CV * scale + center) > gate
            print(f"[prior] carve 게이트: {int(keep_cv.sum())}/{len(CV)} 유지 (s<={gate}m 제거)")
            CV = CV[keep_cv]
        if len(EOn):
            # empty ray 를 chord 상 '고정 샘플'로 변환 → prior 근방 제외 → CV 병합, chord 경로 비활성
            K = 4
            t0 = -(EOn * ED).sum(-1, keepdims=True)
            cp = EOn + ED * t0
            half = np.sqrt(np.clip(1.44 - (cp * cp).sum(-1, keepdims=True), 0.0, None))
            ts = np.clip(t0 + (np.random.rand(len(EOn), K) * 2 - 1) * half, 0.05, None)
            Xe = (EOn[:, None, :] + ED[:, None, :] * ts[..., None]).reshape(-1, 3)
            Xe = Xe[np.all(np.abs(Xe) < 1.2, axis=1)]
            Xe = Xe[_sd(Xe * scale + center) > gate]
            CV = Xe if (CV is None or not len(CV)) else np.concatenate([CV, Xe])
            EOn = np.zeros((0, 3)); ED = np.zeros((0, 3))
            print(f"[prior] empty-ray → 고정 샘플 {len(Xe)} (게이트 적용, chord 경로 비활성)")

    # 2) SDF 생성 — grid_fuse(결정적 융합, MLP 없음) 또는 IGR MLP 학습
    net = None
    if args.grid_fuse:
        assert _sd is not None, "--grid_fuse 는 --prior_field 또는 --prior_mesh 필요"
        assert VB, "--grid_fuse 는 뷰 버퍼 필요 (--prior_carve_views > 0, mask_dir 필수)"
        verts, faces = grid_fuse_tsdf(VB, _sd, center, scale, args, debug_pts=prior_dbg)
        _lap("grid_fuse (TSDF 적분 + carve + marching cubes)")
    else:
        print("IGR SDF 학습 ...")
        net = train_sdf(Pn, N, On, EOn, ED, args, CV=CV, W=Wp, OBS=OBS, PV=PV, PS=PS)

    # 3) 그리드 평가 + marching cubes — 타일 분할로 고해상도(whole-scene) 지원
    G = args.grid if args.grid > 0 else int(round(2 * scale / args.voxel_size))
    G = int(min(G, args.max_grid))
    from skimage.measure import marching_cubes
    if net is not None:
        net.eval()
    lin = np.linspace(-1, 1, G, dtype=np.float32)
    step = 2.0 / (G - 1)

    if net is None:
        pass                                    # grid_fuse 경로: verts/faces 이미 생성됨
    elif args.tile <= 0 or G <= args.tile:
        print(f"SDF 그리드 평가 (G={G}, voxel≈{2*scale/(G-1):.4f} world) + marching cubes ...")
        vol = np.empty((G, G, G), np.float32)
        with torch.no_grad():
            gx, gy = np.meshgrid(lin, lin, indexing="ij")
            for k in range(G):
                pts = np.stack([gx, gy, np.full_like(gx, lin[k])], -1).reshape(-1, 3)
                s = net(torch.tensor(pts, dtype=torch.float32, device="cuda")).cpu().numpy().reshape(G, G)
                vol[:, :, k] = s
        verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(step,) * 3)
        verts = (verts - 1.0) * scale + center
    else:
        # ── 타일 marching: G를 tile 크기 블록으로 쪼개 각각 marching 후 병합 ──
        #    블록 경계는 1복셀 오버랩으로 이어붙여 이음새 없음. 메모리 O(tile^3).
        T = int(args.tile)
        nb = int(np.ceil((G - 1) / (T - 1)))
        print(f"SDF 타일 marching (G={G}, voxel≈{2*scale/(G-1):.4f} world, "
              f"tile={T}, blocks={nb}^3={nb**3}) ...")
        vs_all, fs_all, voff = [], [], 0
        with torch.no_grad():
            for bi in range(nb):
                i0 = bi * (T - 1); i1 = min(i0 + T, G)
                for bj in range(nb):
                    j0 = bj * (T - 1); j1 = min(j0 + T, G)
                    for bk in range(nb):
                        k0 = bk * (T - 1); k1 = min(k0 + T, G)
                        xs, ys, zs = lin[i0:i1], lin[j0:j1], lin[k0:k1]
                        if min(len(xs), len(ys), len(zs)) < 2:
                            continue
                        sub = np.empty((len(xs), len(ys), len(zs)), np.float32)
                        gx, gy = np.meshgrid(xs, ys, indexing="ij")
                        for kk, zv in enumerate(zs):
                            pts = np.stack([gx, gy, np.full_like(gx, zv)], -1).reshape(-1, 3)
                            sub[:, :, kk] = net(torch.tensor(pts, dtype=torch.float32, device="cuda")
                                                ).cpu().numpy().reshape(len(xs), len(ys))
                        if sub.min() > 0 or sub.max() < 0:      # zero-crossing 없는 블록 skip
                            continue
                        v, f, _, _ = marching_cubes(sub, level=0.0, spacing=(step,) * 3)
                        v = v + np.array([xs[0], ys[0], zs[0]]) + 1.0   # 블록 원점 → [-1,1] 좌표
                        vs_all.append((v - 1.0) * scale + center)
                        fs_all.append(f + voff)
                        voff += len(v)
                print(f"  블록 {bi+1}/{nb} 행 완료 (누적 verts {voff})")
        assert vs_all, "zero-crossing 블록 없음 — 학습/스케일 확인"
        verts = np.concatenate(vs_all); faces = np.concatenate(fs_all)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    if args.mask_dist > 0:
        d, _ = tree.query(verts, workers=-1)
        far = d > args.mask_dist
        mesh.remove_vertices_by_mask(far)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        print(f"거리 트리밍(d>{args.mask_dist}): {int(far.sum())}/{len(verts)} 정점 제거")

    # 색: [1] 생성 색을 관측 색 통계에 정합(톤 차이 제거)
    #     [2] 접합부에서 관측색↔생성색 거리 가중 블렌드(하드 컷 제거)
    verts2 = np.asarray(mesh.vertices)
    Cw = np.clip(C, 0, 1).copy()
    n_obs = int(OBS.sum()); n_pri = int((~OBS).sum())
    if (not args.no_color_match) and n_obs > 100 and n_pri > 100:
        mo, so = Cw[OBS].mean(0), Cw[OBS].std(0) + 1e-6
        mp, sp = Cw[~OBS].mean(0), Cw[~OBS].std(0) + 1e-6
        Cw[~OBS] = np.clip((Cw[~OBS] - mp) / sp * so + mo, 0, 1)
        print(f"[색] 생성 색 통계 정합: mean {np.round(mp,3)} → {np.round(mo,3)}")
    if args.color_blend_ramp > 0 and n_obs > 100 and n_pri > 0:
        t_obs = cKDTree(P[OBS]); Cobs = Cw[OBS]
        d_o, i_o = t_obs.query(verts2, workers=-1)
        _, i_a = tree.query(verts2, workers=-1)
        w = np.clip(1.0 - d_o / args.color_blend_ramp, 0, 1)[:, None]
        col = w * Cobs[i_o] + (1 - w) * Cw[i_a]
        print(f"[색] 접합부 블렌드 ramp {args.color_blend_ramp*1000:.0f}mm "
              f"(전이 정점 {int(((w > 0) & (w < 1)).sum())}/{len(verts2)})")
    else:
        _, ni = tree.query(verts2, workers=-1)
        col = Cw[ni]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
    mesh.compute_vertex_normals()

    # 5) safe_post_process_mesh(num_cluster) — TSDF 경로와 동일 로직(클램프 추가)

    out = os.path.expanduser(args.out)
    if not out:
        train_dir = os.path.join(args.model_path, "train", f"ours_{scene.loaded_iter}")
        os.makedirs(train_dir, exist_ok=True)
        out = os.path.join(train_dir, "sdf_fuse.ply")
    # [FIX] Open3D 는 확장자로 포맷 판별 → 확장자 없음/미지원이면 "unknown file extension"
    # 경고만 내고 조용히 실패. 자동 교정 + 저장 성공 여부 검증(실패 시 즉시 중단).
    if os.path.splitext(out)[1].lower() not in (".ply", ".obj", ".stl", ".off", ".gltf", ".glb"):
        print(f"[경고] 출력 확장자 없음/미지원 ('{out}') → '.ply' 부착")
        out = out + ".ply"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ok = o3d.io.write_triangle_mesh(out, mesh)
    assert ok, f"[중단] 메쉬 저장 실패: {os.path.abspath(out)}"
    print(f"mesh saved at {os.path.abspath(out)}  verts {len(verts)} faces {len(faces)}")

    mesh_post = safe_post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    out_post = os.path.splitext(out)[0] + "_post.ply"
    ok = o3d.io.write_triangle_mesh(out_post, mesh_post)
    assert ok, f"[중단] post 메쉬 저장 실패: {os.path.abspath(out_post)}"
    print(f"mesh post processed saved at {os.path.abspath(out_post)}")


if __name__ == "__main__":
    main()