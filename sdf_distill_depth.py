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
    ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2],
                            [0, H / 2, 0, (H - 1) / 2],
                            [0, 0, 0, 1]]).float().cuda().T
    intrins = (cam.projection_matrix @ ndc2pix)[:3, :3].T
    fx, fy = intrins[0, 0].item(), intrins[1, 1].item()
    cx, cy = intrins[0, 2].item(), intrins[1, 2].item()
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


# ---------------------------------------------------------------------------
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


@torch.no_grad()
def collect_oriented_points(scene, gaussians, pipe, background, args, mask_dir=None):
    """뷰별 depth를 월드 점군으로 back-project. 법선은 카메라 방향으로 정렬.
    mask_dir가 있으면 객체 마스크 밖 픽셀은 back-project에서 제외(TSDF 경로와 동일 철학)."""
    views = scene.getTrainCameras()
    P_all, N_all, C_all, O_all = [], [], [], []
    EO_all, ED_all = [], []
    n_masked_views = 0
    for cam in views:
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
        if mask_dir is not None:
            m = load_view_mask(mask_dir, cam.image_name, H, W)
            if m is not None:
                valid &= m
                n_masked_views += 1
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
        if args.empty_per_view > 0:
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
        print(f"객체 마스크 적용: {n_masked_views}/{len(views)} 뷰 (경로 {mask_dir})")
    P = torch.cat(P_all).numpy().astype(np.float64)
    N = torch.cat(N_all).numpy().astype(np.float64)
    C = torch.cat(C_all).numpy().astype(np.float64)
    O = torch.cat(O_all).numpy().astype(np.float64)
    EO = torch.cat(EO_all).numpy().astype(np.float64) if EO_all else np.zeros((0, 3))
    ED = torch.cat(ED_all).numpy().astype(np.float64) if ED_all else np.zeros((0, 3))
    print(f"빈 광선(empty ray) {len(EO)}개 수집")
    return P, N, C, O, EO, ED


def train_sdf(P, N, O, EO, ED, args):
    """정규화된 oriented point cloud에 IGR SDF 피팅.
    O = 점별 관측 카메라 중심(정규화 좌표) — 표면 근처 free-space carving.
    EO/ED = 빈 광선(alpha≈0 픽셀)의 카메라 중심/방향 — empty-ray carving."""
    dev = "cuda"
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    Nt = torch.tensor(N, dtype=torch.float32, device=dev)
    Ot = torch.tensor(O, dtype=torch.float32, device=dev)
    EOt = torch.tensor(EO, dtype=torch.float32, device=dev) if EO is not None and len(EO) else None
    EDt = torch.tensor(ED, dtype=torch.float32, device=dev) if ED is not None and len(ED) else None
    net = SDFNet(pe_L=args.pe_L).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    delta = args.offsurf_delta
    for it in range(args.sdf_iters):
        bi = torch.randint(0, len(Pt), (args.batch,), device=dev)
        pts = Pt[bi].clone().requires_grad_(True)
        nrm = Nt[bi]
        sdf = net(pts)
        g = grad(sdf, pts)
        l_man = sdf.abs().mean()
        l_nrm = (1 - torch.nn.functional.cosine_similarity(g, nrm, dim=-1)).mean()

        # signed off-surface: p+δn -> +δ(바깥), p-δn -> -δ(안쪽). 부호장 안정화(스펀지 방지).
        pp = (Pt[bi] + delta * nrm)
        pm = (Pt[bi] - delta * nrm)
        l_sign = (net(pp) - delta).abs().mean() + (net(pm) + delta).abs().mean()

        # free-space carving (표면 근처): 관측점 p에서 카메라 방향으로 s∈[2δ, free_range] 후퇴한
        # 점은 빈 공간 → SDF ≥ 0. (bbox 안 표면 근처를 집중 샘플 — 멀리 카메라 쪽은 정보 없음)
        l_free = torch.tensor(0.0, device=dev)
        if args.w_free > 0:
            dirv = Pt[bi] - Ot[bi]
            dist = dirv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            dirn = dirv / dist
            s = torch.rand(args.batch, 1, device=dev) * (args.free_range - 2 * delta) + 2 * delta
            s = torch.minimum(s, dist * 0.95)
            xf = Pt[bi] - dirn * s
            l_free = torch.relu(-net(xf)).mean()

        # empty-ray carving: 렌더 alpha≈0 픽셀의 광선은 '아무것도 없음'이 관측된 것 →
        # 광선이 반경 1.2 구(=bbox) 를 지나는 chord 구간 안에서만 샘플해 SDF ≥ 0 강제.
        l_empty = torch.tensor(0.0, device=dev)
        if args.w_empty > 0 and EOt is not None and len(EOt) > 0:
            bj = torch.randint(0, len(EOt), (args.batch,), device=dev)
            o, dn = EOt[bj], EDt[bj]
            t0 = -(o * dn).sum(-1, keepdim=True)                 # 원점 최근접 파라미터
            cp = o + dn * t0
            half = (1.44 - (cp * cp).sum(-1, keepdim=True)).clamp(min=0.0).sqrt()
            t = (t0 + (torch.rand_like(t0) * 2 - 1) * half).clamp(min=0.05)
            xe = o + dn * t
            l_empty = torch.relu(-net(xe)).mean()                         # 음수(내부)만 벌점

        # eikonal: 표면 근처 + 균등 랜덤
        rp = torch.cat([Pt[torch.randint(0, len(Pt), (args.batch,), device=dev)]
                        + 0.02 * torch.randn(args.batch, 3, device=dev),
                        torch.rand(args.batch, 3, device=dev) * 2 - 1], 0).requires_grad_(True)
        ge = grad(net(rp), rp)
        l_eik = ((ge.norm(dim=-1) - 1) ** 2).mean()

        loss = (l_man + args.w_normal * l_nrm + args.w_sign * l_sign
                + args.w_eik * l_eik + args.w_free * l_free + args.w_empty * l_empty)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"[{it}] man {l_man.item():.4f} nrm {l_nrm.item():.4f} "
                  f"sign {l_sign.item():.4f} eik {l_eik.item():.4f} "
                  f"free {l_free.item():.4f} empty {l_empty.item():.4f}")
    return net


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
    parser.add_argument("--out", default="", type=str)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
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
    print("뷰별 depth back-project + 법선 정렬 ...")
    P, N, C, O, EO, ED = collect_oriented_points(scene, gaussians, pipe, background, args, mask_dir=mask_dir)
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
    Pn = (P - center) / scale
    On = (O - center) / scale
    EOn = (EO - center) / scale if len(EO) else EO   # 방향(ED)은 정규화 불변
    if len(EOn):
        t0 = -(EOn * ED).sum(-1)
        dmin = np.linalg.norm(EOn + ED * t0[:, None], axis=-1)
        print(f"empty ray 진단: t0(전방거리) 중앙값 {np.median(t0):.2f} (양수여야 정상), "
              f"dmin(중심 최근접) min/중앙값 {dmin.min():.2f}/{np.median(dmin):.2f} (단위=정규화, bbox≈1)")
        keep_e = (t0 > 0) & (dmin < 1.2)             # 객체 bbox 근처를 실제로 지나는 광선만
        EOn, ED = EOn[keep_e], ED[keep_e]
        print(f"empty ray 필터: {int(keep_e.sum())}/{len(keep_e)} 유지 (bbox 관통 광선)")

    # 2) SDF 학습
    print("IGR SDF 학습 ...")
    net = train_sdf(Pn, N, On, EOn, ED, args)

    # 3) 그리드 평가
    G = args.grid if args.grid > 0 else int(round(2 * scale / args.voxel_size))
    G = int(min(G, args.max_grid))
    print(f"SDF 그리드 평가 (G={G}, voxel≈{2*scale/(G-1):.4f} world) + marching cubes ...")
    lin = np.linspace(-1, 1, G, dtype=np.float32)
    net.eval()
    vol = np.empty((G, G, G), np.float32)
    with torch.no_grad():
        gx, gy = np.meshgrid(lin, lin, indexing="ij")
        for k in range(G):
            pts = np.stack([gx, gy, np.full_like(gx, lin[k])], -1).reshape(-1, 3)
            s = net(torch.tensor(pts, dtype=torch.float32, device="cuda")).cpu().numpy().reshape(G, G)
            vol[:, :, k] = s

    # 4) 전체 볼륨 marching cubes → '메쉬 단계' 트리밍
    #    (복셀 마스킹은 마스크 경계에 인위적 zero-crossing → 계단/큐브 아티팩트를 만들어 폐기.
    #     대신 관측 점군에서 mask_dist 초과인 정점을 메쉬에서 제거 — 경계가 표면을 따라감)
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(2.0 / (G - 1),) * 3)
    verts = verts - 1.0
    verts = verts * scale + center

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

    # 색: 최근접 관측점 (트리밍 후 정점 기준)
    verts2 = np.asarray(mesh.vertices)
    _, ni = tree.query(verts2, workers=-1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C[ni], 0, 1))
    mesh.compute_vertex_normals()

    # 5) safe_post_process_mesh(num_cluster) — TSDF 경로와 동일 로직(클램프 추가)

    out = args.out
    if not out:
        train_dir = os.path.join(args.model_path, "train", f"ours_{scene.loaded_iter}")
        os.makedirs(train_dir, exist_ok=True)
        out = os.path.join(train_dir, "sdf_fuse.ply")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"mesh saved at {out}  verts {len(verts)} faces {len(faces)}")

    mesh_post = safe_post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    out_post = out.replace(".ply", "_post.ply")
    o3d.io.write_triangle_mesh(out_post, mesh_post)
    print(f"mesh post processed saved at {out_post}")


if __name__ == "__main__":
    main()
