#
# RefineGS - gaussian_renderer/__init__.py
# ---------------------------------------------------------------------------
# BASE:  2D Gaussian Splatting (hbb1/2d-gaussian-splatting) renderer
# GRAFT: Split&Splat 2-pass 인스턴스 마스크 렌더링 + id_filter / mask_only
#
# 핵심
#   - RGB pass(B): 2DGS surfel rasterizer → rendered_image + allmap
#       allmap 레이아웃(공식 2DGS):
#         [0:1] depth(누적, alpha로 나눠 expected),  [1:2] alpha,
#         [2:5] normal(view space), [5:6] median depth, [6:7] dist(distortion)
#   - mask pass(A): [S&S] 인스턴스 id 를 색으로 인코딩(get_id_color)해 한 번 더 렌더
#       → mask_image. 2DGS rasterizer 도 shs/colors_precomp 를 받으므로 그대로 동작.
#       opacity 는 RGB pass 와 동일(학습 시 마스크 감독). α=1 실루엣(3.2.2)은 별도 함수.
#
# 의존성: utils.point_utils.depth_to_normal  →  utils/ 는 2DGS base 여야 함.
#
# 변경/추가 지점은 "# [S&S]" 주석으로 표시.
# ---------------------------------------------------------------------------

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None,
           id_filter=None, mask_only=False):   # [S&S] 인스턴스 필터 인자
    """
    Render the scene (2DGS surfel) + instance-id mask (Split&Splat 2-pass).
    Background tensor (bg_color) must be on GPU!
    """
    # [S&S] 선택적 인스턴스 필터링
    if mask_only:
        pc = pc.filter_points()
    if id_filter is not None:
        pc = pc.filter_by_id(id_filter)

    # screen-space means (densification 용 gradient). 두 pass 가 공유.
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # [2DGS] surfel rasterizer 세팅 (antialiasing 인자 없음 — 3DGS-accel 전용)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # [2DGS] scaling/rotation → surfel covariance (또는 precomputed)
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # precomputed covariance 사용 시 normal consistency loss 미지원(2DGS 주석)
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, far - near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # ---- RGB 색상 (SH 또는 precomputed) ----
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # ============ pass B: RGB + geometry ============
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # ============ [S&S] pass A: instance-id mask ============
    # id 를 색(DC)으로 인코딩한 feature 를 같은 geometry/opacity 로 렌더.
    # gradient 는 id(상수)로 흐르지 않고 geometry/opacity 로 흘러 마스크를 GT 에 맞춤.
    mask_image = None
    try:
        mask_shs = pc.get_id_color
    except Exception:
        mask_shs = None
    if mask_shs is not None:
        mask_image, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=mask_shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        mask_image = mask_image.clamp(0, 1)   # 마스크 손실 안정성

    rets = {
        "render": rendered_image,
        "mask": mask_image,                       # [S&S]
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,           # [2DGS] boolean (train.py densify 와 일치)
        "radii": radii,
    }

    # ============ [2DGS] allmap → 정규화용 부가 출력 ============
    render_alpha = allmap[1:2]

    # normal: view space → world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)

    # median depth
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # expected depth = 누적 depth / alpha
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # depth distortion
    render_dist = allmap[6:7]

    # surf depth: depth_ratio 로 expected(0)~median(1) 선택
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median

    # pseudo surface normal (depth 로부터 유도) — normal consistency loss 용
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
        "depth": surf_depth,          # [S&S 호환 키] train.py depth 감독에서 사용
        "rend_alpha": render_alpha,
        "rend_normal": render_normal,
        "rend_dist": render_dist,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
    })
    return rets


def render_silhouette(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor,
                      scaling_modifier=1.0, id_filter=None):
    """
    [S&S 3.2.2] full-opacity(α=1) 실루엣 렌더링 → M^gs.
    2DGS 에서는 pixels(pixel→gauss idx) 트릭 대신 누적 alpha 를 실루엣으로 사용.
    반환: {"mask": alpha[H,W] in [0,1], "depth": surf_depth, "render": rgb}

    주의: 기존 utils_mask/mask_optimizer.py 는 S&S render_simple 의 'pixels' 출력에
    의존하므로, 그 경로는 별도 적응이 필요(가이드 §4.4). 본 함수는 alpha 기반 대체.
    """
    if id_filter is not None:
        pc = pc.filter_by_id(id_filter, keep_occlusions=False)

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=False, device="cuda")
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    full_opacity = torch.ones_like(pc.get_opacity)   # [S&S] α=1 강제
    rendered_image, radii, allmap = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=full_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    render_alpha = allmap[1:2]
    render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    return {
        "render": rendered_image,
        "mask": render_alpha.clamp(0, 1),   # α=1 누적 → solid 실루엣
        "depth": render_depth_expected,
        "radii": radii,
    }
