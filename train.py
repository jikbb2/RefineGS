################################################################################
# RefineGS - train.py
# ---------------------------------------------------------------------------
# 머지 방향: 2DGS base + Split&Splat graft (네 RefineGS 방향과 동일)
#   - loop / mask loss / composition / image_filter 구조는 [S&S] 고유 → 보존
#   - depth distortion + normal consistency 정규화는 [2DGS] → graft
#   - [제거] SparseGaussianAdam, exposure, separate_sh, inverse-depth 감독, antialiasing
#
# [v2 패치 — 원자료 기하 감독]
#   (a) GT-depth 손실 활성화: --gt_depth_dir 로 외부 depth 폴더 지정 가능,
#       frame↔depth 이름 규약 자동 매칭 (기존엔 <source>/depths 없으면 무음 skip 이었음)
#   (b) 비대칭 depth 손실: 렌더 depth 가 GT 보다 '앞'(=카메라~표면 사이에 질량, free-space 위반)
#       이면 --front_mult 배 강벌점 → 학습 단계 carving
#   (c) NV normal 감독: novelview_dir 에 normal_%04d.png(단안 추정, camera-space) 있으면
#       weight ⊙ (1-|cos|) 항 추가 — 생성 뷰의 RGB 대신 방향 정보로 기하 감독
#
# 각 변경 블록에 base 표시: [S&S] / [2DGS] / [제거] / [v2]
# 원저작권: graphdeco-inria 3DGS, 2DGS(hbb1), Split&Splat(LTTM)
################################################################################

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils_mask.mask_filters import image_filter   # [S&S] 빈 뷰 정리

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

# [제거] SparseGaussianAdam (diff_gaussian_rasterization) — 2DGS 는 plain Adam



# === [RefineGS depth supervision, v2] =================================
_DEPTH_CACHE = {}
def _load_gt_depth(cam, source_path, scale=6553.5, override_dir=None):
    """GT metric depth(meters) + (객체∩유효) 마스크. 캐시.
    탐색: override_dir → <source>/depths.  이름: <stem>.png → frame↔depth 치환."""
    import os, cv2, numpy as np, torch
    key = getattr(cam, "image_name", None)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    stem = os.path.splitext(key)[0] if key else None
    p = None
    if stem:
        dirs = ([override_dir] if override_dir else []) + [os.path.join(source_path, "depths")]
        names = [stem + ".png", stem.replace("frame", "depth") + ".png"]
        for d in dirs:
            if not d:
                continue
            for nm in names:
                c = os.path.join(os.path.expanduser(d), nm)
                if os.path.exists(c):
                    p = c
                    break
            if p:
                break
    if not p:
        _DEPTH_CACHE[key] = (None, None); return None, None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / scale  # meters
    d = cv2.resize(d, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
    gd = torch.from_numpy(d[None]).float().cuda()
    am = getattr(cam, "alpha_mask", None)
    am = am if am is not None else torch.ones_like(gd)
    valid = ((gd > 1e-3) & (am > 0.5)).float()
    _DEPTH_CACHE[key] = (gd, valid)
    return gd, valid
# =====================================================================

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # [S&S] composition 여부로 SH degree 결정
    active_sh_degree = 3 if args.composition else 0
    gaussians = GaussianModel(dataset.sh_degree, active_sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    if getattr(args, 'init_ply', None):  # [B4a] 조립 ply 로 init 덮어쓰기
        gaussians.load_ply(args.init_ply)
        gaussians.active_sh_degree = gaussians.max_sh_degree
        print('[B4a] init from ' + args.init_ply + ': ' + str(gaussians.get_xyz.shape[0]) + ' gaussians')
    gaussians.training_setup(opt)
    # [NV] novel-view soft-weighted supervision 로드 (+ [v2] normal_%04d.png 선택 로드)
    _NV_CAMS, _NV_LAMBDA, _NV_EVERY = [], float(getattr(args, "nv_lambda", 0.5)), int(getattr(args, "nv_every", 2))
    _NV_LAMBDA_N = float(getattr(args, "nv_lambda_normal", 0.0))
    _NV_LAMBDA_D = float(getattr(args, "nv_lambda_depth", 0.0))
    if getattr(args, "novelview_dir", None):
        import os as _os, numpy as _np, torch as _t
        from PIL import Image as _Img
        from scene.cameras import MiniCam as _MiniCam
        _nvd = args.novelview_dir
        _recs = _np.load(_os.path.join(_nvd, "poses.npz"), allow_pickle=True)["records"]
        _n_nrm = _n_dep = 0
        for _r in _recs:
            _r = _r.item() if hasattr(_r, "item") and not isinstance(_r, dict) else _r
            _i = int(_r["idx"])
            _gp = _os.path.join(_nvd, "gen_%04d.jpg" % _i)
            _wp = _os.path.join(_nvd, "weight_%04d.png" % _i)
            if not _os.path.exists(_gp) or not _os.path.exists(_wp):
                continue
            _wvt = _t.tensor(_np.asarray(_r["world_view_transform"]), dtype=_t.float32).cuda()
            _fpt = _t.tensor(_np.asarray(_r["full_proj_transform"]), dtype=_t.float32).cuda()
            _cam = _MiniCam(int(_r["width"]), int(_r["height"]), float(_r["FoVy"]), float(_r["FoVx"]),
                            0.01, 100.0, _wvt, _fpt)
            _g = _t.from_numpy(_np.asarray(_Img.open(_gp).convert("RGB"))).float().permute(2, 0, 1).cuda() / 255.0
            _w = _t.from_numpy(_np.asarray(_Img.open(_wp).convert("L"))).float().cuda() / 255.0
            _cam.gt_image = _g
            _cam.weight = _w[None]
            # [v2] 단안 추정 normal (camera-space, png [0,255]→[-1,1])
            _npn = _os.path.join(_nvd, "normal_%04d.png" % _i)
            _cam.gt_normal = None
            if _NV_LAMBDA_N > 0 and _os.path.exists(_npn):
                _n = _t.from_numpy(_np.asarray(_Img.open(_npn).convert("RGB"))).float().permute(2, 0, 1).cuda()
                _cam.gt_normal = _t.nn.functional.normalize(_n / 127.5 - 1.0, dim=0)
                _n_nrm += 1
            # [v3] 생성 뷰 depth (make_gen_points.py, 스케일 정렬됨. 0=무효)
            _dpn = _os.path.join(_nvd, "depth_%04d.npy" % _i)
            _cam.gt_depth_nv = None
            if _NV_LAMBDA_D > 0 and _os.path.exists(_dpn):
                _d = _t.from_numpy(_np.load(_dpn).astype(_np.float32))[None].cuda()
                _cam.gt_depth_nv = _d
                _n_dep += 1
            _NV_CAMS.append(_cam)
        print("[NV] %d novel-view cams (lambda=%.3f every=%d, normal %d뷰 λ=%.2f, depth %d뷰 λ=%.2f) from %s"
              % (len(_NV_CAMS), _NV_LAMBDA, _NV_EVERY, _n_nrm, _NV_LAMBDA_N, _n_dep, _NV_LAMBDA_D, _nvd))
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0     # [2DGS]
    ema_normal_for_log = 0.0   # [2DGS]
    ema_gtd_for_log = 0.0      # [v2]

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # 1000 it 마다 SH degree 상승
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 랜덤 카메라 선택
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        mask = render_pkg["mask"]                              # [S&S]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]   # [2DGS] boolean
        radii = render_pkg["radii"]

        # ---- [S&S] mask loss ----
        GT_mask = viewpoint_cam.original_mask.cuda()
        Ll1_mask = l1_loss(mask, GT_mask)

        # ---- RGB loss (3DGS lineage) ----
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # [S&S] mask 가중치 0.25
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + Ll1_mask * 0.25

        # ---- [2DGS] regularization: depth distortion + normal consistency ----
        lambda_normal = opt.lambda_normal if iteration > 1500 else 0.0
        lambda_dist   = opt.lambda_dist   if iteration > 500  else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * rend_dist.mean()

        total_loss = loss + dist_loss + normal_loss

        # [v2] (a)+(b) 비대칭 GT-depth 손실 — front(카메라~GT표면 사이 질량)=free-space 위반 강벌점
        _gtd_val = 0.0
        _ld = float(getattr(args, "lambda_gtdepth", 0.5)) if iteration > 500 else 0.0
        if _ld > 0 and ('depth' in render_pkg):
            _gd, _vm = _load_gt_depth(viewpoint_cam, dataset.source_path,
                                      scale=float(getattr(args, "gt_depth_scale", 6553.5)),
                                      override_dir=getattr(args, "gt_depth_dir", None))
            if _gd is not None and _vm.sum() > 0:
                _rd = render_pkg['depth']
                if _rd.dim() == 2: _rd = _rd[None]
                _diff = _rd - _gd
                _front = torch.relu(-_diff)              # 렌더가 GT 앞 → carving 벌점
                _back = torch.relu(_diff)
                _fm = float(getattr(args, "front_mult", 3.0))
                _dl = ((_fm * _front + _back) * _vm).sum() / _vm.sum().clamp_min(1.0)
                total_loss = total_loss + _ld * _dl
                _gtd_val = _dl.item()

        # [NV] novel-view weighted supervision (+ [v2] normal 항)
        if _NV_CAMS and (iteration % _NV_EVERY == 0):
            from random import randint as _ri
            _nv = _NV_CAMS[_ri(0, len(_NV_CAMS) - 1)]
            _pkg = render(_nv, gaussians, pipe, bg)
            _nvr = _pkg["render"]
            _w = _nv.weight
            if _w.shape[-2:] != _nvr.shape[-2:]:
                _w = torch.nn.functional.interpolate(_w[None], _nvr.shape[-2:], mode="bilinear")[0]
            _gt = _nv.gt_image
            if _gt.shape[-2:] != _nvr.shape[-2:]:
                _gt = torch.nn.functional.interpolate(_gt[None], _nvr.shape[-2:], mode="bilinear")[0]
            _nv_l1 = (torch.abs(_nvr - _gt) * _w).sum() / _w.sum().clamp_min(1.0)
            total_loss = total_loss + _NV_LAMBDA * _nv_l1
            # [v2] (c) normal 감독: world→camera 회전 후 방향 일치 (1-|cos| — 부호 규약 무관)
            if _NV_LAMBDA_N > 0 and getattr(_nv, "gt_normal", None) is not None:
                _rn = _pkg["rend_normal"]                                    # (3,H,W) world
                _Rw2c = _nv.world_view_transform[:3, :3].T                   # wvt=w2c^T → R_w2c
                _rn_c = torch.einsum('ij,jhw->ihw', _Rw2c, _rn)
                _rn_c = torch.nn.functional.normalize(_rn_c, dim=0)
                _gn = _nv.gt_normal
                if _gn.shape[-2:] != _rn_c.shape[-2:]:
                    _gn = torch.nn.functional.normalize(
                        torch.nn.functional.interpolate(_gn[None], _rn_c.shape[-2:], mode="bilinear")[0], dim=0)
                _cos = (_rn_c * _gn).sum(0, keepdim=True).abs()
                _nl = ((1.0 - _cos) * _w).sum() / _w.sum().clamp_min(1.0)
                total_loss = total_loss + _NV_LAMBDA_N * _nl
            # [v3] 생성 뷰 depth 감독 — 생성 영역(weight>0 ∧ gt_depth>0)에서 렌더 depth L1
            if _NV_LAMBDA_D > 0 and getattr(_nv, "gt_depth_nv", None) is not None and ("depth" in _pkg):
                _rd = _pkg["depth"]
                if _rd.dim() == 2: _rd = _rd[None]
                _gd_nv = _nv.gt_depth_nv
                if _gd_nv.shape[-2:] != _rd.shape[-2:]:
                    _gd_nv = torch.nn.functional.interpolate(_gd_nv[None], _rd.shape[-2:], mode="nearest")[0]
                _vm_nv = ((_gd_nv > 1e-3).float() * _w)
                if _vm_nv.sum() > 0:
                    _dl_nv = (torch.abs(_rd - _gd_nv) * _vm_nv).sum() / _vm_nv.sum().clamp_min(1.0)
                    total_loss = total_loss + _NV_LAMBDA_D * _dl_nv
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_gtd_for_log = 0.4 * _gtd_val + 0.6 * ema_gtd_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "dist": f"{ema_dist_for_log:.5f}",
                    "normal": f"{ema_normal_for_log:.5f}",
                    "gtd": f"{ema_gtd_for_log:.4f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log & save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations,
                            scene, render, (pipe, background), dataset.train_test_exp)
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # ---- Densification ----
            if iteration < opt.densify_until_iter:          # [2DGS]
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull,
                                                scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter) or \
                   (scene.composition and iteration == 0):   # [S&S]
                    gaussians.reset_opacity()

            # ---- Optimizer step ----
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # [S&S] per-instance 학습 종료 후 빈/검은 뷰 정리
    if not scene.composition:
        black_cameras = scene.getBlackCameras()
        print("Empty views: ", len(black_cameras))
        try:
            image_filter(gaussians, black_cameras, pipe, dataset)
        except Exception as e:
            print(f"[warn] image_filter skipped (renderer 적응 필요): {e}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, 35, 5)]},
        )
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(out["render"], 0.0, 1.0)
                    mask = torch.clamp(out["mask"], 0.0, 1.0) if out["mask"] is not None else None
                    depth = out["depth"]

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = viewpoint.original_mask.cuda()

                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                        gt_mask = gt_mask[..., gt_mask.shape[-1] // 2:]

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if mask is not None:
                            tb_writer.add_images(config['name'] + "_mask_{}/render".format(viewpoint.image_name), mask[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_depth_{}/render".format(viewpoint.image_name), depth[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_mask_{}/ground_truth".format(viewpoint.image_name), gt_mask[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--novelview_dir", type=str, default=None)  # [NV]
    parser.add_argument("--nv_lambda", type=float, default=0.5)      # [NV]
    parser.add_argument("--nv_every", type=int, default=2)           # [NV]
    parser.add_argument("--init_ply", type=str, default=None)  # [B4a]
    # [v2] 원자료 기하 감독
    parser.add_argument("--gt_depth_dir", type=str, default=None,
                        help="GT depth 폴더(예: nice-slam results). 미지정 시 <source>/depths 탐색")
    parser.add_argument("--gt_depth_scale", type=float, default=6553.5)
    parser.add_argument("--lambda_gtdepth", type=float, default=0.5, help="0=off")
    parser.add_argument("--front_mult", type=float, default=3.0,
                        help="렌더 depth < GT depth (free-space 위반) 벌점 배율")
    parser.add_argument("--nv_lambda_normal", type=float, default=0.0,
                        help=">0: novelview_dir 의 normal_%%04d.png 로 NV normal 감독")
    parser.add_argument("--nv_lambda_depth", type=float, default=0.0,
                        help=">0: novelview_dir 의 depth_%%04d.npy(make_gen_points, 스케일 정렬)로 NV depth 감독")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
