################################################################################
# RefineGS - train.py
# ---------------------------------------------------------------------------
# 머지 방향: 2DGS base + Split&Splat graft (네 RefineGS 방향과 동일)
#   - loop / mask loss / composition / image_filter 구조는 [S&S] 고유 → 보존
#   - depth distortion + normal consistency 정규화는 [2DGS] → graft
#   - [제거] SparseGaussianAdam, exposure, separate_sh, inverse-depth 감독, antialiasing
#
# 각 변경 블록에 base 표시: [S&S] / [2DGS] / [제거]
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



# === [RefineGS depth supervision] ====================================
_DEPTH_CACHE = {}
def _load_gt_depth(cam, source_path, scale=6553.5):
    """GT metric depth(meters) + (객체∩유효) 마스크. 캐시."""
    import os, cv2, numpy as np, torch
    key = getattr(cam, "image_name", None)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    stem = os.path.splitext(key)[0] if key else None
    p = os.path.join(source_path, "depths", stem + ".png") if stem else None
    if not p or not os.path.exists(p):
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
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # [제거] depth_l1_weight (inverse-depth 감독) — 필요 시 직접 depth 감독으로 재도입(가이드 §5.3)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0     # [2DGS]
    ema_normal_for_log = 0.0   # [2DGS]

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        # ---- network gui (선택) ----
        # network gui 비활성화 (headless + 2DGS network_gui 시그니처 상이 → 뷰어 미사용)

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

        # [변경] render 호출에서 use_trained_exp/separate_sh 제거
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

        # [S&S] mask 가중치 0.25 (composition 시 sed 로 0.05/0.1/0.25 치환 — README 그대로 동작)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + Ll1_mask * 0.25

        # ---- [2DGS] regularization: depth distortion + normal consistency ----
#         lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
#         lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        lambda_normal = opt.lambda_normal if iteration > 1500 else 0.0
        lambda_dist   = opt.lambda_dist   if iteration > 500  else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * rend_dist.mean()
        # (선택, 가이드 §5.3) 경계 보호: 기하 손실을 마스크 내부로 한정하려면
        #   m = GT_mask[:1] if GT_mask.dim()==3 else GT_mask
        #   normal_loss = lambda_normal * (normal_error * m).mean(); dist_loss = lambda_dist * (rend_dist * m).mean()

        total_loss = loss + dist_loss + normal_loss
        # [RefineGS depth supervision] GT metric depth L1 (마스크 내부)
        _ld = 0.5 if iteration > 500 else 0.0
        if _ld > 0 and ('depth' in render_pkg):
            _gd, _vm = _load_gt_depth(viewpoint_cam, dataset.source_path)
            if _gd is not None and _vm.sum() > 0:
                _rd = render_pkg['depth']
                if _rd.dim() == 2: _rd = _rd[None]
                _dl = (torch.abs(_rd - _gd) * _vm).sum() / _vm.sum().clamp_min(1.0)
                total_loss = total_loss + _ld * _dl
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "dist": f"{ema_dist_for_log:.5f}",
                    "normal": f"{ema_normal_for_log:.5f}",
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
            if iteration < opt.densify_until_iter:          # [2DGS] 단순 조건 (S&S 의 max_num_splats cap 제거)
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # [2DGS] densify_and_prune 4-arg (S&S 의 trailing radii 제거)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull,
                                                scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter) or \
                   (scene.composition and iteration == 0):   # [S&S] composition 시작 시 opacity reset
                    gaussians.reset_opacity()

            # ---- Optimizer step ([제거] sparse adam / exposure) ----
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # [S&S] per-instance 학습 종료 후 빈/검은 뷰 정리
    # ⚠️ image_filter 가 내부에서 render 를 호출하면 변경된 시그니처/2DGS 출력과 안 맞을 수 있음.
    #    첫 동작 검증 때 깨지면 try/except 로 감싸거나 잠시 주석 처리 (가이드 §4.4).
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
                    # [변경] render 호출 단순화 (separate_sh/use_trained_exp 제거)
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

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
