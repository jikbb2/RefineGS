"""Fine-tune an object's gaussians with prior-mesh depth as geometric supervision.

    L = L_photo(training view)  +  ld * L_depth(novel pose, unseen pixels only)
                                +  la * L_alpha(novel pose, unseen pixels only)

The photometric term holds the observed side in place; the depth term is the only
signal acting on the unobserved side, which is why the prior gaussians must be
injected first -- they are what the depth gradient has to push around. Starting from
the bare checkpoint leaves nothing under those pixels and the depth term does nothing.

No densification and no pruning: this refines gaussians that already exist. Pruning by
opacity would delete exactly the injected ones, which no training view supports.

Input and output follow the repo's model-dir convention, so inject_prior_gaussians.py
feeds this directly and mesh_from_gaussians.py consumes the result with -m.
"""
import os, argparse, functools, random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as tf

print = functools.partial(print, flush=True)

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import focal2fov, getProjectionMatrix


class NovelCam:
    """Only the fields the rasteriser reads.

    Forks disagree on Camera.__init__ (image as tensor vs PIL, with or without a
    resolution argument), and a novel pose has no image at all, so building the
    matrices here avoids depending on any of that.
    """

    def __init__(self, R_w2c, t_w2c, K, H, W, device, znear=0.01, zfar=100.0):
        self.image_width, self.image_height = int(W), int(H)
        self.FoVx = focal2fov(float(K[0, 0]), W)
        self.FoVy = focal2fov(float(K[1, 1]), H)
        self.znear, self.zfar = znear, zfar
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3], w2c[:3, 3] = R_w2c, t_w2c
        # stored transposed, as getWorld2View2 leaves them
        self.world_view_transform = torch.tensor(w2c, device=device).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear, zfar, self.FoVx, self.FoVy).to(device).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0)
                                    .bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]


def load_novel(path, device):
    z = np.load(os.path.expanduser(path))
    H, W = int(z["H"]), int(z["W"])
    cams = [NovelCam(z["R"][i], z["t"][i], z["K"], H, W, device) for i in range(len(z["R"]))]
    d = torch.from_numpy(z["depth"]).float().to(device)                  # (N,H,W)
    m = torch.from_numpy(z["unseen"]).to(device).bool() & (d > 0)        # hit AND unobserved
    print(f"[novel] {len(cams)} poses {W}x{H}  supervised {m.float().mean()*100:.1f}% of pixels")
    return cams, d, m


def fit_to(x, H, W):
    """Nearest only: the prior depth is 0 where the mesh was missed, and interpolating
    across that boundary would invent depths between a surface and nothing."""
    if tuple(x.shape[-2:]) == (H, W):
        return x
    y = tf.interpolate(x[None, None].float(), size=(H, W), mode="nearest")[0, 0]
    return y.bool() if x.dtype == torch.bool else y


def load_masks(cams, root, device):
    """Object masks for the photometric term.

    objects_voted/<gid> is a label slice of the SCENE model, so its cfg_args carries the
    scene's source_path and every room frame. Rendering one object against a full-room
    photo pins L_photo at a constant ~0.73 whose gradient asks the object's gaussians to
    explain the background. make_object_dirs.py symlinks the room frames unmasked, so
    restricting the view list is not enough -- both sides have to be masked here.
    """
    keep, M, miss = [], [], []
    for c in cams:
        p = os.path.join(root, c.image_name + ".png")
        if not os.path.exists(p):
            miss.append(c.image_name); continue
        a = torch.from_numpy(np.array(Image.open(p).convert("L")))[None, None].float()
        a = tf.interpolate(a, size=(c.image_height, c.image_width), mode="nearest")
        keep.append(c); M.append((a[0, 0] > 127).to(device))
    assert len(keep) >= 10, (
        f"only {len(keep)} of {len(cams)} views have a mask under {root}. "
        f"Point -s at the per-object dataset (data/<scene>/masks/<gid>), not the scene.")
    if miss:
        print(f"[photo] {len(miss)} views without a mask, dropped")
    cov = float(torch.stack([m.float().mean() for m in M]).mean()) * 100
    print(f"[photo] masked by {root}   {len(keep)} views, object covers {cov:.2f}% of a frame")
    return keep, M


def photo_loss(pkg, gt, m, opt):
    if m is None:
        return (1.0 - opt.lambda_dssim) * l1_loss(pkg["render"], gt) \
            + opt.lambda_dssim * (1.0 - ssim(pkg["render"], gt))
    # No SSIM here: masking cuts a hard edge into both images that SSIM reads as
    # structure to match. L1 inside the mask is the honest term.
    w = m.float()[None]
    return ((pkg["render"] - gt).abs() * w).sum() / (w.sum() * 3.0)


def novel_loss(pkg, d_ref, mask, args):
    """Depth + opacity on pixels the prior covers and no training view saw.

    L_alpha matters more than it looks. Where the unobserved side is empty the render
    is transparent, surf_depth is ~0, and |0 - d_ref| is large -- but there is no
    gaussian there to receive that gradient. Pushing alpha to 1 acts on the injected
    gaussians' opacity and scale directly, which is what makes them stick.
    """
    n = mask.sum()
    if n < args.min_sup_px:
        return None, 0.0, 0.0
    m = mask.float()
    e = (pkg["surf_depth"].squeeze(0) - d_ref).abs()
    # Huber: the prior is only approximately right, so outliers must not dominate
    h = args.huber
    ld = ((torch.where(e < h, 0.5 * e ** 2 / h, e - 0.5 * h)) * m).sum() / n
    la = ((1.0 - pkg["rend_alpha"].squeeze(0)).abs() * m).sum() / n
    return args.lambda_depth * ld + args.lambda_alpha * la, ld.item(), la.item()


def main():
    ap = argparse.ArgumentParser()
    # sentinel=True makes every ModelParams default None. get_combined_args overwrites
    # a cfg_args value with any command-line value that is not None, so without it the
    # empty default source_path wins and Scene looks for sparse/ in the cwd.
    mp, pp, op = ModelParams(ap, sentinel=True), PipelineParams(ap), OptimizationParams(ap)
    ap.add_argument("--prior_depth", required=True, help="npz from make_prior_depth.py")
    ap.add_argument("--start_ply", default="",
                    help="ply to start from; default = -m's checkpoint. Point this at "
                         "inject_prior_gaussians.py's output -- without injected "
                         "gaussians the depth term has nothing to act on.")
    ap.add_argument("--load_iteration", default=-1, type=int)
    ap.add_argument("--out", required=True, help="model dir, written like the repo's")
    ap.add_argument("--save_iteration", default=30000, type=int,
                    help="iteration folder to write, so downstream defaults still apply")
    ap.add_argument("--iters", default=3000, type=int)
    ap.add_argument("--lr_scale", default=0.1, type=float,
                    help="the checkpoint has converged, so the training-time learning "
                         "rates would undo it. Scales every rate.")
    ap.add_argument("--lambda_depth", default=1.0, type=float)
    ap.add_argument("--lambda_alpha", default=0.1, type=float)
    ap.add_argument("--huber", default=0.05, type=float, help="metres")
    ap.add_argument("--warmup", default=200, type=int, help="photo-only iters first")
    ap.add_argument("--min_sup_px", default=200, type=int)
    ap.add_argument("--freeze_xyz", action="store_true",
                    help="opacity/scale/rotation only; positions stay on the prior surface")
    ap.add_argument("--masks", default="",
                    help="per-view object masks; default <source_path>/masks. "
                         "Pass 'none' to disable (only correct for a model trained "
                         "on this exact image set).")
    ap.add_argument("--log_every", default=250, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = get_combined_args(ap)      # source_path and resolution come from -m's cfg_args
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    dev = torch.device("cuda")
    dataset, pipe, opt = mp.extract(args), pp.extract(args), op.extract(args)
    assert dataset.source_path and os.path.isdir(dataset.source_path), (
        f"source_path did not resolve ({dataset.source_path!r}). "
        f"{args.model_path}/cfg_args is missing or unreadable; pass -s explicitly.")
    print(f"[cfg] source {dataset.source_path}  resolution {dataset.resolution}")
    opt.iterations = opt.position_lr_max_steps = args.iters
    for k in ("position_lr_init", "position_lr_final", "feature_lr", "opacity_lr",
              "scaling_lr", "rotation_lr"):
        setattr(opt, k, getattr(opt, k) * args.lr_scale)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    if args.start_ply:
        gaussians.load_ply(os.path.expanduser(args.start_ply))
        print(f"[init] {args.start_ply}")
    # Scene sets spatial_lr_scale only on the create_from_pcd path; loading a ply leaves
    # it at 0, which silently zeroes the position learning rate.
    gaussians.spatial_lr_scale = scene.cameras_extent
    gaussians.training_setup(opt)
    if args.freeze_xyz:
        gaussians._xyz.requires_grad_(False)

    novel, d_all, m_all = load_novel(args.prior_depth, dev)
    train = scene.getTrainCameras().copy()
    mroot = args.masks or os.path.join(dataset.source_path, "masks")
    if mroot == "none":
        masks = [None] * len(train)
        print("[photo] unmasked -- correct only if -m was trained on these images")
    else:
        train, masks = load_masks(train, os.path.expanduser(mroot), dev)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device=dev)
    print(f"[init] {gaussians.get_xyz.shape[0]:,} gaussians  {len(train)} train views  "
          f"extent {scene.cameras_extent:.2f}m  lr x{args.lr_scale}")

    pool, npool, acc = [], [], np.zeros(4)
    for it in range(1, args.iters + 1):
        gaussians.update_learning_rate(it)
        if not pool:
            pool = list(range(len(train))); random.shuffle(pool)
        i = pool.pop()
        pkg = render(train[i], gaussians, pipe, bg)
        lp = photo_loss(pkg, train[i].original_image.to(dev), masks[i], opt)
        loss, ld, la = lp, 0.0, 0.0

        if it > args.warmup:
            if not npool:
                npool = list(range(len(novel))); random.shuffle(npool)
            j = npool.pop()
            npkg = render(novel[j], gaussians, pipe, bg)
            h, w = npkg["surf_depth"].shape[-2:]
            ln, ld, la = novel_loss(npkg, fit_to(d_all[j], h, w), fit_to(m_all[j], h, w), args)
            if ln is not None:
                loss = loss + ln

        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        acc += (lp.item(), ld, la, 1)

        if it % args.log_every == 0 or it == args.iters:
            p, dd, aa, n = acc
            print(f"[{it:5d}/{args.iters}] photo {p/n:.4f}  depth {dd/n:.4f}m  alpha {aa/n:.4f}")
            acc[:] = 0

    dst = os.path.join(os.path.expanduser(args.out), "point_cloud",
                       f"iteration_{args.save_iteration}")
    os.makedirs(dst, exist_ok=True)
    gaussians.save_ply(os.path.join(dst, "point_cloud.ply"))
    print(f"[out] {dst}/point_cloud.ply   {gaussians.get_xyz.shape[0]:,} gaussians")


if __name__ == "__main__":
    main()