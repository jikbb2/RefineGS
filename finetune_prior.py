"""Fine-tune an object's gaussians with prior-mesh depth as geometric supervision.

    L = L_photo(training view)  +  ld * L_depth(novel pose, unseen pixels only)
                                +  la * L_alpha(novel pose, unseen pixels only)

The photometric term holds the observed side in place; the depth term is the only
signal acting on the unobserved side, which is why the prior gaussians must be
injected first (they are what the depth gradient has to push around).

  python finetune_prior.py -s <obj_colmap> -m <objects_voted/6> \
      --prior_depth /tmp/pd6.npz --start_ply <injected.ply> --out <run_dir>
"""
import os, sys, inspect, argparse, functools, random
import numpy as np
import torch
import torch.nn.functional as tf

print = functools.partial(print, flush=True)

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import focal2fov


# ---------------------------------------------------------------- novel cameras
def make_camera(R_w2c, t_w2c, K, H, W, uid, device):
    """Build a Camera the renderer accepts, binding only arguments it declares.

    3DGS/2DGS store R as the camera-to-world rotation (getWorld2View transposes it
    back) and T as the world-to-camera translation, which is what make_prior_depth
    writes out as R_w2c / t_w2c.
    """
    kw = dict(
        colmap_id=uid, uid=uid, R=R_w2c.T, T=t_w2c,
        FoVx=focal2fov(float(K[0, 0]), W), FoVy=focal2fov(float(K[1, 1]), H),
        image=torch.zeros(3, H, W, device=device),
        gt_alpha_mask=None, image_name=f"novel_{uid:04d}",
        depth=None, invdepthmap=None, depth_params=None,
        data_device=str(device), resolution=(W, H),
    )
    ok = set(inspect.signature(Camera.__init__).parameters) - {"self"}
    cam = Camera(**{k: v for k, v in kw.items() if k in ok})
    # Some forks size themselves from the image, others from explicit fields.
    for a, v in (("image_width", W), ("image_height", H)):
        if getattr(cam, a, None) != v:
            setattr(cam, a, v)
    return cam


def load_novel(path, device):
    z = np.load(path)
    R, t, K = z["R"], z["t"], z["K"]
    H, W = int(z["H"]), int(z["W"])
    d = torch.from_numpy(z["depth"]).float().to(device)          # (N,H,W)
    m = torch.from_numpy(z["unseen"]).bool().to(device)          # (N,H,W)
    print(f"[novel] {len(R)} poses  {W}x{H}  supervised {m.float().mean()*100:.1f}% of pixels")
    return R, t, K, H, W, d, m


def fit_to(x, H, W, mode):
    if x.shape[-2:] == (H, W):
        return x
    y = tf.interpolate(x[None, None].float(), size=(H, W), mode=mode)[0, 0]
    return y.bool() if x.dtype == torch.bool else y


# ---------------------------------------------------------------------- losses
def novel_loss(pkg, d_ref, mask, args):
    """Depth + opacity loss on pixels the prior covers and no training view saw."""
    if mask.sum() < args.min_sup_px:
        return None, 0.0, 0.0
    d = pkg["surf_depth"].squeeze(0)
    a = pkg["rend_alpha"].squeeze(0)
    m = mask.float()
    n = m.sum()
    # Huber: the prior is only approximately right, so cap the pull of outliers.
    e = (d - d_ref).abs()
    ld = (torch.where(e < args.huber, 0.5 * e ** 2 / args.huber, e - 0.5 * args.huber) * m).sum() / n
    la = ((1.0 - a).abs() * m).sum() / n
    return args.lambda_depth * ld + args.lambda_alpha * la, ld.item(), la.item()


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    mp, pp, op = ModelParams(ap), PipelineParams(ap), OptimizationParams(ap)
    ap.add_argument("--prior_depth", required=True, help="npz from make_prior_depth.py")
    ap.add_argument("--start_ply", default="", help="injected ply; default = checkpoint")
    ap.add_argument("--load_iteration", default=-1, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", default=3000, type=int)
    ap.add_argument("--lambda_depth", default=1.0, type=float)
    ap.add_argument("--lambda_alpha", default=0.1, type=float)
    ap.add_argument("--huber", default=0.05, type=float, help="metres")
    ap.add_argument("--novel_every", default=1, type=int)
    ap.add_argument("--warmup", default=200, type=int, help="photo-only iters first")
    ap.add_argument("--min_sup_px", default=200, type=int)
    ap.add_argument("--freeze_xyz", action="store_true",
                    help="optimise opacity/scale/rotation only; positions stay put")
    ap.add_argument("--log_every", default=250, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args(sys.argv[1:])
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    dev = torch.device("cuda")
    dataset, pipe, opt = mp.extract(args), pp.extract(args), op.extract(args)
    opt.iterations = args.iters

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    if args.start_ply:
        gaussians.load_ply(args.start_ply)
        print(f"[init] {args.start_ply}")
    gaussians.training_setup(opt)
    if args.freeze_xyz:
        gaussians._xyz.requires_grad_(False)
    print(f"[init] {gaussians.get_xyz.shape[0]} gaussians")

    R, t, K, Hn, Wn, d_all, m_all = load_novel(args.prior_depth, dev)
    novel = [make_camera(R[i], t[i], K, Hn, Wn, i, dev) for i in range(len(R))]

    train = scene.getTrainCameras().copy()
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device=dev)
    print(f"[init] {len(train)} training views, {len(novel)} novel poses")

    pool, npool, acc = [], [], np.zeros(4)
    for it in range(1, args.iters + 1):
        if not pool:
            pool = list(range(len(train))); random.shuffle(pool)
        cam = train[pool.pop()]
        pkg = render(cam, gaussians, pipe, bg)
        gt = cam.original_image.to(dev)
        lp = (1.0 - opt.lambda_dssim) * l1_loss(pkg["render"], gt) \
             + opt.lambda_dssim * (1.0 - ssim(pkg["render"], gt))
        loss, ld, la = lp, 0.0, 0.0

        if it > args.warmup and it % args.novel_every == 0:
            if not npool:
                npool = list(range(len(novel))); random.shuffle(npool)
            j = npool.pop()
            npkg = render(novel[j], gaussians, pipe, bg)
            h, w = npkg["surf_depth"].shape[-2:]
            ln, ld, la = novel_loss(npkg, fit_to(d_all[j], h, w, "bilinear"),
                                    fit_to(m_all[j], h, w, "nearest"), args)
            if ln is not None:
                loss = loss + ln

        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        acc += (lp.item(), ld, la, 1)

        if it % args.log_every == 0 or it == args.iters:
            p, dd, aa, n = acc
            print(f"[{it:5d}/{args.iters}] photo {p/n:.4f}  depth {dd/n:.4f}  alpha {aa/n:.4f}")
            acc[:] = 0

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "point_cloud.ply")
    gaussians.save_ply(out)
    print(f"[out] {out}   {gaussians.get_xyz.shape[0]} gaussians")


if __name__ == "__main__":
    main()
