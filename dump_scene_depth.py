#!/usr/bin/env python3
"""Dump per-view depth from the whole-scene model as npz -- the evidence used for
free-space carving in the per-object SDF.

Saving the depth of a well-regularised whole-scene model (e.g. scene_mono_reg) for all
200 views lets sdf_distill_depth.py read it back with --carve_depth_dir and apply the
"between the camera and the surface = empty space" constraint inside the object bbox.
That carves the space beside and under a table, which the 8 object views alone never
see.

  python dump_scene_depth.py -m output/replica_room0_v2/scene_mono_reg -s data/replica_room0_v2 \
    --iteration 30000 --depth_ratio 1 --out_dir ~/carve_depth_mono
"""
import os
import sys
import numpy as np
import torch
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args


def cam_intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2],
                            [0, H / 2, 0, (H - 1) / 2],
                            [0, 0, 0, 1]]).float().cuda().T
    intrins = (cam.projection_matrix @ ndc2pix)[:3, :3].T
    fx, fy = intrins[0, 0].item(), intrins[1, 1].item()
    cx, cy = intrins[0, 2].item(), intrins[1, 2].item()
    extrinsic = cam.world_view_transform.T
    return fx, fy, cx, cy, W, H, extrinsic


@torch.no_grad()
def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--alpha_thr", default=0.5, type=float,
                        help="pixels at or below this alpha get depth=0 (invalid)")
    # A ray that hits nothing renders as inf. float16 casts that to inf, and every consumer
    # treats "depth > 0.01" as a valid reading, so one inf pixel votes the whole ray empty
    # and carves straight through the scene. Clamp before the cast, not after.
    parser.add_argument("--max_depth", default=20.0, type=float,
                        help="drop depths above this (m) and any non-finite value")
    parser.add_argument("--out_dir", required=True, type=str)
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    # --depth_ratio belongs to PipelineParams and its default is not render.py's. Every
    # other consumer forces 1 (mesh_tsdf_views.py, sdf_distill_depth.py); a silent
    # mismatch here makes the carve reference a different quantity from the meshes it is
    # compared against.
    if hasattr(pipe, "depth_ratio") and not any(
            a.startswith("--depth_ratio") for a in sys.argv[1:]):
        pipe.depth_ratio = 1.0
    print(f"[cfg] depth_ratio {getattr(pipe, 'depth_ratio', 'n/a')}  "
          f"alpha_thr {args.alpha_thr}  max_depth {args.max_depth}m  store float16 "
          f"(quantisation ~2mm at 2.5m, ~4mm at 5m; carve margin is 15mm)")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    views = scene.getTrainCameras()
    n_bad = n_px = 0
    for i, cam in enumerate(views):
        pkg = render(cam, gaussians, pipe, background)
        depth = pkg["surf_depth"][0]
        alpha = pkg["rend_alpha"][0]
        depth = torch.where(alpha > args.alpha_thr, depth, torch.zeros_like(depth))
        bad = (~torch.isfinite(depth)) | (depth > args.max_depth) | (depth < 0)
        n_bad += int(bad.sum()); n_px += depth.numel()
        depth = torch.where(bad, torch.zeros_like(depth), depth)
        fx, fy, cx, cy, W, H, extrinsic = cam_intrinsics(cam)
        c2w = torch.inverse(extrinsic).cpu().numpy().astype(np.float32)
        np.savez_compressed(
            os.path.join(out_dir, f"{os.path.splitext(cam.image_name)[0]}.npz"),
            depth=depth.cpu().numpy().astype(np.float16),
            fx=np.float32(fx), fy=np.float32(fy), cx=np.float32(cx), cy=np.float32(cy),
            c2w=c2w)
        if i % 50 == 0:
            print(f"{i}/{len(views)} ...")
    print(f"[clamp] dropped {n_bad:,}/{n_px:,} px ({n_bad / max(n_px, 1) * 100:.3f}%) "
          f"as non-finite or > {args.max_depth}m")
    print(f"done: {len(views)} views -> {out_dir}")


if __name__ == "__main__":
    main()