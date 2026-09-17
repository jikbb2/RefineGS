"""TSDF a gaussian model from training views AND novel poses.

This is the extraction step the design always called for, and the reason
mesh_from_gaussians.py should not be used on an injected model.

render.py integrates rendered depth over training views only, so geometry no training
view can see never reaches the mesh -- which is why the camera-free disc/Poisson path
was written before novel poses existed. But rendered depth is the ALPHA-COMPOSITE of
every gaussian along the ray, and that compositing is a smoothing step. Poisson gets
raw per-gaussian normals instead and loses it: on obj6 the observed surface went from
NC 0.9533 to 0.7321 while the distances barely moved, i.e. it turned bumpy, not
displaced. Sampling discs also leaves the -new_dist band around existing gaussians
empty, and poisson_trim/num_cluster then delete thin prior surface.

make_prior_depth.py now supplies poses that see the unobserved side, so the workaround
is unnecessary: integrate rendered depth over both sets and keep the compositing.

  python mesh_tsdf_views.py -m <objects_inj/6> --prior_depth ~/prior/pd6.npz \
      --out <.../train/ours_30000/mesh.ply>
"""
import os, argparse, functools
import numpy as np
import torch
import open3d as o3d

print = functools.partial(print, flush=True)

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from finetune_prior import NovelCam


def intrinsic_of(cam):
    W, H = int(cam.image_width), int(cam.image_height)
    fx = W / (2.0 * np.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * np.tan(cam.FoVy * 0.5))
    return o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W / 2.0, H / 2.0)


def integrate(vol, cam, pkg, min_alpha, depth_trunc):
    """Integrate one rendered view. Transparent pixels are dropped: an alpha near zero
    means no gaussian is on that ray, and its depth is meaningless."""
    rgb = pkg["render"].clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    d = pkg["surf_depth"].squeeze(0).cpu().numpy().astype(np.float32)
    a = pkg["rend_alpha"].squeeze(0).cpu().numpy()
    d[(a < min_alpha) | ~np.isfinite(d)] = 0.0
    if not (d > 0).any():
        return False
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray((rgb * 255).astype(np.uint8))),
        o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False)
    # world_view_transform is stored transposed; the extrinsic is the plain w2c matrix
    ext = cam.world_view_transform.transpose(0, 1).cpu().numpy().astype(np.float64)
    vol.integrate(rgbd, intrinsic_of(cam), ext)
    return True


def main():
    ap = argparse.ArgumentParser()
    mp, pp = ModelParams(ap, sentinel=True), PipelineParams(ap)
    ap.add_argument("--prior_depth", default="",
                    help="npz from make_prior_depth.py. Without it this is render.py's "
                         "TSDF and the unobserved side is missing again.")
    ap.add_argument("--start_ply", default="", help="override -m's checkpoint")
    ap.add_argument("--load_iteration", default=-1, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", default=0.004, type=float, help="render.py's --voxel_size")
    ap.add_argument("--sdf_trunc", default=0.02, type=float)
    ap.add_argument("--depth_trunc", default=5.0, type=float)
    ap.add_argument("--depth_ratio", default=1.0, type=float,
                    help="render.py uses 1 (median depth), which is sharper on flat surfaces")
    ap.add_argument("--min_alpha", default=0.5, type=float)
    ap.add_argument("--num_cluster", default=1, type=int, help="0 keeps every component")
    ap.add_argument("--skip_train_views", action="store_true",
                    help="novel poses only, to see what the prior alone contributes")
    args = get_combined_args(ap)

    dev = torch.device("cuda")
    dataset, pipe = mp.extract(args), pp.extract(args)
    if hasattr(pipe, "depth_ratio"):
        pipe.depth_ratio = args.depth_ratio
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    if args.start_ply:
        gaussians.load_ply(os.path.expanduser(args.start_ply))

    cams = [] if args.skip_train_views else list(scene.getTrainCameras())
    n_train = len(cams)
    if args.prior_depth:
        z = np.load(os.path.expanduser(args.prior_depth))
        H, W = int(z["H"]), int(z["W"])
        cams += [NovelCam(z["R"][i], z["t"][i], z["K"], H, W, dev) for i in range(len(z["R"]))]
    print(f"[views] {n_train} training + {len(cams) - n_train} novel   "
          f"{gaussians.get_xyz.shape[0]:,} gaussians")

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    used = 0
    with torch.no_grad():
        for i, c in enumerate(cams):
            bg = torch.zeros(3, device=dev)
            used += integrate(vol, c, render(c, gaussians, pipe, bg),
                              args.min_alpha, args.depth_trunc)
            if (i + 1) % 100 == 0 or i + 1 == len(cams):
                print(f"  {i + 1}/{len(cams)} integrated ({used} non-empty)")

    m = vol.extract_triangle_mesh()
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    if args.num_cluster > 0 and len(m.triangles):
        lab, cnt, _ = m.cluster_connected_triangles()
        keep = np.argsort(-np.asarray(cnt))[:args.num_cluster]
        m.remove_triangles_by_mask(~np.isin(np.asarray(lab), keep))
        m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    assert o3d.io.write_triangle_mesh(out, m), f"failed to write {out}"
    print(f"[out] {out}   {len(m.vertices):,} verts  {len(m.triangles):,} tris")


if __name__ == "__main__":
    main()
