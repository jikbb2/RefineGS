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
import os, sys, argparse, functools
import numpy as np
import torch
import torch.nn.functional as tf
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


def valid_mask(cam, pkg, args):
    """Which pixels carry depth worth integrating.

    The seen/unseen boundary is where a TSDF goes rough, for three reasons that
    render.py does not guard against:

      alpha      a half-transparent pixel has no surface on its ray, so its
                 composite depth is a blend of whatever it did hit
      grazing    where the surface turns away from the camera the disc is seen
                 edge-on; its depth is the least reliable exactly at the silhouette
      jump       at the silhouette the depth steps from object to background, and
                 the volume fills that step with a skirt joining the two

    Dropping these removes the rough band rather than repairing it. That is the
    intent: the gap is then filled by the prior, so only trustworthy observation
    reaches the conditioning points.
    """
    d = pkg["surf_depth"].squeeze(0)
    ok = (pkg["rend_alpha"].squeeze(0) >= args.min_alpha) & torch.isfinite(d) & (d > 0)

    n = pkg.get("rend_normal")
    if n is not None and args.min_cos > 0:
        # world -> camera; in the CV convention a face-on surface has |n_z| near 1
        R = cam.world_view_transform.transpose(0, 1)[:3, :3]
        nz = torch.einsum("ij,jhw->ihw", R, n)[2]
        ok &= nz.abs() >= args.min_cos

    if args.max_jump > 0:
        dv = torch.where(ok, d, torch.zeros_like(d))[None, None]
        hi = tf.max_pool2d(dv, 3, 1, 1)
        lo = -tf.max_pool2d(-torch.where(ok, d, torch.full_like(d, 1e4))[None, None], 3, 1, 1)
        ok &= (hi - lo)[0, 0] <= args.max_jump   # among valid pixels; erode handles the rim

    for _ in range(args.erode):                  # min-pool: keep only if every neighbour is ok
        ok = (-tf.max_pool2d(-ok.float()[None, None], 3, 1, 1)[0, 0]) > 0.5
    return ok


def integrate(vol, cam, pkg, args):
    ok = valid_mask(cam, pkg, args)
    rgb = pkg["render"].clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy()
    sd = pkg["surf_depth"].squeeze(0)
    d = torch.where(ok, sd, torch.zeros_like(sd)).cpu().numpy()
    d = np.ascontiguousarray(d, np.float32)
    if not (d > 0).any():
        return 0.0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray((rgb * 255).astype(np.uint8))),
        o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=args.depth_trunc,
        convert_rgb_to_intensity=False)
    # world_view_transform is stored transposed; the extrinsic is the plain w2c matrix
    ext = cam.world_view_transform.transpose(0, 1).cpu().numpy().astype(np.float64)
    vol.integrate(rgbd, intrinsic_of(cam), ext)
    return float(ok.float().mean())


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
    ap.add_argument("--min_alpha", default=0.5, type=float,
                    help="skip pixels this transparent")
    ap.add_argument("--min_cos", default=0.2, type=float,
                    help="skip pixels whose surface is more than ~78 deg off the view "
                         "direction. 0 disables.")
    ap.add_argument("--max_jump", default=0.05, type=float,
                    help="skip pixels within one pixel of a depth step this large (m), "
                         "which is what creates the silhouette skirt. 0 disables.")
    ap.add_argument("--erode", default=1, type=int,
                    help="shrink the valid region by this many pixels afterwards")
    ap.add_argument("--num_cluster", default=1, type=int, help="0 keeps every component")
    ap.add_argument("--skip_train_views", action="store_true",
                    help="novel poses only, to see what the prior alone contributes")
    args = get_combined_args(ap)

    dev = torch.device("cuda")
    dataset, pipe = mp.extract(args), pp.extract(args)
    # --depth_ratio belongs to PipelineParams. Its default there is not render.py's, and
    # a silent mismatch would make this mesh incomparable to fuse_post, so force 1 unless
    # it was given explicitly.
    if hasattr(pipe, "depth_ratio") and not any(
            a.startswith("--depth_ratio") for a in sys.argv[1:]):
        pipe.depth_ratio = 1.0
    print(f"[cfg] depth_ratio {getattr(pipe, 'depth_ratio', 'n/a')}  "
          f"min_cos {args.min_cos}  max_jump {args.max_jump}  erode {args.erode}")
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
    kept, used = [], 0
    with torch.no_grad():
        bg = torch.zeros(3, device=dev)
        for i, c in enumerate(cams):
            f = integrate(vol, c, render(c, gaussians, pipe, bg), args)
            used += f > 0
            kept.append(f)
            if (i + 1) % 100 == 0 or i + 1 == len(cams):
                print(f"  {i + 1}/{len(cams)}  {used} non-empty  "
                      f"pixels kept {np.mean(kept) * 100:.2f}%")

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