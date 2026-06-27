#!/usr/bin/env python3
"""RefineGS B(See3D) — 객체 둘레 orbit 포즈에서 recon 2DGS를 렌더해 See3D warp 입력 생성.

출력: <out>/warp_XXXX.jpg (렌더 RGB) + mask_XXXX.png (255=내용/관측, 0=hole/미관측) + poses.npz(포즈 보관).
See3D inference.py 의 warp_root_dir 형식(warp_*, mask_*)에 맞춤. 포즈는 *우리 scene 좌표*라
canonical→scene 매핑 불필요. (orbit이 관측각에선 객체를, 미관측각에선 hole을 보여줌.)

실행 (split_and_splat, /home/elicer/RefineGS 에서):
  python render_orbit_warps.py \
    --gaussians output/replica_room0_v2/refinegs_fix/24/point_cloud/iteration_10000/point_cloud.ply \
    --out ~/See3D/dataset/refinegs_obj24/warp_images --n_az 16 --elevations 0 25 \
    --fov 50 --res 512 --radius_mult 2.2

주의:
  - up 벡터 기본 [0,0,1](scene z=수직 가정; camera_extent z가 작았음). 틀리면 --up 로 교체.
  - rend_alpha 키가 렌더 출력에 없으면 RGB>0 으로 mask 대체.
"""
import os, argparse
import numpy as np, torch
from PIL import Image
from argparse import Namespace

from scene import GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def look_at(eye, center, up):
    z = center - eye; z = z / (np.linalg.norm(z) + 1e-9)     # forward (+z, COLMAP)
    x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-9)  # right
    y = np.cross(z, x)                                        # down
    R_w2c = np.stack([x, y, z], 0)                            # rows = world2cam
    t = -R_w2c @ eye
    R_input = R_w2c.T                                         # getWorld2View2가 다시 전치
    return R_input.astype(np.float64), t.astype(np.float64)


def make_cam(R, T, W, H, fovx, fovy):
    znear, zfar = 0.01, 100.0
    wv = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0)).transpose(0, 1).float().cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return MiniCam(W, H, fovy, fovx, znear, zfar, wv, full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_az", type=int, default=16)
    ap.add_argument("--elevations", type=float, nargs="+", default=[0.0, 25.0])
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--radius_mult", type=float, default=2.2)
    ap.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument("--sh_degree", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g = GaussianModel(a.sh_degree); g.load_ply(a.gaussians); g.active_sh_degree = g.max_sh_degree
    xyz = g.get_xyz.detach().cpu().numpy()
    center = xyz.mean(0)
    size = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))     # bbox 대각
    radius = a.radius_mult * size / 2.0
    up = np.array(a.up, dtype=np.float64)
    print(f"center={center.round(3)} size(diag)={size:.3f} radius={radius:.3f} up={up}")

    pipe = Namespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    fov = np.deg2rad(a.fov)

    poses = []; i = 0
    for el in a.elevations:
        ele = np.deg2rad(el)
        for k in range(a.n_az):
            az = 2 * np.pi * k / a.n_az
            eye = center + radius * np.array([np.cos(ele) * np.cos(az),
                                              np.cos(ele) * np.sin(az),
                                              np.sin(ele)])
            R, T = look_at(eye, center, up)
            cam = make_cam(R, T, a.res, a.res, fov, fov)
            with torch.no_grad():
                pkg = render(cam, g, pipe, bg)
            rgb = pkg["render"].clamp(0, 1)
            img = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if "rend_alpha" in pkg and pkg["rend_alpha"] is not None:
                alpha = pkg["rend_alpha"].squeeze().detach().cpu().numpy()
            else:
                alpha = (rgb.sum(0) > 0.02).float().cpu().numpy()
            mask = (alpha > 0.5).astype(np.uint8) * 255
            Image.fromarray(img).save(os.path.join(a.out, f"warp_{i:04d}.jpg"))
            Image.fromarray(mask).save(os.path.join(a.out, f"mask_{i:04d}.png"))
            # 포즈 보관(나중에 학습 카메라 주입용): R 은 getWorld2View2 입력 규약(=R_w2c.T)
            poses.append(dict(idx=i, el=el, az=float(np.rad2deg(az)),
                              R_in=R, T=T, fov=a.fov, eye=eye))
            i += 1

    np.savez(os.path.join(a.out, "poses.npz"),
             idx=np.array([p["idx"] for p in poses]),
             R_in=np.stack([p["R_in"] for p in poses]),   # getWorld2View2 입력 R
             T=np.stack([p["T"] for p in poses]),
             eye=np.stack([p["eye"] for p in poses]),
             fov=a.fov, center=center, radius=radius, res=a.res)
    print(f"saved {i} warp/mask + poses.npz -> {a.out}")
    print("정성 확인: 관측각 warp엔 객체가, 미관측각엔 검은 hole(mask=0)이 보여야 함.")


if __name__ == "__main__":
    main()
