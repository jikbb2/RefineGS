#!/usr/bin/env python3
"""RefineGS B(See3D) — fuse_post 메시를 orbit 포즈에서 *backface-cull 포인트 splat*으로 렌더해
See3D warp 입력 생성. GL 불필요(순수 numpy, 헤드리스 안전).

원리: 메시 표면 샘플(점+normal+color) → target 포즈로 투영, **front-facing(normal이 카메라 향함)만**
남기고 z-buffer splat. 관측각=표면 보임(warp 내용), 미관측각=front 면 없음 → **진짜 hole**.
fuse_post는 num_cluster=1이라 floater 없음(깨끗). warp는 GT가 아니라 See3D 조건 입력
(관측=정확 / 미관측=hole, 내용은 See3D가 생성).

출력: <out>/warp_XXXX.jpg + mask_XXXX.png(255=내용,0=hole) + poses.npz.

실행:
  python warp_mesh_backface.py \
    --mesh output/replica_room0_v2/refinegs_fix/24/train/ours_10000/fuse_post.ply \
    --out ~/See3D/dataset/refinegs_obj24/warp_images \
    --n_az 16 --elevations 0 25 --fov 50 --res 512 --radius_mult 2.2 \
    --n_sample 400000 --splat 1

Deps: numpy, trimesh, PIL.
"""
import os, argparse
import numpy as np, trimesh
from PIL import Image


def look_at(eye, center, up):
    z = center - eye; z /= np.linalg.norm(z) + 1e-9      # forward
    x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-9   # right
    y = np.cross(z, x)                                     # down
    R_w2c = np.stack([x, y, z], 0)                         # world->cam (rows)
    t = -R_w2c @ eye
    return R_w2c, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_az", type=int, default=16)
    ap.add_argument("--elevations", type=float, nargs="+", default=[0.0, 25.0])
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--radius_mult", type=float, default=2.2)
    ap.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument("--n_sample", type=int, default=400000)
    ap.add_argument("--splat", type=int, default=1, help="splat 반경(px). 구멍 메우기")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = trimesh.load(a.mesh, process=False, force="mesh")
    m.fix_normals()                                        # 일관 outward normal
    P, fidx = trimesh.sample.sample_surface(m, a.n_sample)
    P = np.asarray(P, np.float64)
    N = np.asarray(m.face_normals[fidx], np.float64)
    if hasattr(m.visual, "vertex_colors") and len(m.visual.vertex_colors) == len(m.vertices):
        vc = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        C = vc[m.faces].mean(1)[fidx]
    else:
        C = np.full((len(P), 3), 0.6)

    center = P.mean(0); size = float(np.linalg.norm(P.max(0) - P.min(0)))
    radius = a.radius_mult * size / 2.0
    up = np.array(a.up, np.float64)
    W = H = a.res
    f = (W / 2) / np.tan(np.deg2rad(a.fov) / 2); cx = cy = W / 2
    print(f"center={center.round(3)} size={size:.3f} radius={radius:.3f} pts={len(P)}")

    poses = []; i = 0
    for el in a.elevations:
        ele = np.deg2rad(el)
        for k in range(a.n_az):
            az = 2 * np.pi * k / a.n_az
            eye = center + radius * np.array([np.cos(ele) * np.cos(az),
                                              np.cos(ele) * np.sin(az), np.sin(ele)])
            R_w2c, t = look_at(eye, center, up)

            front = (N * (eye - P)).sum(1) > 0            # ★ backface cull
            Xc = P @ R_w2c.T + t
            Zc = Xc[:, 2]
            ok = front & (Zc > 1e-4)
            u = (f * Xc[:, 0] / Zc + cx)
            v = (f * Xc[:, 1] / Zc + cy)
            ui = np.round(u).astype(np.int64); vi = np.round(v).astype(np.int64)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

            warp = np.zeros((H, W, 3), np.float64)
            zbuf = np.full((H, W), np.inf)
            idx = np.where(ok)[0]
            order = idx[np.argsort(-Zc[idx])]             # 먼 점 먼저 → 가까운 점이 덮음
            for r in range(-a.splat, a.splat + 1):
                for c in range(-a.splat, a.splat + 1):
                    yy = np.clip(vi[order] + r, 0, H - 1); xx = np.clip(ui[order] + c, 0, W - 1)
                    warp[yy, xx] = C[order]; zbuf[yy, xx] = Zc[order]
            mask = (zbuf < np.inf).astype(np.uint8) * 255
            Image.fromarray((warp * 255).astype(np.uint8)).save(os.path.join(a.out, f"warp_{i:04d}.jpg"))
            Image.fromarray(mask).save(os.path.join(a.out, f"mask_{i:04d}.png"))
            poses.append(dict(idx=i, R_w2c=R_w2c, t=t, eye=eye, el=el,
                              az=float(np.rad2deg(az))))
            i += 1

    np.savez(os.path.join(a.out, "poses.npz"),
             idx=np.array([p["idx"] for p in poses]),
             R_w2c=np.stack([p["R_w2c"] for p in poses]),
             t=np.stack([p["t"] for p in poses]),
             eye=np.stack([p["eye"] for p in poses]),
             fov=a.fov, res=a.res, center=center, radius=radius)
    print(f"saved {i} warp/mask + poses.npz -> {a.out}")
    print("정성: 관측각=table 표면(내용), 미관측각=검은 hole(mask=0). floater 없음.")


if __name__ == "__main__":
    main()
