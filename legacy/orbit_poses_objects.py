#!/usr/bin/env python3
"""RefineGS — 객체 중심 orbit novel pose (See3D 미관측 완성용).

각 객체(등록된 gen 메쉬)를 중심으로 여러 각도(azimuth×elevation) orbit pose 생성.
→ 관측 앞면은 GT-warp가 실색, 미관측 뒤/옆/아래는 hole → See3D가 채움 = 객체 뒷면·옆면·다리 완성.
free-space 필터(벽 뒤 등 도달불가 pose 제외)로 prior-bound 영역의 헛생성 방지.

출력: poses.npz (warp_gt_to_pose / patch_train_novelview 호환)

실행:
  python orbit_poses_objects.py \
    --gen_root ~/Amodal3R/poc_output --colmap data/replica_room0_v2/sparse/0 \
    --occluder_mesh output/replica_room0/scene_base/train/ours_30000/fuse_cropped.ply \
    --n_az 8 --elevations -20 20 50 --radius_scale 2.5 --up_axis 1 --max_per_obj 12 \
    --out ~/See3D/dataset/whole_orbit/poses

Deps: numpy, plyfile, open3d(free-space; 없으면 필터 skip).
"""
import argparse, os, glob, struct
import numpy as np
from plyfile import PlyData


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]], float)


def read_one_cam(d):
    """intrinsics + 한 카메라(참조)만 필요."""
    if os.path.isfile(os.path.join(d, "cameras.bin")):
        with open(os.path.join(d, "cameras.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; mp = {0:3,1:4,2:4,3:5}
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24)); k = mp[model]
            p = struct.unpack(f"<{k}d", f.read(8*k))
            fx, fy = (p[0], p[1]) if model == 1 else (p[0], p[0])
            W, H = int(w), int(h)
    else:
        ln = [l for l in open(os.path.join(d, "cameras.txt")) if not l.startswith("#") and l.strip()][0]
        tt = ln.split(); model = tt[1]; W, H = int(tt[2]), int(tt[3]); p = list(map(float, tt[4:]))
        fx, fy = (p[0], p[1]) if model == "PINHOLE" else (p[0], p[0])
    return fx, fy, W, H


def proj_stored(znear, zfar, fovx, fovy):
    tx, ty = np.tan(fovx/2), np.tan(fovy/2)
    P = np.zeros((4, 4)); P[0, 0] = 1/tx; P[1, 1] = 1/ty
    P[2, 2] = zfar/(zfar-znear); P[2, 3] = -(zfar*znear)/(zfar-znear); P[3, 2] = 1.0
    return P.T


def look_at_pose(cam_pos, center, up):
    fwd = center - cam_pos; fwd /= (np.linalg.norm(fwd)+1e-9)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right)+1e-9)
    down = np.cross(fwd, right)
    Rc2w = np.stack([right, down, fwd], axis=1)      # 열=카메라축(월드)
    t = -Rc2w.T @ cam_pos                            # W2C translation
    M = np.eye(4); M[:3, :3] = Rc2w.T; M[:3, 3] = t  # W2C (Rw2c=Rc2w.T)
    return M.T                                       # stored(transpose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default=os.path.expanduser("~/Amodal3R/poc_output"))
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--occluder_mesh", default="", help="free-space 필터용 base 메쉬(벽 포함). 없으면 필터 skip")
    ap.add_argument("--gids", default="")
    ap.add_argument("--n_az", type=int, default=8)
    ap.add_argument("--elevations", type=float, nargs="+", default=[-20, 20, 50])
    ap.add_argument("--radius_scale", type=float, default=2.5, help="obj_radius 배율(관측거리)")
    ap.add_argument("--up_axis", type=int, default=1)
    ap.add_argument("--max_per_obj", type=int, default=12, help="객체당 최대 pose(free-space 통과분에서)")
    ap.add_argument("--znear", type=float, default=0.01)
    ap.add_argument("--zfar", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fx, fy, W, H = read_one_cam(a.colmap)
    fovx = 2*np.arctan(W/(2*fx)); fovy = 2*np.arctan(H/(2*fy))
    proj = proj_stored(a.znear, a.zfar, fovx, fovy)
    up = np.zeros(3); up[a.up_axis] = 1.0

    # free-space raycast scene
    rc = None
    if a.occluder_mesh and os.path.exists(a.occluder_mesh):
        try:
            import open3d as o3d
            m = o3d.io.read_triangle_mesh(a.occluder_mesh)
            rc = o3d.t.geometry.RaycastingScene()
            rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        except Exception as e:
            print(f"[warn] free-space 필터 skip: {e}")

    gids = [g.strip() for g in a.gids.split(",") if g.strip()] or \
           sorted([os.path.basename(p) for p in glob.glob(os.path.join(a.gen_root, "*")) if os.path.isdir(p)])

    recs = []; idx = 0
    for gid in gids:
        gp = os.path.join(a.gen_root, gid, "seed_1", "mesh_registered_clean.ply")
        if not os.path.exists(gp): continue
        v = PlyData.read(gp)["vertex"]
        pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
        center = pts.mean(0)
        radius = float(np.percentile(np.linalg.norm(pts-center, axis=1), 95))
        r = max(radius, 0.15) * a.radius_scale

        cand = []
        a2 = [j for j in range(3) if j != a.up_axis]
        for el in a.elevations:
            e = np.deg2rad(el)
            for k in range(a.n_az):
                az = 2*np.pi*k/a.n_az
                off = np.zeros(3); off[a2[0]] = np.cos(e)*np.cos(az); off[a2[1]] = np.cos(e)*np.sin(az); off[a.up_axis] = np.sin(e)
                cam_pos = center + r*off
                # free-space: 카메라→객체 사이가 막히면 제외
                if rc is not None:
                    d = center - cam_pos; dist = np.linalg.norm(d)
                    import open3d as o3d
                    ray = o3d.core.Tensor([[*cam_pos, *(d/dist)]], dtype=o3d.core.Dtype.Float32)
                    thit = float(rc.cast_rays(ray)["t_hit"].numpy()[0])
                    if np.isfinite(thit) and thit < dist - radius*1.5:
                        continue
                cand.append(cam_pos)
        if len(cand) > a.max_per_obj:
            sel = np.linspace(0, len(cand)-1, a.max_per_obj).astype(int)
            cand = [cand[i] for i in sel]
        for cam_pos in cand:
            wvt = look_at_pose(cam_pos, center, up)
            full = wvt @ proj
            recs.append(dict(idx=idx, world_view_transform=wvt.astype(np.float32),
                             full_proj_transform=full.astype(np.float32),
                             FoVx=float(fovx), FoVy=float(fovy), width=W, height=H))
            idx += 1
        print(f"gid {gid}: center {np.round(center,2).tolist()} r={radius:.2f} → {len(cand)} pose")

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "poses.npz"), records=np.array(recs, dtype=object))
    print(f"\n총 {len(recs)} orbit pose (객체 {len(gids)}개) → {a.out}/poses.npz")
    print("→ warp_gt_to_pose 로 GT-warp(앞면 실색+미관측 hole) → See3D 로 미관측(뒤/옆/다리) 채움.")


if __name__ == "__main__":
    main()
