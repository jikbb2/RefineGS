#!/usr/bin/env python3
"""Forward-warp GT depth to a novel pose, as input for See3D in place of a noisy render.

[v2]
  - Fixed a cross-source z-buffer bug: the colour write now passes a z-test. Previously a
    later source overwrote unconditionally, so a far surface punched through a near one --
    worse the larger k_nearest is.
  - Added a depth-boundary (streamer) filter: depth pixels on a silhouette that mix
    foreground and background are dropped (--edge_thr). They are what produces fragments
    floating in mid-air at a wide-baseline novel pose.

The problem: rendering 2DGS at an unseen pose gives floaters and streaks, which makes a poor
See3D input. The fix, which is See3D's own approach: forward-project the ACTUAL GT pixels to
the target novel pose using GT depth. The observed surface then carries true colour with no
noise, and a hole is a real unobserved or disoccluded region, which is small.

Input poses: poses.npz from render_hole_novel --soft_out (reachable novel poses, which we
estimate well).
Output, in the soft_in layout the existing See3D pipeline expects:
  view_<i>.jpg    the GT warp (true colour; holes are black)
  weight_<i>.png  hole map (255 = hole / unobserved, 0 = known / true colour), which
                  generate_novel_views see3d reads as its mask
  poses.npz       copied through
With --scene_mesh the weight has three levels instead: 255 = observed, 128 = unobserved real
surface (what See3D should generate), 0 = outside the frustum (excluded).

Run (stride 1 advised):
  python warp_gt_to_pose.py \
    --poses ~/See3D/dataset/obj24/soft/poses.npz \
    --gt_images data/<scene>/images \
    --gt_depth "$HOME"/nice-slam/Datasets/Replica/room0/results \
    --colmap data/<scene>/sparse/0 \
    --depth_scale 6553.5 --k_nearest 24 --src_stride 1 --edge_thr 0.05 \
    --scene_mesh output/<scene>/scene_mono_reg/train/ours_30000/fuse_post.ply \
    --out ~/See3D/dataset/obj24/soft_in_gtwarp

Deps: numpy, PIL.
"""
import argparse, os, struct, glob, shutil
import numpy as np
from PIL import Image


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]], float)


def read_colmap(d):
    cams, imgs = {}, []
    if os.path.isfile(os.path.join(d, "cameras.bin")):
        with open(os.path.join(d, "cameras.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; mp = {0:3,1:4,2:4,3:5}
            for _ in range(n):
                cid, model, w, h = struct.unpack("<iiQQ", f.read(24)); k = mp[model]
                p = struct.unpack(f"<{k}d", f.read(8*k))
                fx, fy, cx, cy = (p[0],p[1],p[2],p[3]) if model==1 else (p[0],p[0],p[1],p[2])
                cams[cid] = (fx,fy,cx,cy,int(w),int(h))
        with open(os.path.join(d, "images.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                struct.unpack("<I", f.read(4)); q = struct.unpack("<4d", f.read(32))
                t = np.array(struct.unpack("<3d", f.read(24))); cid = struct.unpack("<I", f.read(4))[0]
                nm = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00": break
                    nm += c
                n2 = struct.unpack("<Q", f.read(8))[0]; f.read(24*n2)
                imgs.append((qvec2rot(q), t, cid, nm.decode()))
    else:
        for ln in open(os.path.join(d,"cameras.txt")):
            if ln.startswith("#") or not ln.strip(): continue
            tt = ln.split(); cid=int(tt[0]); model=tt[1]; w,h=int(tt[2]),int(tt[3]); p=list(map(float,tt[4:]))
            fx,fy,cx,cy = (p[0],p[1],p[2],p[3]) if model=="PINHOLE" else (p[0],p[0],p[1],p[2])
            cams[cid]=(fx,fy,cx,cy,w,h)
        L=[l for l in open(os.path.join(d,"images.txt")) if not l.startswith("#")]
        for i in range(0,len(L),2):
            tt=L[i].split()
            if len(tt)<10: continue
            q=list(map(float,tt[1:5])); t=np.array(list(map(float,tt[5:8])))
            imgs.append((qvec2rot(q),t,int(tt[8]),tt[9]))
    out=[]
    for R,t,cid,nm in imgs:
        fx,fy,cx,cy,w,h = cams[cid]
        out.append(dict(R=R,t=t,fx=fx,fy=fy,cx=cx,cy=cy,W=w,H=h,
                        stem=os.path.splitext(os.path.basename(nm))[0]))
    return out


def load_depth(path, scale, W, H):
    im = Image.open(path)
    d = np.asarray(im).astype(np.float32)
    if d.ndim == 3: d = d[..., 0]
    d = d / scale
    if d.shape != (H, W):
        d = np.asarray(Image.fromarray(d).resize((W, H), Image.NEAREST))
    return d


def target_from_pose(rec):
    wvt = np.asarray(rec["world_view_transform"], float)   # stored = getWorld2View2(R,T).T
    M = wvt.T                                              # W2C 4x4
    R, t = M[:3, :3], M[:3, 3]
    W, H = int(rec["width"]), int(rec["height"])
    fx = W / (2*np.tan(float(rec["FoVx"])/2)); fy = H / (2*np.tan(float(rec["FoVy"])/2))
    return R, t, fx, fy, W/2.0, H/2.0, W, H


def cam_center(R, t):  # -R^T t
    return -R.T @ t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--gt_images", required=True)
    ap.add_argument("--gt_depth", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--depth_scale", type=float, default=6553.5)
    ap.add_argument("--k_nearest", type=int, default=6)
    ap.add_argument("--src_stride", type=int, default=1,
                    help="pixel stride on the source, for speed. 1 = full, and advised")
    ap.add_argument("--edge_thr", type=float, default=0.05,
                    help="depth boundary filter: drop a pixel whose relative depth change "
                         "against its neighbour exceeds this, which prevents streamers. 0 = off")
    ap.add_argument("--scene_margin", type=float, default=0.10,
                    help="[v4] cancel a phantom-known pixel when the warped depth lies this "
                         "far (m) behind the scene mesh surface")
    ap.add_argument("--void_fallback", default="aabb", choices=["aabb", "off"],
                    help="[v5] second test, against the room AABB, for a ray that missed the "
                         "mesh -- promotes a hole in the indoor mesh to 128 (to be generated)")
    ap.add_argument("--scene_mesh", default="",
                    help="the scene (room) mesh, e.g. a base fuse_cropped.ply. With it the "
                         "weight has three levels: 255 = observed (true colour), 128 = "
                         "unobserved real surface (See3D's target, 0.5), 0 = outside the "
                         "frustum (void, excluded). Without it the weight is the old hole map.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Scene raycast: tells "outside the frustum" apart from "unobserved surface".
    rc_scene, aabb_lo, aabb_hi = None, None, None
    if a.scene_mesh and os.path.exists(a.scene_mesh):
        import open3d as o3d
        _m = o3d.io.read_triangle_mesh(a.scene_mesh)
        rc_scene = o3d.t.geometry.RaycastingScene()
        rc_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(_m))
        _vv = np.asarray(_m.vertices)
        aabb_lo, aabb_hi = _vv.min(0) - 0.05, _vv.max(0) + 0.05
        print(f"scene raycast on {a.scene_mesh} -> three-level weight"
              f" (void fallback={a.void_fallback}: a direction through a hole in the mesh is"
              f" promoted to unobserved surface, 128)")

    cams = read_colmap(a.colmap)
    src = []
    for c in cams:
        ip = os.path.join(a.gt_images, c["stem"] + ".jpg")
        if not os.path.exists(ip): ip = os.path.join(a.gt_images, c["stem"] + ".png")
        dp = os.path.join(a.gt_depth, c["stem"].replace("frame", "depth") + ".png")
        if os.path.exists(ip) and os.path.exists(dp):
            c["img_path"], c["depth_path"] = ip, dp
            c["center"] = cam_center(c["R"], c["t"])
            src.append(c)
    if not src:
        raise SystemExit("no GT image/depth pair matched -- check the paths and the "
                         "frame<->depth naming")
    print(f"source GT frames: {len(src)}")
    src_centers = np.stack([c["center"] for c in src])

    recs = list(np.load(a.poses, allow_pickle=True)["records"])
    os.makedirs(a.out, exist_ok=True)
    st = a.src_stride

    for rec in recs:
        i = int(rec["idx"])
        Rt, tt, fxt, fyt, cxt, cyt, W, H = target_from_pose(rec)
        tc = cam_center(Rt, tt)
        order = np.argsort(np.linalg.norm(src_centers - tc, axis=1))[:a.k_nearest]

        zbuf = np.full(H*W, np.inf, np.float32)
        color = np.zeros((H*W, 3), np.float32)
        # ---- pass 1: accumulate a global min-z over every source ----
        cache = []
        for si in order:
            c = src[si]
            img = np.asarray(Image.open(c["img_path"]).convert("RGB")).astype(np.float32)
            Hs, Ws = img.shape[:2]
            dep = load_depth(c["depth_path"], a.depth_scale, Ws, Hs)
            # Depth boundary filter: drop the mixed depth pixels on a silhouette, which is
            # what puts fragments in mid-air.
            if a.edge_thr > 0:
                gy, gx = np.gradient(dep)
                edge = (np.abs(gx) + np.abs(gy)) > a.edge_thr * np.clip(dep, 1e-3, None)
            else:
                edge = np.zeros_like(dep, bool)
            vs, us = np.mgrid[0:Hs:st, 0:Ws:st]
            us = us.ravel(); vs = vs.ravel(); d = dep[vs, us]
            ok = (d > 1e-3) & (~edge[vs, us])
            us, vs, d = us[ok], vs[ok], d[ok]
            x = (us - c["cx"]) / c["fx"] * d
            y = (vs - c["cy"]) / c["fy"] * d
            Xc = np.stack([x, y, d], 1)
            Xw = (Xc - c["t"]) @ c["R"]
            Xct = Xw @ Rt.T + tt
            z = Xct[:, 2]
            front = z > 1e-3
            Xct, z = Xct[front], z[front]
            col = img[vs[front], us[front]]
            ut = (Xct[:, 0]/z*fxt + cxt).astype(np.int64)
            vt = (Xct[:, 1]/z*fyt + cyt).astype(np.int64)
            inb = (ut >= 0)&(ut < W)&(vt >= 0)&(vt < H)
            flat = vt[inb]*W + ut[inb]; zz = z[inb]; cc = col[inb]
            np.minimum.at(zbuf, flat, zz)
            cache.append((flat, zz, cc))
        # ---- pass 2: write colour only where the global z-test passes, so one source
        #      cannot punch through another ----
        for flat, zz, cc in cache:
            win = zz <= zbuf[flat] * 1.002
            color[flat[win]] = cc[win]
        del cache

        filled = (zbuf < np.inf).reshape(H, W)
        view = color.reshape(H, W, 3).astype(np.uint8)
        Image.fromarray(view).save(os.path.join(a.out, f"view_{i:04d}.jpg"), quality=95)
        # [v3] Save the warped depth, to reclassify phantom-known pixels: background behind
        #      an object that shows through it.
        np.save(os.path.join(a.out, f"depth_{i:04d}.npy"),
                np.where(zbuf < np.inf, zbuf, 0).reshape(H, W).astype(np.float16))

        if rc_scene is not None:
            import open3d as o3d
            uu, vv = np.meshgrid(np.arange(W), np.arange(H))
            dc = np.stack([(uu-cxt)/fxt, (vv-cyt)/fyt, np.ones_like(uu, float)], -1).reshape(-1, 3)
            Rc2w = Rt.T; ccam = (-Rt.T @ tt).astype(np.float32)
            dw = dc @ Rc2w.T
            dnorm2 = np.linalg.norm(dw, axis=1, keepdims=True) + 1e-9
            dwn = dw / dnorm2
            rays = np.concatenate([np.broadcast_to(ccam, dwn.shape), dwn], 1).astype(np.float32)
            thit = rc_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            scene_hit = np.isfinite(thit).reshape(H, W)
            # [v5] void fallback: a ray that missed the mesh but still crosses the room's
            #      AABB is indoors, so a surface certainly exists there. This is what turns a
            #      hole in the mono mesh -- in the floor, say -- into 128 (to be generated)
            #      rather than 0 (excluded).
            if a.void_fallback == "aabb" and aabb_lo is not None:
                inv = 1.0 / np.where(np.abs(dwn) < 1e-9, 1e-9, dwn)
                t0s = (aabb_lo[None, :] - ccam[None, :]) * inv
                t1s = (aabb_hi[None, :] - ccam[None, :]) * inv
                tmin = np.minimum(t0s, t1s).max(1)
                tmax = np.maximum(t0s, t1s).min(1)
                aabb_hit = (tmax >= np.maximum(tmin, 0.0)).reshape(H, W)
                scene_hit = scene_hit | aabb_hit
            # [v4] Scene-level phantom filter: a known pixel that came from BEHIND the mesh
            #      surface is background showing through, so its known status is cancelled
            #      (blacked out and promoted to unobserved). This removes the floor seen
            #      through a sofa, the flower floating in the air, and similar artefacts.
            #      hit point = ccam + dwn*thit, so the camera z is Rt[2] . hit + tt[2].
            hitp = ccam[None, :] + dwn * thit[:, None]
            z_mesh_cam = np.where(np.isfinite(thit), hitp @ Rt[2] + tt[2], np.inf).reshape(H, W)
            zb = np.where(zbuf < np.inf, zbuf, 0).reshape(H, W)
            phantom = filled & scene_hit & (zb > z_mesh_cam + a.scene_margin)
            if phantom.any():
                view[phantom] = 0
                Image.fromarray(view).save(os.path.join(a.out, f"view_{i:04d}.jpg"), quality=95)
                filled = filled & ~phantom
            w = np.zeros((H, W), np.uint8)
            w[filled] = 255
            w[(~filled) & scene_hit] = 128
            Image.fromarray(w).save(os.path.join(a.out, f"weight_{i:04d}.png"))
            print(f"[{i:04d}] known {filled.mean():.3f}  unobserved surface "
                  f"{((~filled)&scene_hit).mean():.3f}  outside frustum "
                  f"{((~filled)&~scene_hit).mean():.3f}  phantom {phantom.mean():.3f}")
        else:
            hole = ~filled
            Image.fromarray((hole*255).astype(np.uint8)).save(os.path.join(a.out, f"weight_{i:04d}.png"))
            print(f"[{i:04d}] filled {filled.mean():.3f}  hole {hole.mean():.3f}")

    shutil.copy(a.poses, os.path.join(a.out, "poses.npz"))
    print(f"\n-> {a.out} (view = GT warp, weight, poses.npz).")


if __name__ == "__main__":
    main()