#!/usr/bin/env python3
"""
축3-① post-hoc visual-hull thin-structure recovery (재학습 0).

가설: 평면·monocular-depth(=G4Splat 비평면 처리)가 놓치는 thin 구조(다리 등)를,
객체 실루엣들의 교집합(visual hull = shape-from-silhouette)이 복원한다.
이를 recon이 비운 곳에 채우되, free-space(관측 표면 앞)는 제거해 observation-consistent.

파이프라인:
  1) 객체 bbox(recon 기반)에 voxel 격자.
  2) space carving: voxel center가 in-frustum 뷰들에서 마스크 안에 들어오는 비율 >= vh_frac → hull.
  3) (옵션) free-space carve: recon 표면보다 앞(카메라에 더 가까움)인 voxel 제거(RaycastingScene).
  4) hull voxel = observation-consistent 객체 점유. recon이 비운 thin 영역 복원 여부를 GT로 평가.
평가: GT→nearest 거리(completion)를 recon / hull / union(recon∪hull)로 비교, z-band(다리=낮은 z)별로도.

의존: numpy, open3d, trimesh, scipy. split_and_splat env.

실행:
    conda activate split_and_splat
    python visual_hull_recover.py \
        --masks_dir ~/obj_masks_98 \
        --colmap_dir data/replica_room0/masks/98/sparse/0 \
        --recon_ply output/replica_room0/axis3_sweep/reg_strong/98/train/ours_7000/fuse_post.ply \
        --gt_mesh ../room_0/habitat/mesh_semantic.ply --gt_id 7 \
        --voxel 0.02 --vh_frac 0.9 --freespace --margin 0.03 \
        --out_dir ~/vh_98
"""
import argparse, glob, os, struct
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ── COLMAP ──
def _q2r(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                     [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                     [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])

def _cams_txt(p):
    c={}
    for ln in open(p):
        if ln.startswith("#") or not ln.strip(): continue
        t=ln.split(); cid=int(t[0]); model=t[1]; w,h=int(t[2]),int(t[3]); pr=list(map(float,t[4:]))
        if model=="PINHOLE": fx,fy,cx,cy=pr[:4]
        else: fx=fy=pr[0]; cx,cy=pr[1],pr[2]
        c[cid]=(fx,fy,cx,cy,w,h)
    return c

def _imgs_txt(p):
    out=[]; L=[l for l in open(p) if not l.startswith("#")]
    for i in range(0,len(L),2):
        t=L[i].split()
        if len(t)<10: continue
        q=list(map(float,t[1:5])); tv=np.array(list(map(float,t[5:8])))
        out.append({"R":_q2r(q),"t":tv,"camera_id":int(t[8]),"name":t[9]})
    return out

def _bin(d):
    cams={}
    with open(os.path.join(d,"cameras.bin"),"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; mp={0:3,1:4,2:4,3:5}
        for _ in range(n):
            cid,model,w,h=struct.unpack("<iiQQ",f.read(24)); k=mp[model]; pr=struct.unpack(f"<{k}d",f.read(8*k))
            if model==1: fx,fy,cx,cy=pr[:4]
            else: fx=fy=pr[0]; cx,cy=pr[1],pr[2]
            cams[cid]=(fx,fy,cx,cy,int(w),int(h))
    imgs=[]
    with open(os.path.join(d,"images.bin"),"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        for _ in range(n):
            struct.unpack("<I",f.read(4)); q=struct.unpack("<4d",f.read(32)); tv=np.array(struct.unpack("<3d",f.read(24)))
            cid=struct.unpack("<I",f.read(4))[0]; name=b""
            while True:
                ch=f.read(1)
                if ch==b"\x00": break
                name+=ch
            n2=struct.unpack("<Q",f.read(8))[0]; f.read(24*n2)
            imgs.append({"R":_q2r(q),"t":tv,"camera_id":cid,"name":name.decode()})
    return cams,imgs

def load_cams(d):
    if os.path.isfile(os.path.join(d,"images.bin")): cams,imgs=_bin(d)
    else: cams=_cams_txt(os.path.join(d,"cameras.txt")); imgs=_imgs_txt(os.path.join(d,"images.txt"))
    out={}
    for im in imgs:
        fx,fy,cx,cy,w,h=cams[im["camera_id"]]
        out[os.path.splitext(os.path.basename(im["name"]))[0]]=dict(R=im["R"],t=im["t"],fx=fx,fy=fy,cx=cx,cy=cy,W=w,H=h)
    return out


def load_ply_xyz(path):
    m = o3d.io.read_triangle_mesh(path)
    if len(m.vertices) > 0:
        return np.asarray(m.vertices), m
    p = o3d.io.read_point_cloud(path)
    return np.asarray(p.points), None


def load_gt_verts(gt_mesh, gt_id):
    import trimesh
    with open(gt_mesh, "rb") as f:
        data = trimesh.exchange.ply.load_ply(f)
    verts = np.asarray(data["vertices"], np.float64)
    fd = data["metadata"]["_ply_raw"]["face"]["data"]
    fm = (fd["object_id"] == gt_id)
    return verts[np.unique(fd["vertex_indices"]["f1"][fm].flatten())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--recon_ply", required=True)
    ap.add_argument("--gt_mesh", required=True); ap.add_argument("--gt_id", type=int, required=True)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--vh_frac", type=float, default=0.9, help="hull: 마스크 안 비율 임계")
    ap.add_argument("--pad", type=float, default=0.1, help="bbox 패딩(m)")
    ap.add_argument("--freespace", action="store_true")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recon_xyz, recon_mesh = load_ply_xyz(args.recon_ply)
    gt = load_gt_verts(args.gt_mesh, args.gt_id)
    cams = load_cams(args.colmap_dir)
    masks = {}
    for f in glob.glob(os.path.join(args.masks_dir, "*.png")):
        from PIL import Image
        masks[os.path.splitext(os.path.basename(f))[0]] = np.array(Image.open(f).convert("L")) > 0
    cams = {k: v for k, v in cams.items() if k in masks}
    print(f"recon={len(recon_xyz)}  GT={len(gt)}  views(with mask)={len(cams)}")

    # bbox: GT가 진실이지만 평가 누수 방지 위해 recon bbox + pad 사용 (실루엣으로 확장)
    lo = recon_xyz.min(0) - args.pad; hi = recon_xyz.max(0) + args.pad
    gx = np.arange(lo[0], hi[0], args.voxel)
    gy = np.arange(lo[1], hi[1], args.voxel)
    gz = np.arange(lo[2], hi[2], args.voxel)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    vox = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float64)
    print(f"voxels: {len(vox)} (grid {len(gx)}x{len(gy)}x{len(gz)})")

    in_mask = np.zeros(len(vox), np.int32)
    in_frus = np.zeros(len(vox), np.int32)
    for stem, cam in cams.items():
        m = masks[stem]
        Xc = vox @ cam["R"].T + cam["t"]; z = Xc[:, 2]; ok = z > 1e-6
        u = cam["fx"]*Xc[:,0]/np.where(ok,z,1)+cam["cx"]; v = cam["fy"]*Xc[:,1]/np.where(ok,z,1)+cam["cy"]
        Hm, Wm = m.shape; sy=Hm/cam["H"]; sx=Wm/cam["W"]
        ui=(u*sx).astype(np.int64); vi=(v*sy).astype(np.int64)
        infr = ok & (ui>=0)&(ui<Wm)&(vi>=0)&(vi<Hm)
        in_frus += infr.astype(np.int32)
        idx = np.where(infr)[0]
        hit = m[vi[idx], ui[idx]]
        in_mask[idx[hit]] += 1
    frac = np.where(in_frus > 0, in_mask / np.maximum(in_frus, 1), 0.0)
    hull = (in_frus >= 3) & (frac >= args.vh_frac)
    print(f"hull voxels: {hull.sum()}")

    if args.freespace and recon_mesh is not None and hull.sum() > 0:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(recon_mesh))
        keep = np.ones(int(hull.sum()), bool)
        hv = vox[hull]
        for stem, cam in cams.items():
            C = (-cam["R"].T @ cam["t"]).astype(np.float64)
            d = hv - C; tg = np.linalg.norm(d, axis=1); d = d/np.maximum(tg[:,None],1e-9)
            rays = np.hstack([np.broadcast_to(C, hv.shape), d]).astype(np.float32)
            th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            front = np.isfinite(th) & (tg < th - args.margin)
            keep &= ~front
        hv = hv[keep]
        print(f"  after free-space: {len(hv)}")
    else:
        hv = vox[hull]

    # 저장
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(hv)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "hull.ply"), pc)

    # 평가: completion = GT -> nearest (recon / hull / union), 전체 + z-band
    def completion(pts):
        if len(pts) == 0: return np.full(len(gt), np.inf)
        return cKDTree(pts).query(gt)[0]
    d_rec = completion(recon_xyz)
    d_hull = completion(hv)
    d_uni = np.minimum(d_rec, d_hull)
    zb = np.quantile(gt[:, 2], [0, 0.33, 0.66, 1.0])
    print(f"\n{'band':>14} {'n':>7} {'recon(mm)':>11} {'hull(mm)':>10} {'union(mm)':>11}")
    for lo_, hi_, name in [(zb[0],zb[1],"low-z(legs?)"),(zb[1],zb[2],"mid-z"),(zb[2],zb[3],"high-z(top?)"),(zb[0],zb[3],"ALL")]:
        sel = (gt[:,2] >= lo_) & (gt[:,2] <= hi_) if name=="ALL" else (gt[:,2]>=lo_)&(gt[:,2]<hi_)
        if sel.sum()==0: continue
        print(f"{name:>14} {sel.sum():>7} {d_rec[sel].mean()*1000:>11.1f} "
              f"{d_hull[sel].mean()*1000:>10.1f} {d_uni[sel].mean()*1000:>11.1f}")
    print(f"\n저장: {args.out_dir}/hull.ply")
    print("판정: union(recon∪hull) completion이 recon보다 줄면(특히 low-z=다리), "
          "visual hull이 recon이 놓친 thin 구조를 observation-consistent하게 복원 = 축3-① 가치.")


if __name__ == "__main__":
    main()
