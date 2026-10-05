#!/usr/bin/env python3
"""
축3 (observation-consistency) — 임의 mesh의 증거-일관성 certificate.

어떤 reconstruction/completion mesh든 받아, 각 표면 점을 관측 대비 분류:
  - VIOLATING : 어떤 뷰에서든 관측 표면보다 *앞*(카메라에 더 가까움)이거나, 이미지 안인데
                ray가 관측 표면에 안 맞음(빈 공간에 뜸) = 증거와 모순 = 입증적으로 틀림.
  - OBSERVED  : 어떤 뷰에서 관측 표면 위(±band)에 있음 = 검증됨.
  - UNVERIFIED: 위반도 관측도 아님(항상 표면 뒤=occluded) = 검증 불가(=prior 추측).
보고: violation_rate(↓, 0이 보장), unverified_rate(정량화된 hallucination 위험).

free-space 기준이라 객체 마스크 불필요. 관측 표면 = recon mesh(raycast).
recon→~0 위반/대부분 OBSERVED, naive fusion→위반+UNVERIFIED, carve→위반~0+UNVERIFIED 정량.

의존: numpy, open3d. split_and_splat env.

실행:
    conda activate split_and_splat
    python obs_consistency_report.py \
        --mesh <인증할 mesh.ply> \
        --recon_ply <관측표면 recon mesh.ply> \
        --colmap_dir data/replica_room0/masks/98/sparse/0 \
        --stride 2 --margin 0.03 --n 50000
  비교: --mesh를 recon / append-fused / carve-fused 각각으로 돌려 표 작성.
"""
import argparse, os, struct
import numpy as np
import open3d as o3d


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
    out=[]
    for im in imgs:
        fx,fy,cx,cy,w,h=cams[im["camera_id"]]
        out.append(dict(R=im["R"],t=im["t"],fx=fx,fy=fy,cx=cx,cy=cy,W=w,H=h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="인증할 mesh")
    ap.add_argument("--recon_ply", required=True, help="관측 표면 기준 recon mesh")
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.03, help="앞/표면 판정 band(m)")
    ap.add_argument("--n", type=int, default=50000, help="mesh 표면 샘플 수")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.triangles) == 0:
        print("[ERROR] mesh에 face 없음"); return
    pts = np.asarray(mesh.sample_points_uniformly(args.n).points)
    recon = o3d.io.read_triangle_mesh(args.recon_ply)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(recon))
    cams = load_cams(args.colmap_dir)[::args.stride]
    print(f"mesh pts={len(pts)}  recon faces={len(recon.triangles)}  views={len(cams)}")

    N = len(pts)
    violating = np.zeros(N, bool)
    observed = np.zeros(N, bool)
    seen_any = np.zeros(N, bool)   # in-image in >=1 view
    for cam in cams:
        Xc = pts @ cam["R"].T + cam["t"]; z = Xc[:, 2]; ok = z > 1e-6
        u = cam["fx"]*Xc[:,0]/np.where(ok,z,1)+cam["cx"]; v = cam["fy"]*Xc[:,1]/np.where(ok,z,1)+cam["cy"]
        inb = ok & (u>=0)&(u<cam["W"])&(v>=0)&(v<cam["H"])
        idx = np.where(inb)[0]
        if len(idx)==0: continue
        seen_any[idx] = True
        C = (-cam["R"].T @ cam["t"]).astype(np.float64)
        P = pts[idx]; d = P - C; tg = np.linalg.norm(d,axis=1); d = d/np.maximum(tg[:,None],1e-9)
        rays = np.hstack([np.broadcast_to(C,P.shape), d]).astype(np.float32)
        th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(th)
        front = np.zeros(len(idx),bool); onsurf = np.zeros(len(idx),bool)
        front[hit] = tg[hit] < th[hit] - args.margin           # 관측 표면 앞 = 위반
        onsurf[hit] = np.abs(tg[hit] - th[hit]) <= args.margin  # 표면 위 = 관측됨
        nohit_front = ~hit                                      # 이미지 안인데 표면 없음 = 빈공간에 뜸
        violating[idx[front | nohit_front]] = True
        observed[idx[onsurf]] = True

    observed &= ~violating
    unverified = seen_any & ~violating & ~observed   # 항상 표면 뒤(occluded) = 검증불가
    never_seen = ~seen_any                            # 어느 뷰에도 안 들어옴
    print(f"\n=== observation-consistency certificate: {os.path.basename(args.mesh)} ===")
    print(f"  VIOLATING (증거 모순, ↓0 보장 목표): {violating.mean()*100:6.2f}%")
    print(f"  OBSERVED  (검증됨)                : {observed.mean()*100:6.2f}%")
    print(f"  UNVERIFIED(occluded, prior 추측)  : {unverified.mean()*100:6.2f}%")
    print(f"  never-in-frustum                  : {never_seen.mean()*100:6.2f}%")
    print("\n해석: recon→VIOLATING~0·OBSERVED 多. naive fusion→VIOLATING>0+UNVERIFIED多(hallucination).")
    print("carve→VIOLATING~0(보장)+UNVERIFIED 정량화. G4Splat류 inpaint는 보장 없음(이 지표로 드러남).")


if __name__ == "__main__":
    main()
