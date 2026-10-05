#!/usr/bin/env python3
"""RefineGS B2 — mesh(Amodal3R gen / recon) → 2DGS surfel point_cloud.ply.

베이스 체크포인트와 *동일 스키마*(Split&Splat 448필드)로 출력 → 그들 train.py/GaussianModel 이 그대로 로드.
필드: x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..44, id_0..2, desc_0..383,
      opacity, scale_0, scale_1, rot_0..3   (총 448, 모두 float32)

surfel 구성:
  - position = mesh 표면 균일 샘플
  - normal(nx,ny,nz) = face normal  ── rotation(rot_*)= [t1,t2,n] 프레임 quaternion (z축=normal)
  - color → f_dc = (rgb-0.5)/C0  (SH DC),  f_rest=0  (view-dep 없음)
  - scale_0=scale_1=log(r),  r = scale_mult * sqrt(surface_area / n_samples)  (표면 타일링)
  - opacity = logit(opacity_init)  (기본 0.99, 높게)
  - id_*, desc_* = 0  (geometry joint refine 엔 불필요; 베이스와 필드만 일치)

실행:
  python mesh_to_surfels.py --mesh <gen_or_recon.ply> --out <surfels.ply> \
      --n_samples 200000 --scale_mult 1.0 --opacity 0.99

검증(왕복 로드 + 필드 수):
  python -c "from plyfile import PlyData;p=PlyData.read('<surfels.ply>');v=p['vertex'];\
print('verts',len(v.data),'props',len(v.properties))"   # props=448 기대

Deps: numpy, trimesh, plyfile.
"""
import argparse, numpy as np, trimesh
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814           # SH degree-0 factor
N_REST = 45                        # f_rest (SH deg3: 16*3-3)
N_DESC = 384                       # Split&Splat descriptor dim


def field_names():
    names = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2"]
    names += [f"f_rest_{i}" for i in range(N_REST)]
    names += ["id_0","id_1","id_2"]
    names += [f"desc_{i}" for i in range(N_DESC)]
    names += ["opacity","scale_0","scale_1","rot_0","rot_1","rot_2","rot_3"]
    return names


def rotmat_to_quat(R):
    """R:(N,3,3) 회전행렬 → quat (w,x,y,z), 정규화."""
    m00,m11,m22 = R[:,0,0],R[:,1,1],R[:,2,2]
    tr = m00+m11+m22
    q = np.zeros((len(R),4))
    # branchless-ish: 안정성 위해 trace 양수 분기 + fallback
    s = np.sqrt(np.maximum(tr+1.0,1e-12))*2
    w = 0.25*s
    x = (R[:,2,1]-R[:,1,2])/s
    y = (R[:,0,2]-R[:,2,0])/s
    z = (R[:,1,0]-R[:,0,1])/s
    q[:,0],q[:,1],q[:,2],q[:,3] = w,x,y,z
    # trace<=0 인 경우 가장 큰 대각으로 재계산
    bad = tr<=0
    if bad.any():
        for i in np.where(bad)[0]:
            Ri=R[i]; d=np.array([Ri[0,0],Ri[1,1],Ri[2,2]]); k=int(np.argmax(d))
            if k==0:
                s=np.sqrt(1.0+Ri[0,0]-Ri[1,1]-Ri[2,2])*2
                q[i]=[(Ri[2,1]-Ri[1,2])/s,0.25*s,(Ri[0,1]+Ri[1,0])/s,(Ri[0,2]+Ri[2,0])/s]
            elif k==1:
                s=np.sqrt(1.0-Ri[0,0]+Ri[1,1]-Ri[2,2])*2
                q[i]=[(Ri[0,2]-Ri[2,0])/s,(Ri[0,1]+Ri[1,0])/s,0.25*s,(Ri[1,2]+Ri[2,1])/s]
            else:
                s=np.sqrt(1.0-Ri[0,0]-Ri[1,1]+Ri[2,2])*2
                q[i]=[(Ri[1,0]-Ri[0,1])/s,(Ri[0,2]+Ri[2,0])/s,(Ri[1,2]+Ri[2,1])/s,0.25*s]
    q /= np.linalg.norm(q,axis=1,keepdims=True)+1e-12
    return q


def quats_from_normals(n):
    n = n/(np.linalg.norm(n,axis=1,keepdims=True)+1e-12)
    ref = np.tile(np.array([0,0,1.0]),(len(n),1))
    par = np.abs((n*ref).sum(1))>0.99
    ref[par] = np.array([1.0,0,0])
    t1 = np.cross(ref,n); t1/=np.linalg.norm(t1,axis=1,keepdims=True)+1e-12
    t2 = np.cross(n,t1)
    R = np.stack([t1,t2,n],axis=2)        # 열 = [t1,t2,n], 3번째 열(z축)=normal
    return rotmat_to_quat(R)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mesh",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--n_samples",type=int,default=200000)
    ap.add_argument("--scale_mult",type=float,default=1.0)
    ap.add_argument("--opacity",type=float,default=0.99)
    ap.add_argument("--default_rgb",type=float,nargs=3,default=[0.6,0.6,0.6])
    args=ap.parse_args()

    m=trimesh.load(args.mesh,process=False,force="mesh")
    if m.is_empty or len(m.faces)==0:
        raise SystemExit(f"빈 메시: {args.mesh}")
    pts,fidx=trimesh.sample.sample_surface(m,args.n_samples)
    pts=np.asarray(pts,np.float64); normals=np.asarray(m.face_normals[fidx],np.float64)

    # color: face별 vertex color 평균 → 샘플
    if hasattr(m.visual,"vertex_colors") and m.visual.vertex_colors is not None \
       and len(m.visual.vertex_colors)==len(m.vertices):
        vc=np.asarray(m.visual.vertex_colors)[:,:3].astype(np.float64)/255.0
        fc=vc[m.faces].mean(1)            # (F,3)
        rgb=fc[fidx]
    else:
        rgb=np.tile(np.array(args.default_rgb),(len(pts),1))

    # scale: 표면 타일링 반경
    area=float(m.area) if m.area>0 else 1.0
    r=args.scale_mult*np.sqrt(area/max(args.n_samples,1))
    log_r=np.log(max(r,1e-6))

    # 활성화 역변환
    f_dc=(rgb-0.5)/C0
    op_logit=np.log(args.opacity/(1-args.opacity))
    quat=quats_from_normals(normals)

    N=len(pts)
    names=field_names()
    dt=np.dtype([(n,"f4") for n in names])
    arr=np.zeros(N,dtype=dt)
    arr["x"],arr["y"],arr["z"]=pts[:,0],pts[:,1],pts[:,2]
    arr["nx"],arr["ny"],arr["nz"]=normals[:,0],normals[:,1],normals[:,2]
    arr["f_dc_0"],arr["f_dc_1"],arr["f_dc_2"]=f_dc[:,0],f_dc[:,1],f_dc[:,2]
    # f_rest_*, id_*, desc_* = 0 (이미 zeros)
    arr["opacity"]=op_logit
    arr["scale_0"]=log_r; arr["scale_1"]=log_r
    arr["rot_0"],arr["rot_1"],arr["rot_2"],arr["rot_3"]=quat[:,0],quat[:,1],quat[:,2],quat[:,3]

    el=PlyElement.describe(arr,"vertex")
    PlyData([el],text=False).write(args.out)
    print(f"surfels {N}  scale r={r:.4f}m  props={len(names)}  → {args.out}")


if __name__=="__main__":
    main()
