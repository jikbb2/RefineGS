#!/usr/bin/env python3
"""
축2 (교정) — mask SELECTION + 3-way 일관성 평가.

GS reproject(마스크 합성) 폐기. 각 뷰에서 SAM의 *실제* 마스크 중 3D voting 결과
(obj_support)와 가장 잘 맞는 것을 *선택* → SAM 품질 유지 + granularity/instance 일관.

뷰별로 3가지를 GT instance와 IoU 비교:
  - naive   : SAM 자체 confidence 최고 마스크 (축2 없을 때; granularity·instance 흔들림)
  - axis2   : obj_support와 3D 겹침 최대인 SAM 마스크 (우리 선택)
  - oracle  : GT와 IoU 최대 (상한, 참고)
판정: axis2 가 naive보다 mean↑·std↓·min↑ 이고 oracle에 근접하면 축2 가치.

규약: stage3(autocast bf16). sam3 env. selected.npz(xyz,obj_support)+COLMAP 필요.

실행:
    conda activate sam3
    LD_LIBRARY_PATH= python axis2_select_and_eval.py \
        --in_dir ~/axis2_vote_98 \
        --colmap_dir /home/elicer/RefineGS/data/replica_room0/masks/98/sparse/0 \
        --images_dir /home/elicer/RefineGS/data/replica_room0/masks/98/images \
        --gt_dir /home/elicer/room_0/imap/00/semantic_instance --gt_id 7 \
        --concept table --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 10 --out_dir ~/axis2_selected_98
"""
import argparse, glob, os, re, struct
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ── COLMAP (axis2_vote와 동일) ──
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


def to_bool(m):
    if hasattr(m,"detach"): m=m.detach().float().cpu().numpy()
    m=np.squeeze(np.asarray(m))
    if m.ndim==2: m=m[None]
    return [x>0.5 if x.dtype!=bool else x for x in m]

def resize_to(mask,hw):
    if mask.shape==hw: return mask
    return np.array(Image.fromarray((mask*255).astype(np.uint8)).resize((hw[1],hw[0]),Image.NEAREST))>127

def iou(a,b):
    u=np.logical_or(a,b).sum(); return np.logical_and(a,b).sum()/u if u else 0.0

def visible(xyz,cam):
    Xc=xyz@cam["R"].T+cam["t"]; z=Xc[:,2]; ok=z>1e-6
    u=cam["fx"]*Xc[:,0]/np.where(ok,z,1)+cam["cx"]; v=cam["fy"]*Xc[:,1]/np.where(ok,z,1)+cam["cy"]
    W,H=cam["W"],cam["H"]; ui=np.round(u).astype(np.int64); vi=np.round(v).astype(np.int64)
    inb=ok&(ui>=0)&(ui<W)&(vi>=0)&(vi<H); idx=np.where(inb)[0]
    if len(idx)==0: return idx,ui,vi
    pid=vi[idx]*W+ui[idx]; o=np.argsort(z[idx]); _,fi=np.unique(pid[o],return_index=True)
    return idx[o[fi]],ui,vi


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_dir",required=True)
    ap.add_argument("--colmap_dir",required=True)
    ap.add_argument("--images_dir",required=True)
    ap.add_argument("--gt_dir",required=True); ap.add_argument("--gt_id",type=int,required=True)
    ap.add_argument("--concept",default="table")
    ap.add_argument("--bpe",default=None); ap.add_argument("--stride",type=int,default=10)
    ap.add_argument("--out_dir",default=os.path.expanduser("~/axis2_selected"))
    args=ap.parse_args(); os.makedirs(args.out_dir,exist_ok=True)

    d=np.load(os.path.join(args.in_dir,"selected.npz"))
    xyz=d["xyz"]; obj=d["obj_support"]
    cams=load_cams(args.colmap_dir)
    mk=dict(enable_inst_interactivity=True)
    if args.bpe: mk["bpe_path"]=args.bpe
    model=build_sam3_image_model(**mk); proc=Sam3Processor(model)

    imgs=sorted(glob.glob(os.path.join(args.images_dir,"*")))[::args.stride]
    naive,ax,ora=[],[],[]
    print(f"{'frame':>12} {'naive':>8} {'axis2':>8} {'oracle':>8}")
    print("-"*42)
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        for ip in imgs:
            stem=os.path.splitext(os.path.basename(ip))[0]
            cam=cams.get(stem)
            gtp=os.path.join(args.gt_dir,f"semantic_instance_{int(re.sub(chr(92)+'D','',stem))}.png")
            if cam is None or not os.path.exists(gtp): continue
            gt=(np.array(Image.open(gtp)).astype(np.int64)==args.gt_id)
            if gt.sum()==0: continue
            image=Image.open(ip).convert("RGB"); state=proc.set_image(image)
            out=proc.set_text_prompt(state=state,prompt=args.concept)
            masks=to_bool(out.get("masks")) if isinstance(out,dict) else []
            scores=np.asarray(out.get("scores")).reshape(-1) if (isinstance(out,dict) and out.get("scores") is not None) else None
            if not masks: continue
            vis,ui,vi=visible(xyz,cam)
            visset=np.zeros(len(xyz),bool); visset[vis]=True
            vobj=obj&visset
            # 후보별 3D score (obj_support 겹침) & GT IoU
            s3d=[]; gtio=[]
            for m in masks:
                mm=resize_to(m,(cam["H"],cam["W"]))
                cov=np.zeros(len(xyz),bool); sel=mm[vi[vis],ui[vis]]; cov[vis[sel]]=True
                s3d.append(iou(cov,vobj))
                gtio.append(iou(resize_to(m,gt.shape),gt))
            i_ax=int(np.argmax(s3d))
            i_na=int(np.argmax(scores)) if scores is not None and len(scores)==len(masks) else int(np.argmax([m.sum() for m in masks]))
            naive.append(gtio[i_na]); ax.append(gtio[i_ax]); ora.append(max(gtio))
            Image.fromarray((resize_to(masks[i_ax],(cam["H"],cam["W"]))*255).astype(np.uint8)).save(
                os.path.join(args.out_dir,f"{stem}.png"))
            print(f"{stem:>12} {gtio[i_na]:>8.3f} {gtio[i_ax]:>8.3f} {max(gtio):>8.3f}")
    def stat(a):
        a=np.array(a); return f"mean={a.mean():.3f} std={a.std():.3f} min={a.min():.3f} n={len(a)}"
    print("-"*42)
    print(f"naive (SAM conf) : {stat(naive)}")
    print(f"axis2 (3D select): {stat(ax)}")
    print(f"oracle (best→GT) : {stat(ora)}")
    print(f"\n저장 선택 마스크: {args.out_dir}")
    print("판정: axis2가 naive보다 std↓·min↑(일관성)이고 oracle에 근접하면 축2 mask-selection 가치.")


if __name__=="__main__":
    main()
