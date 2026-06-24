#!/usr/bin/env python3
"""
축1 ❶❷ — SAM3 기반 re-labeling + conflation 해결 (3D voting 추적).

SAM3가 프레임별로 인스턴스를 *분리*해 반환(확인됨). 각 인스턴스를 COLMAP 3D점에
투영해 3D 점집합 signature로 만들고, 뷰 간 Jaccard로 클러스터링 → 같은 물리 객체끼리
묶어 일관 track-ID 부여. (파편 병합; 한 인스턴스가 2 클러스터에 크게 걸치면 분리.)
출력: per-object per-view 이진 마스크 → <out_root>/<gid>/<stem>.png (prepare_folder 입력).

규약: stage3(autocast bf16, mask squeeze). scene-agnostic(frames/colmap/vocab 인자).
의존: numpy, torch, PIL, sam3. (COLMAP points3D는 bin/txt 모두 지원)

실행 (sam3 env):
    conda activate sam3
    LD_LIBRARY_PATH= python sam3_relabel.py \
        --frames <images_dir> --img_ext .jpg \
        --colmap_dir <sparse/0> \
        --vocab_json /home/elicer/sam3/vocab.json \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 2 --min_area 0.003 --jac_th 0.2 --min_track 3 \
        --out_root ~/relabel_out
"""
import argparse, glob, json, os, re, struct
import numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ── COLMAP ──
def _q2r(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                     [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                     [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])

def _read_bin(d):
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

def _read_txt(d):
    cams={}
    for ln in open(os.path.join(d,"cameras.txt")):
        if ln.startswith("#") or not ln.strip(): continue
        t=ln.split(); cid=int(t[0]); model=t[1]; w,h=int(t[2]),int(t[3]); pr=list(map(float,t[4:]))
        if model=="PINHOLE": fx,fy,cx,cy=pr[:4]
        else: fx=fy=pr[0]; cx,cy=pr[1],pr[2]
        cams[cid]=(fx,fy,cx,cy,w,h)
    imgs=[]; L=[l for l in open(os.path.join(d,"images.txt")) if not l.startswith("#")]
    for i in range(0,len(L),2):
        t=L[i].split()
        if len(t)<10: continue
        q=list(map(float,t[1:5])); tv=np.array(list(map(float,t[5:8])))
        imgs.append({"R":_q2r(q),"t":tv,"camera_id":int(t[8]),"name":t[9]})
    return cams,imgs

def load_cams(d):
    cams,imgs=(_read_bin(d) if os.path.isfile(os.path.join(d,"images.bin")) else _read_txt(d))
    out={}
    for im in imgs:
        fx,fy,cx,cy,w,h=cams[im["camera_id"]]
        out[os.path.splitext(os.path.basename(im["name"]))[0]]=dict(R=im["R"],t=im["t"],fx=fx,fy=fy,cx=cx,cy=cy,W=w,H=h)
    return out

def load_points3D(d):
    pb=os.path.join(d,"points3D.bin"); pt=os.path.join(d,"points3D.txt")
    xyz=[]
    if os.path.isfile(pb):
        with open(pb,"rb") as f:
            n=struct.unpack("<Q",f.read(8))[0]
            for _ in range(n):
                struct.unpack("<Q",f.read(8)); x,y,z=struct.unpack("<3d",f.read(24))
                f.read(3); struct.unpack("<d",f.read(8))
                tl=struct.unpack("<Q",f.read(8))[0]; f.read(8*tl)
                xyz.append((x,y,z))
    elif os.path.isfile(pt):
        for ln in open(pt):
            if ln.startswith("#") or not ln.strip(): continue
            t=ln.split(); xyz.append((float(t[1]),float(t[2]),float(t[3])))
    return np.array(xyz, np.float64)


def to_bool(m):
    if hasattr(m,"detach"): m=m.detach().float().cpu().numpy()
    m=np.squeeze(np.asarray(m))
    if m.ndim==2: m=m[None]
    return [x>0.5 if x.dtype!=bool else x for x in m]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames",required=True); ap.add_argument("--img_ext",default=".jpg")
    ap.add_argument("--colmap_dir",required=True)
    ap.add_argument("--vocab_json",default=None); ap.add_argument("--vocab",default=None)
    ap.add_argument("--bpe",default=None)
    ap.add_argument("--stride",type=int,default=2)
    ap.add_argument("--min_area",type=float,default=0.003,help="인스턴스 최소 면적(프레임 비율)")
    ap.add_argument("--jac_th",type=float,default=0.2,help="3D signature Jaccard 클러스터 임계")
    ap.add_argument("--merge_th",type=float,default=0.3,
                    help="병합 패스: Jaccard(|A∩B|/|A∪B|) 이 이상이면 같은 객체 파편으로 병합")
    ap.add_argument("--min_track",type=int,default=3,help="유효 객체로 인정할 최소 관측 뷰")
    ap.add_argument("--exclude_concepts",default="",
                    help="쉼표 구분, 이 concept이 dominant인 객체는 출력 제외(예: door,blind,vent,window,wall,floor,ceiling)")
    ap.add_argument("--out_root",required=True)
    args=ap.parse_args(); os.makedirs(args.out_root,exist_ok=True)

    VOCAB=(json.load(open(args.vocab_json))["vocab"] if args.vocab_json
           else [v.strip() for v in args.vocab.split(",")])
    cams=load_cams(args.colmap_dir); P3=load_points3D(args.colmap_dir)
    print(f"vocab={len(VOCAB)} cams={len(cams)} points3D={len(P3)}")
    if len(P3)==0: print("[ERROR] points3D 없음 — sparse에 points3D.bin/txt 필요"); return

    model=build_sam3_image_model(**({"bpe_path":args.bpe} if args.bpe else {})); proc=Sam3Processor(model)
    frames=sorted(glob.glob(os.path.join(args.frames,f"*{args.img_ext}")))[::args.stride]
    print(f"frames={len(frames)}")

    obs=[]  # (stem, concept, mask(bool HxW), pt_idx set(frozenset))
    for fp in frames:
        stem=os.path.splitext(os.path.basename(fp))[0]; cam=cams.get(stem)
        if cam is None: continue
        img=Image.open(fp).convert("RGB"); W,H=img.size
        # 3D점 투영(이 프레임)
        Xc=P3@cam["R"].T+cam["t"]; z=Xc[:,2]; ok=z>1e-6
        u=cam["fx"]*Xc[:,0]/np.where(ok,z,1)+cam["cx"]; v=cam["fy"]*Xc[:,1]/np.where(ok,z,1)+cam["cy"]
        ui=np.round(u).astype(np.int64); vi=np.round(v).astype(np.int64)
        inb=ok&(ui>=0)&(ui<W)&(vi>=0)&(vi<H); gi=np.where(inb)[0]
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
            st=proc.set_image(img)
            for c in VOCAB:
                out=proc.set_text_prompt(state=st,prompt=c)
                for m in to_bool(out.get("masks")) if isinstance(out,dict) else []:
                    if m.shape!=(H,W) or m.mean()<args.min_area: continue
                    inm=m[vi[gi],ui[gi]]; idset=frozenset(gi[inm].tolist())
                    if len(idset)>=3: obs.append((stem,c,m,idset))
    print(f"인스턴스 관측: {len(obs)}")

    # 3D Jaccard 클러스터링 (greedy, union 갱신)
    def jac(a,b):
        if not a or not b: return 0.0
        i=len(a&b); return i/(len(a)+len(b)-i)
    clusters=[]  # dict(sig=set, members=[obs_idx], views=set, concepts=Counter)
    from collections import Counter
    order=sorted(range(len(obs)),key=lambda i:-len(obs[i][3]))
    for i in order:
        best,bj=-1,0.0
        for ci,c in enumerate(clusters):
            j=jac(obs[i][3],c["sig"])
            if j>bj: bj,best=j,ci
        if best>=0 and bj>args.jac_th:
            c=clusters[best]; c["members"].append(i); c["views"].add(obs[i][0])
            c["sig"]=c["sig"]|obs[i][3]; c["concepts"][obs[i][1]]+=1
        else:
            clusters.append(dict(sig=set(obs[i][3]),members=[i],views={obs[i][0]},
                                 concepts=Counter([obs[i][1]])))
    # ── 병합 패스: over-fragmentation 해소 (Jaccard 대칭, concept 무관) ──
    # Jaccard라 같은 객체 파편(상호겹침 큼)만 합치고, 작은-객체-in-큰-평면(겹침 작음)은 안 합침.
    def jacm(a, b):
        if not a or not b: return 0.0
        i = len(a & b); return i / (len(a) + len(b) - i)
    n_before = len(clusters)
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if jacm(clusters[i]["sig"], clusters[j]["sig"]) > args.merge_th:
                    clusters[i]["sig"] |= clusters[j]["sig"]
                    clusters[i]["members"] += clusters[j]["members"]
                    clusters[i]["views"] |= clusters[j]["views"]
                    clusters[i]["concepts"] += clusters[j]["concepts"]
                    del clusters[j]; changed = True; break
            if changed: break
    print(f"병합: {n_before} → {len(clusters)} 클러스터 (Jaccard>{args.merge_th})")

    excl = {c.strip() for c in args.exclude_concepts.split(",") if c.strip()}
    objs=[c for c in clusters if len(c["views"])>=args.min_track]
    if excl:
        kept=[c for c in objs if c["concepts"].most_common(1)[0][0] not in excl]
        print(f"구조물 제외({sorted(excl)}): {len(objs)} → {len(kept)} 객체")
        objs=kept
    objs.sort(key=lambda c:-len(c["views"]))
    print(f"유효 객체(views>={args.min_track}) {len(objs)}")

    # 출력: 객체별 per-view 마스크 (그 프레임에 그 객체의 인스턴스가 있으면 저장)
    for gid,c in enumerate(objs):
        od=os.path.join(args.out_root,str(gid)); os.makedirs(od,exist_ok=True)
        per_view={}
        for mi in c["members"]:
            stem,_,m,_=obs[mi]
            if stem not in per_view or m.sum()>per_view[stem].sum(): per_view[stem]=m
        for stem,m in per_view.items():
            Image.fromarray((m*255).astype(np.uint8)).save(os.path.join(od,f"{stem}.png"))
        top=c["concepts"].most_common(1)[0][0]
        print(f"  obj{gid}: views={len(c['views'])} concept~{top} masks={len(per_view)}")
    print(f"\n저장: {args.out_root}/<gid>/<stem>.png  (prepare_folder 입력)")
    print("판정: 유효 객체 수가 GT 객체 수에 근접하고, 각 객체가 뷰 간 일관(views 충분)하면 re-labeling OK.")


if __name__=="__main__":
    main()
