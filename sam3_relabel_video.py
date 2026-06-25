#!/usr/bin/env python3
"""
축1 ❶❷ (video) — SAM3 video predictor 기반 instance re-labeling (재설계: 임계 robust).

설계 원리 (probe로 검증: SAM3 concept-video는 streaming detection):
  1) concept당 single-prompt(frame 0) + propagate → SAM3가 비디오 전체에서 인스턴스를
     자동 검출·추적(out_obj_ids). multi-keyframe 재프롬프트 제거(중복 생성 원인 제거).
  2) instance unification = '비디오 고유 신호' 기반(임계 의존 최소):
     - 같은 concept: 두 track이 *같은 프레임에 공존(co-occur)*하면 DISTINCT (하드 제약,
       임계 없음 — 한 객체가 동시에 두 곳일 수 없음 → 인접 distinct를 *원천 보호*).
       *시간적으로 배타적*이고 3D footprint가 겹치면 → re-identification 병합.
     - 다른 concept(cushion/pillow 등): 3D 겹침이 크면 같은 객체로 병합(synonym).
  3) 구조물 concept 제외 + min_track 필터.

출력: <out_root>/<gid>/<stem>.png (마스크) + points3d.ply (init) → prepare_folder 입력.

실행 (sam3 env):
    LD_LIBRARY_PATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python sam3_relabel_video.py \
        --frames data/replica_room0/images --img_ext .jpg \
        --colmap_dir data/replica_room0/sparse/0 \
        --vocab_json /home/elicer/sam3/vocab.json \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 10 --min_area 0.003 --min_track 3 \
        --reid_th 0.3 --cross_th 0.3 \
        --exclude_concepts "door,blind,vent,window,wall,floor,ceiling,light switch,thermostat" \
        --out_root ~/relabel_video_room0
"""
import argparse, glob, json, os, struct, tempfile
from collections import Counter, defaultdict
import numpy as np, torch
from PIL import Image


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

def write_ply(path, xyz):
    """per-object 초기 포인트(COLMAP subset)를 binary PLY로 저장 → prepare_folder가 points3d.ply로."""
    xyz = np.asarray(xyz, np.float32); n = len(xyz)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")])
        a = np.empty(n, dt)
        if n: a["x"],a["y"],a["z"]=xyz[:,0],xyz[:,1],xyz[:,2]; a["r"]=a["g"]=a["b"]=180
        f.write(a.tobytes())


def load_points3D(d):
    pb=os.path.join(d,"points3D.bin"); pt=os.path.join(d,"points3D.txt"); xyz=[]
    if os.path.isfile(pb):
        with open(pb,"rb") as f:
            n=struct.unpack("<Q",f.read(8))[0]
            for _ in range(n):
                struct.unpack("<Q",f.read(8)); x,y,z=struct.unpack("<3d",f.read(24))
                f.read(3); struct.unpack("<d",f.read(8)); tl=struct.unpack("<Q",f.read(8))[0]; f.read(8*tl)
                xyz.append((x,y,z))
    elif os.path.isfile(pt):
        for ln in open(pt):
            if ln.startswith("#") or not ln.strip(): continue
            t=ln.split(); xyz.append((float(t[1]),float(t[2]),float(t[3])))
    return np.array(xyz,np.float64)


def jac(a, b):
    if not a or not b: return 0.0
    i = len(a & b); return i / (len(a) + len(b) - i)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames",required=True); ap.add_argument("--img_ext",default=".jpg")
    ap.add_argument("--colmap_dir",required=True)
    ap.add_argument("--vocab_json",default=None); ap.add_argument("--vocab",default=None)
    ap.add_argument("--bpe",default=None)
    ap.add_argument("--stride",type=int,default=10,help="SAM3 propagate 프레임 subsample(메모리/속도)")
    ap.add_argument("--prompt_frame",type=int,default=0,help="concept를 프롬프트할 단일 프레임 인덱스")
    ap.add_argument("--min_area",type=float,default=0.003,help="프레임 마스크 최소 면적(노이즈 제거)")
    ap.add_argument("--min_track",type=int,default=3,help="유효 객체 최소 관측 프레임 수")
    ap.add_argument("--reid_th",type=float,default=0.3,
                    help="같은 concept: 시간 배타적 track의 re-id 병합 3D Jaccard 임계")
    ap.add_argument("--cross_th",type=float,default=0.3,
                    help="다른 concept: synonym 병합 3D Jaccard 임계")
    ap.add_argument("--exclude_concepts",default="")
    ap.add_argument("--out_root",required=True)
    args=ap.parse_args(); os.makedirs(args.out_root,exist_ok=True)

    VOCAB=(json.load(open(args.vocab_json))["vocab"] if args.vocab_json
           else [v.strip() for v in args.vocab.split(",")])
    cams=load_cams(args.colmap_dir); P3=load_points3D(args.colmap_dir)
    print(f"vocab={len(VOCAB)} cams={len(cams)} points3D={len(P3)}")

    # 정수명 심링크 + idx→stem
    src=sorted(glob.glob(os.path.join(args.frames,f"*{args.img_ext}")))[::args.stride]
    tmp=tempfile.mkdtemp(prefix="sam3relabel_"); idx2stem=[]
    for i,f in enumerate(src):
        os.symlink(os.path.abspath(f),os.path.join(tmp,f"{i}.jpg"))
        idx2stem.append(os.path.splitext(os.path.basename(f))[0])
    N=len(src); print(f"frames={N} → {tmp}  (single-prompt @frame {args.prompt_frame}, streaming)")

    # 프레임별 3D점 투영 사전계산(stem→(gi,ui,vi))
    proj={}
    for i,stem in enumerate(idx2stem):
        cam=cams.get(stem)
        if cam is None: continue
        Xc=P3@cam["R"].T+cam["t"]; z=Xc[:,2]; ok=z>1e-6
        u=cam["fx"]*Xc[:,0]/np.where(ok,z,1)+cam["cx"]; v=cam["fy"]*Xc[:,1]/np.where(ok,z,1)+cam["cy"]
        ui=np.round(u).astype(np.int64); vi=np.round(v).astype(np.int64)
        inb=ok&(ui>=0)&(ui<cam["W"])&(vi>=0)&(vi<cam["H"]); gi=np.where(inb)[0]
        proj[stem]=(gi,ui,vi)

    from sam3.model_builder import build_sam3_video_predictor
    try:
        predictor=build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()),bpe_path=args.bpe) \
                  if args.bpe else build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    except TypeError:
        predictor=build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))

    def propagate(sid):
        out={}
        for r in predictor.handle_stream_request(dict(type="propagate_in_video",session_id=sid)):
            out[r["frame_index"]]=r["outputs"]
        return out

    # ── concept별 single-prompt streaming → track 수집 ──
    # track = dict(concept, masks{stem:mask}, frames:set(stem), sig:set(int), score)
    tracks=[]
    pf=int(np.clip(args.prompt_frame,0,N-1))
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        sid=predictor.handle_request(dict(type="start_session",resource_path=tmp))["session_id"]
        for c in VOCAB:
            predictor.handle_request(dict(type="reset_session",session_id=sid))
            predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=pf,text=c))  # single prompt
            opf=propagate(sid)
            byid={}
            for fidx,o in opf.items():
                stem=idx2stem[fidx]; ids=np.asarray(o["out_obj_ids"]).reshape(-1)
                masks=np.asarray(o["out_binary_masks"]); probs=np.asarray(o["out_probs"]).reshape(-1)
                for k,oid in enumerate(ids):
                    m=masks[k]
                    if m.mean()<args.min_area: continue
                    d=byid.setdefault(int(oid),{"masks":{},"score":0.0})
                    d["masks"][stem]=m>0; d["score"]=max(d["score"],float(probs[k]))
            kept=0
            for oid,d in byid.items():
                if len(d["masks"])<args.min_track: continue
                sig=set()
                for stem,m in d["masks"].items():
                    if stem not in proj: continue
                    gi,ui,vi=proj[stem]; inm=m[vi[gi],ui[gi]]; sig|=set(gi[inm].tolist())
                tracks.append(dict(concept=c,masks=d["masks"],frames=set(d["masks"].keys()),
                                   sig=sig,score=d["score"]))
                kept+=1
            print(f"  [{c}] SAM3 ids={len(byid)} → valid tracks={kept}")
        predictor.handle_request(dict(type="close_session",session_id=sid))
    try: predictor.shutdown()
    except Exception: pass
    print(f"\nnative tracks(전 concept): {len(tracks)}")

    # ── co-occurrence 기반 instance unification (union-find) ──
    parent=list(range(len(tracks)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y): parent[find(x)]=find(y)

    n_reid=n_cross=0
    for i in range(len(tracks)):
        for j in range(i+1,len(tracks)):
            A,B=tracks[i],tracks[j]
            j3=jac(A["sig"],B["sig"])
            if A["concept"]==B["concept"]:
                cooccur=bool(A["frames"] & B["frames"])
                if (not cooccur) and j3>args.reid_th:      # 시간 배타 + 같은 위치 = re-id
                    union(i,j); n_reid+=1
                # cooccur → DISTINCT (병합 금지, 하드 제약)
            else:
                if j3>args.cross_th:                        # 다른 concept synonym
                    union(i,j); n_cross+=1
    groups=defaultdict(list)
    for i in range(len(tracks)): groups[find(i)].append(i)
    print(f"unification: re-id 병합={n_reid}, cross-concept 병합={n_cross} → 그룹 {len(groups)}")

    # ── 그룹 → 객체 (masks OR, sig union, concept 다수결) ──
    excl={c.strip() for c in args.exclude_concepts.split(",") if c.strip()}
    objs=[]
    for members in groups.values():
        masks={}; sig=set(); concepts=Counter()
        for mi in members:
            t=tracks[mi]; sig|=t["sig"]; concepts[t["concept"]]+=1
            for stem,msk in t["masks"].items():
                masks[stem]=(masks[stem]|msk) if stem in masks else msk
        if concepts.most_common(1)[0][0] in excl: continue
        if len(masks)<args.min_track: continue
        objs.append(dict(masks=masks,sig=sig,concepts=concepts))
    objs.sort(key=lambda o:-len(o["masks"]))
    print(f"구조물 제외+min_track 후 유효 객체: {len(objs)}")

    # ── 저장 ──
    for gid,o in enumerate(objs):
        od=os.path.join(args.out_root,str(gid)); os.makedirs(od,exist_ok=True)
        for stem,m in o["masks"].items():
            Image.fromarray((m*255).astype(np.uint8)).save(os.path.join(od,f"{stem}.png"))
        write_ply(os.path.join(od,"points3d.ply"),
                  P3[sorted(o["sig"])].astype(np.float32) if o["sig"] else np.zeros((0,3),np.float32))
        print(f"  obj{gid}: frames={len(o['masks'])} init_pts={len(o['sig'])} "
              f"concept~{o['concepts'].most_common(1)[0][0]}")
    print(f"저장: {args.out_root}/<gid>/<stem>.png + points3d.ply")
    print("판정: single-prompt streaming + co-occurrence unification — "
          "인접 distinct는 공존 제약으로 보호, re-id만 3D로 병합(임계 robust).")


if __name__=="__main__":
    main()