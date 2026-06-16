#!/usr/bin/env python3
"""
축1 ❶❷ (video) — SAM3 video predictor 기반 re-labeling (임계 최소화).

프레임=video 궤적. 개념별 add_prompt(text)→propagate_in_video로 *native 일관 track-ID*
(out_obj_ids ↔ out_binary_masks). 프레임 간 일관성은 임계 튜닝 없이 video predictor가 보장.
여러 keyframe에 프롬프트해 후반 등장 객체 발견. cross-concept 중복(cushion/pillow 같은
객체)만 COLMAP 3D overlap으로 가볍게 정리. 구조물 concept 제외.
출력: <out_root>/<gid>/<stem>.png (prepare_folder 입력).

probe 확인 출력 구조:
  outputs[frame] = {out_obj_ids(N,), out_probs(N,), out_boxes_xywh(N,4), out_binary_masks(N,H,W), frame_stats}

실행 (sam3 env):
    conda activate sam3
    LD_LIBRARY_PATH= python sam3_relabel_video.py \
        --frames data/replica_room0/images --img_ext .JPEG \
        --colmap_dir data/replica_room0/sparse/0 \
        --vocab_json /home/elicer/sam3/vocab.json \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 2 --n_prompt_frames 5 --min_area 0.003 --min_track 3 \
        --dedup_th 0.3 --exclude_concepts "door,blind,vent,window,wall,floor,ceiling,light switch,thermostat" \
        --out_root ~/relabel_video_room0
"""
import argparse, glob, json, os, struct, tempfile
from collections import Counter
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


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames",required=True); ap.add_argument("--img_ext",default=".JPEG")
    ap.add_argument("--colmap_dir",required=True)
    ap.add_argument("--vocab_json",default=None); ap.add_argument("--vocab",default=None)
    ap.add_argument("--bpe",default=None)
    ap.add_argument("--stride",type=int,default=2)
    ap.add_argument("--n_prompt_frames",type=int,default=5,help="개념을 프롬프트할 keyframe 수(후반 등장 객체 발견)")
    ap.add_argument("--min_area",type=float,default=0.003)
    ap.add_argument("--min_track",type=int,default=3,help="유효 객체로 인정할 최소 관측 프레임")
    ap.add_argument("--dedup_th",type=float,default=0.3,help="cross-concept 중복 병합 3D Jaccard 임계")
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
    N=len(src); print(f"frames={N} → {tmp}")
    kf=[int(round(x)) for x in np.linspace(0,N-1,max(1,args.n_prompt_frames))]
    kf=sorted(set(kf)); print(f"prompt keyframes={kf}")

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

    tracks=[]  # dict(concept, masks{stem:mask}, sig:set, score)
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        sid=predictor.handle_request(dict(type="start_session",resource_path=tmp))["session_id"]
        for c in VOCAB:
            predictor.handle_request(dict(type="reset_session",session_id=sid))
            for fidx in kf:
                predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=fidx,text=c))
            opf=propagate(sid)
            # obj_id별 per-frame 마스크 수집
            byid={}
            for fidx,o in opf.items():
                stem=idx2stem[fidx]; ids=np.asarray(o["out_obj_ids"]).reshape(-1)
                masks=np.asarray(o["out_binary_masks"]); probs=np.asarray(o["out_probs"]).reshape(-1)
                for k,oid in enumerate(ids):
                    m=masks[k]
                    if m.mean()<args.min_area: continue
                    byid.setdefault(int(oid),{"masks":{},"score":0.0})
                    byid[int(oid)]["masks"][stem]=m>0
                    byid[int(oid)]["score"]=max(byid[int(oid)]["score"],float(probs[k]))
            for oid,d in byid.items():
                if len(d["masks"])<args.min_track: continue
                sig=set()
                for stem,m in d["masks"].items():
                    if stem not in proj: continue
                    gi,ui,vi=proj[stem]; inm=m[vi[gi],ui[gi]]; sig|=set(gi[inm].tolist())
                tracks.append(dict(concept=c,masks=d["masks"],sig=sig,score=d["score"]))
        predictor.handle_request(dict(type="close_session",session_id=sid))
    try: predictor.shutdown()
    except Exception: pass
    print(f"native tracks(전 concept): {len(tracks)}")

    # cross-concept 중복 병합 (3D Jaccard) — within-concept은 이미 native라 dedup 불필요
    def jac(a,b):
        if not a or not b: return 0.0
        i=len(a&b); return i/(len(a)+len(b)-i)
    tracks.sort(key=lambda t:-len(t["sig"]))
    merged=[]
    for t in tracks:
        hit=None
        for m in merged:
            if jac(t["sig"],m["sig"])>args.dedup_th: hit=m; break
        if hit:
            for stem,msk in t["masks"].items():
                if stem not in hit["masks"] or msk.sum()>hit["masks"][stem].sum(): hit["masks"][stem]=msk
            hit["sig"]|=t["sig"]; hit["concepts"][t["concept"]]+=1
        else:
            merged.append(dict(masks=dict(t["masks"]),sig=set(t["sig"]),
                               concepts=Counter([t["concept"]])))
    print(f"cross-concept 병합 후: {len(merged)}")

    excl={c.strip() for c in args.exclude_concepts.split(",") if c.strip()}
    objs=[m for m in merged if m["concepts"].most_common(1)[0][0] not in excl and len(m["masks"])>=args.min_track]
    objs.sort(key=lambda m:-len(m["masks"]))
    print(f"구조물 제외+min_track 후 유효 객체: {len(objs)}")
    for gid,m in enumerate(objs):
        od=os.path.join(args.out_root,str(gid)); os.makedirs(od,exist_ok=True)
        for stem,msk in m["masks"].items():
            Image.fromarray((msk*255).astype(np.uint8)).save(os.path.join(od,f"{stem}.png"))
        # per-object 초기 포인트(이 객체의 COLMAP 3D 점) → prepare_folder가 points3d.ply로 변환
        pts = P3[sorted(m["sig"])] if m["sig"] else np.zeros((0,3))
        write_ply(os.path.join(od,"points.ply"), pts)
        print(f"  obj{gid}: frames={len(m['masks'])} init_pts={len(pts)} "
              f"concept~{m['concepts'].most_common(1)[0][0]}")
    print(f"\n저장: {args.out_root}/<gid>/<stem>.png")
    print("판정: native 추적이라 same-object over-fragmentation이 3D-voting 버전보다 줄고, "
          "유효 객체가 GT에 근접하면 video re-labeling 확정.")


if __name__=="__main__":
    main()
