#!/usr/bin/env python3
"""
축1 ❶❷ (video) — SAM3 video predictor 기반 instance re-labeling (재설계 v2: depth-dense sig).

설계 원리 (probe로 검증: SAM3 concept-video는 streaming detection):
  1) concept당 single-prompt(frame 0) + propagate → SAM3가 비디오 전체에서 인스턴스를
     자동 검출·추적(out_obj_ids). multi-keyframe 재프롬프트 제거(중복 생성 원인 제거).
  2) 3D signature(sig) = ★GT-depth dense back-projection★ (v2 변경):
     - 마스크 픽셀을 GT depth로 역투영 → 객체 *앞면 실제 표면점*만 → 배경(벽/바닥) 원천 제거.
     - voxel 해시 + multi-view consistency(여러 프레임에서 일관되게 찍힌 voxel만) → 노이즈 제거.
     - 이전 v1은 sparse COLMAP points3D를 마스크에 투영해서 ray 상 배경(벽/바닥)이 새어들어와
       purity를 떨어뜨리고 re-id 병합을 거칠게 만들었음 → dense depth로 교체.
  3) instance unification = '비디오 고유 신호' 기반(임계 의존 최소):
     - 같은 concept: 두 track이 *같은 프레임에 공존(co-occur)*하면 DISTINCT (하드 제약, 임계 없음).
       *시간적으로 배타적*이고 3D footprint(voxel)가 겹치면 → re-identification 병합.
     - 다른 concept(cushion/pillow 등): 3D 겹침이 크면 같은 객체로 병합(synonym).
  4) 구조물 concept 제외 + min_track 필터. (작은 객체 coverage: --min_area/--min_track 완화)

★ v2.1 메모리 패치: --window N ★
  SAM3 propagate 는 한 concept당 비디오 *전체* 프레임을 스트리밍하며 프레임별 버퍼를 GPU에 누적한다.
  stride 를 낮춰(dense) 프레임 수가 늘면 CUDACachingAllocator/NVML assert(=OOM)로 죽는다.
  --window N 을 주면 프레임을 N개 단위 window로 잘라 window마다 세션을 새로 열고 닫아
  (start_session → 모든 concept propagate → close_session → empty_cache) GPU 메모리 상한을 고정한다.
  window 경계로 쪼개진 동일 객체는 프레임이 겹치지 않으므로 아래 3D voxel unification 이
  '시간 배타 → re-id 병합'으로 자동으로 다시 이어붙인다(다운스트림 무변경).
  --window 0(기본)이면 기존과 동일(전체 한 번).
  window로 쪼갤 때 per-track min_track 필터는 완화(1)하고, 최종 min_track 은 병합된 객체 단위(아래)로 적용.

출력: <out_root>/<gid>/<stem>.png (마스크) + points3d.ply (depth voxel 센터, init) → prepare_folder 입력.

실행 (sam3 env):
    LD_LIBRARY_PATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python sam3_relabel_video.py \
        --frames data/replica_room0/images --img_ext .jpg \
        --colmap_dir data/replica_room0/sparse/0 \
        --depth_dir data/replica_room0/images --depth_scale 6553.5 \
        --vocab_json /home/elicer/sam3/vocab.json \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 2 --window 200 --min_area 0.0008 --min_track 2 \
        --vox 0.03 --sig_frac 0.25 --reid_th 0.3 \
        --exclude_concepts "door,blind,vent,window,wall,floor,ceiling,light switch,thermostat" \
        --out_root ~/relabel_video_room0
"""
import argparse, glob, json, os, gc, tempfile
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
    import struct
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
    """per-object 초기 포인트(depth voxel 센터)를 binary PLY로 저장 → prepare_folder가 points3d.ply로."""
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


def jac(a, b):
    if not a or not b: return 0.0
    i = len(a & b); return i / (len(a) + len(b) - i)


# ── ★ depth-dense 3D signature (v2) ★ ──
def load_depth(stem, dcfg, cache):
    """stem(frameNNNNNN) → 대응 depth 맵(meters). dcfg=(dir,pfrom,pto,ext,scale). 캐시."""
    if stem in cache: return cache[stem]
    ddir,pfrom,pto,ext,scale = dcfg
    dn = stem.replace(pfrom, pto) + ext
    path = os.path.join(ddir, dn)
    if not os.path.isfile(path):
        cache[stem]=None; return None
    D = np.asarray(Image.open(path)).astype(np.float32)
    if D.ndim==3: D=D[...,0]
    cache[stem]=D/scale          # meters
    return cache[stem]

def backproject_voxels(mask, cam, D, vox, max_px, zmin=0.05, zmax=20.0):
    """마스크 픽셀을 GT depth로 역투영 → world 좌표 → voxel-key(tuple) 집합."""
    if cam is None or D is None: return set()
    ys, xs = np.nonzero(mask)
    if len(xs)==0: return set()
    if len(xs)>max_px:
        sel=np.random.choice(len(xs),max_px,replace=False); xs,ys=xs[sel],ys[sel]
    Hd,Wd = D.shape
    sx=Wd/cam["W"]; sy=Hd/cam["H"]                 # depth 해상도가 RGB와 다를 수 있음
    xd=np.clip((xs*sx).astype(np.int64),0,Wd-1); yd=np.clip((ys*sy).astype(np.int64),0,Hd-1)
    d=D[yd,xd]
    ok=(d>zmin)&(d<zmax)
    if not ok.any(): return set()
    xs,ys,d=xs[ok].astype(np.float64),ys[ok].astype(np.float64),d[ok].astype(np.float64)
    Xc=np.stack([(xs-cam["cx"])/cam["fx"]*d, (ys-cam["cy"])/cam["fy"]*d, d],1)  # 카메라좌표
    Xw=(Xc-cam["t"])@cam["R"]                       # world = R^T (Xc - t)
    keys=np.floor(Xw/vox).astype(np.int64)
    return set(map(tuple, keys.tolist()))

def compute_sig(masks, cams, dcfg, dcache, vox, max_px, sig_frac):
    """track 의 모든 프레임 마스크 → depth voxel 집계 → multi-view 일관 voxel만 sig."""
    vcount=Counter()
    for stem,m in masks.items():
        D=load_depth(stem, dcfg, dcache)
        for k in backproject_voxels(m, cams.get(stem), D, vox, max_px):
            vcount[k]+=1
    nf=max(len(masks),1); thr=max(2,int(sig_frac*nf))
    return set(k for k,cnt in vcount.items() if cnt>=thr)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames",required=True); ap.add_argument("--img_ext",default=".jpg")
    ap.add_argument("--colmap_dir",required=True)
    ap.add_argument("--vocab_json",default=None); ap.add_argument("--vocab",default=None)
    ap.add_argument("--bpe",default=None)
    ap.add_argument("--stride",type=int,default=10,help="SAM3 propagate 프레임 subsample(메모리/속도)")
    ap.add_argument("--window",type=int,default=0,
                    help="★프레임을 이 개수 단위 window로 나눠 세션별 처리(0=전체 한 번). GPU 메모리 상한 고정.")
    ap.add_argument("--offload_state",action="store_true",default=True,
                    help="★프레임별 state를 CPU로 offload(GPU 메모리 프레임수 무관 평평). 기본 ON.")
    ap.add_argument("--no_offload_state",dest="offload_state",action="store_false")
    ap.add_argument("--offload_video",action="store_true",default=True,
                    help="★비디오 프레임 텐서를 CPU로 offload. 기본 ON.")
    ap.add_argument("--no_offload_video",dest="offload_video",action="store_false")
    ap.add_argument("--prompt_frame",type=int,default=0,help="concept를 프롬프트할 window-로컬 프레임 인덱스")
    ap.add_argument("--min_area",type=float,default=0.0008,
                    help="프레임 마스크 최소 면적(작은 객체 coverage 위해 v1 0.003 → 완화)")
    ap.add_argument("--min_track",type=int,default=2,help="유효 객체 최소 관측 프레임 수(완화)")
    # ── depth-dense sig 파라미터 ──
    ap.add_argument("--depth_dir",default=None,help="GT depth 폴더(기본: --frames 와 동일)")
    ap.add_argument("--depth_from",default="frame",help="stem 의 이 접두어를")
    ap.add_argument("--depth_to",default="depth",help="이걸로 치환해 depth 파일명 생성")
    ap.add_argument("--depth_ext",default=".png")
    ap.add_argument("--depth_scale",type=float,default=6553.5,help="uint16 → meters 나눗셈 인자")
    ap.add_argument("--vox",type=float,default=0.03,help="voxel 크기(m). 3cm 기본")
    ap.add_argument("--max_px",type=int,default=3000,help="프레임당 역투영 픽셀 상한(속도)")
    ap.add_argument("--min_sig",type=int,default=8,help="안정 voxel 이보다 적으면 노이즈 track 폐기")
    ap.add_argument("--sig_frac",type=float,default=0.25,
                    help="voxel 을 객체로 인정할 최소 프레임 비율(multi-view consistency)")
    ap.add_argument("--reid_th",type=float,default=0.3,help="시간 배타 track re-id 병합 voxel-Jaccard 임계")
    ap.add_argument("--iou_th",type=float,default=0.5,help="공존 프레임 2D 마스크 IoU 임계(이상=synonym/중복 병합)")
    ap.add_argument("--cand_th",type=float,default=0.05,help="voxel-Jaccard 후보 하한")
    ap.add_argument("--exclude_concepts",default="")
    ap.add_argument("--out_root",required=True)
    args=ap.parse_args(); os.makedirs(args.out_root,exist_ok=True)

    VOCAB=(json.load(open(args.vocab_json))["vocab"] if args.vocab_json
           else [v.strip() for v in args.vocab.split(",")])
    cams=load_cams(args.colmap_dir)
    ddir=args.depth_dir or args.frames
    dcfg=(ddir,args.depth_from,args.depth_to,args.depth_ext,args.depth_scale)
    dcache={}
    print(f"vocab={len(VOCAB)} cams={len(cams)} depth_dir={ddir} vox={args.vox}m")

    # 전역 프레임 리스트 (stride 적용) — window 로 나눠 세션별 처리
    src=sorted(glob.glob(os.path.join(args.frames,f"*{args.img_ext}")))[::args.stride]
    N=len(src)
    stems_all=[os.path.splitext(os.path.basename(f))[0] for f in src]
    win = args.window if args.window>0 else N
    win = max(1, min(win, N)) if N else 1
    windows=[range(s, min(s+win, N)) for s in range(0, N, win)] if N else []
    # per-track min_track: window로 쪼갤 땐 완화(1). 최종 필터는 병합 객체 단위(아래 line ~min_track).
    mt_track = 1 if args.window>0 else args.min_track
    print(f"frames={N}  window={win}  n_windows={len(windows)}  (single-prompt @local frame {args.prompt_frame}, streaming)")

    # depth 접근성 sanity check (첫 프레임)
    if N:
        _D=load_depth(stems_all[0],dcfg,dcache)
        print(f"depth probe [{stems_all[0]}]: "
              + (f"OK shape={_D.shape} range=[{_D[_D>0].min():.2f},{_D.max():.2f}]m" if _D is not None
                 else "★없음★ — --depth_dir/--depth_from/--depth_to 확인 필요"))

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

    # ── window별 세션 → concept별 single-prompt streaming → track 수집 ──
    tracks=[]
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        for wi,wr in enumerate(windows):
            # window 프레임을 정수명으로 심링크 (로컬 idx→전역 stem)
            wdir=tempfile.mkdtemp(prefix=f"sam3relabel_w{wi}_"); local2stem=[]
            for li,gi in enumerate(wr):
                os.symlink(os.path.abspath(src[gi]),os.path.join(wdir,f"{li}.jpg"))
                local2stem.append(stems_all[gi])
            Nw=len(local2stem); pf=int(np.clip(args.prompt_frame,0,max(Nw-1,0)))
            sid=predictor.handle_request(dict(type="start_session",resource_path=wdir,
                    offload_video_to_cpu=args.offload_video,
                    offload_state_to_cpu=args.offload_state))["session_id"]
            wtracks=0
            for c in VOCAB:
                predictor.handle_request(dict(type="reset_session",session_id=sid))
                predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=pf,text=c))
                opf=propagate(sid)
                byid={}
                for fidx,o in opf.items():
                    stem=local2stem[fidx]; ids=np.asarray(o["out_obj_ids"]).reshape(-1)
                    masks=np.asarray(o["out_binary_masks"]); probs=np.asarray(o["out_probs"]).reshape(-1)
                    for k,oid in enumerate(ids):
                        m=masks[k]
                        if m.mean()<args.min_area: continue
                        dd=byid.setdefault(int(oid),{"masks":{},"score":0.0})
                        dd["masks"][stem]=m>0; dd["score"]=max(dd["score"],float(probs[k]))
                kept=0
                for oid,dd in byid.items():
                    if len(dd["masks"])<mt_track: continue        # window: 완화(1)
                    # ★ depth-dense sig (배경 제거) ★
                    sig=compute_sig(dd["masks"],cams,dcfg,dcache,args.vox,args.max_px,args.sig_frac)
                    if len(sig)<args.min_sig: continue    # 안정 표면 voxel 부족 → 폐기
                    tracks.append(dict(concept=c,masks=dd["masks"],frames=set(dd["masks"].keys()),
                                       sig=sig,score=dd["score"]))
                    kept+=1; wtracks+=1
                print(f"  [{c}] SAM3 ids={len(byid)} → valid tracks={kept}"
                      + (f"  (window {wi+1}/{len(windows)})" if len(windows)>1 else ""))
            # window 세션 해제 → GPU 메모리 반환 (프레임 축 누적 차단)
            predictor.handle_request(dict(type="close_session",session_id=sid))
            gc.collect(); torch.cuda.empty_cache()
            print(f"  [window {wi+1}/{len(windows)}] frames {wr.start}..{wr.stop-1}  "
                  f"new tracks={wtracks}  total={len(tracks)}")
    try: predictor.shutdown()
    except Exception: pass
    print(f"\nnative tracks(전 concept·전 window): {len(tracks)}")

    # ── co-occurrence 기반 instance unification (union-find) ──
    #     window 경계로 쪼개진 동일 객체 = 프레임 배타 → voxel-Jaccard re-id 로 자동 재결합.
    parent=list(range(len(tracks)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y): parent[find(x)]=find(y)

    def mask_iou_shared(A,B,maxf=8):
        sh=sorted(A["frames"] & B["frames"])
        if not sh: return 0.0
        if len(sh)>maxf: sh=[sh[k] for k in np.linspace(0,len(sh)-1,maxf).astype(int)]
        v=[]
        for s in sh:
            a=A["masks"][s]; b=B["masks"][s]
            inter=int(np.logical_and(a,b).sum()); uni=int(np.logical_or(a,b).sum())
            v.append(inter/uni if uni else 0.0)
        return float(np.mean(v))

    n_syn=n_reid=0
    for i in range(len(tracks)):
        for j in range(i+1,len(tracks)):
            A,B=tracks[i],tracks[j]
            j3=jac(A["sig"],B["sig"])               # voxel-Jaccard (dense, 배경 없음)
            if j3<args.cand_th: continue
            if A["frames"] & B["frames"]:           # 공존: 마스크 IoU로 synonym vs 접촉 구분
                if mask_iou_shared(A,B)>args.iou_th:
                    union(i,j); n_syn+=1
            else:                                   # 시간 배타(다른 window 포함): 같은 위치=같은 객체(re-id)
                if j3>args.reid_th:
                    union(i,j); n_reid+=1
    groups=defaultdict(list)
    for i in range(len(tracks)): groups[find(i)].append(i)
    print(f"unification: synonym/dup 병합={n_syn}, re-id 병합={n_reid} → 그룹 {len(groups)}")

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
        if len(masks)<args.min_track: continue      # ★최종 필터: 병합된 객체의 전체 관측 프레임 수
        objs.append(dict(masks=masks,sig=sig,concepts=concepts))
    objs.sort(key=lambda o:-len(o["masks"]))
    print(f"구조물 제외+min_track 후 유효 객체: {len(objs)}")

    # ── 저장 (sig voxel-key → 센터 점) ──
    for gid,o in enumerate(objs):
        od=os.path.join(args.out_root,str(gid)); os.makedirs(od,exist_ok=True)
        for stem,m in o["masks"].items():
            Image.fromarray((m*255).astype(np.uint8)).save(os.path.join(od,f"{stem}.png"))
        if o["sig"]:
            pts=(np.array(sorted(o["sig"]),dtype=np.float64)+0.5)*args.vox
        else:
            pts=np.zeros((0,3),np.float64)
        write_ply(os.path.join(od,"points3d.ply"),pts.astype(np.float32))
        print(f"  obj{gid}: frames={len(o['masks'])} init_pts={len(pts)} "
              f"concept~{o['concepts'].most_common(1)[0][0]}")
    print(f"저장: {args.out_root}/<gid>/<stem>.png + points3d.ply")
    print("판정(v2.1): window 세션 분할로 프레임 축 메모리 상한 고정. "
          "3D voxel unification 이 window 경계 동일 객체를 re-id로 재결합.")


if __name__=="__main__":
    main()
