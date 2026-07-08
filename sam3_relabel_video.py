#!/usr/bin/env python3
"""
축1 ❶❷ (video) — SAM3 video predictor 기반 instance re-labeling (재설계 v2: depth-dense sig).

설계 원리 (probe로 검증: SAM3 concept-video는 streaming detection):
  1) concept당 single-prompt(frame 0) + propagate → SAM3가 비디오 전체에서 인스턴스를
     자동 검출·추적(out_obj_ids). multi-keyframe 재프롬프트 제거(중복 생성 원인 제거).
  2) 3D signature(sig) = ★GT-depth dense back-projection★:
     - 마스크 픽셀을 GT depth로 역투영 → 객체 *앞면 실제 표면점*만 → 배경(벽/바닥) 원천 제거.
     - voxel 해시 + multi-view consistency(여러 프레임에서 일관되게 찍힌 voxel만) → 노이즈 제거.
  3) instance unification = '비디오 고유 신호' 기반(임계 의존 최소):
     - 같은 concept: 두 track이 *같은 프레임에 공존(co-occur)*하면 mask IoU로 synonym/distinct 판별.
       *시간적으로 배타적*이고 3D footprint(voxel)가 겹치면 → re-identification 병합.
  4) 구조물 concept 제외 + min_track 필터.

★ v2.1: --window N — 프레임을 window로 잘라 세션별 처리(GPU 상한 고정), --win_overlap 로 경계 보호.
★ v2.2: pose-coverage — compute_sig 분모=유효 프레임만, cam coverage 진단, --posed_only(기본 ON),
        concept별 empty_cache.
★ v2.3: CPU RAM — 마스크 bit-pack(8×↓), propagate 스트리밍 즉시 압축, dcache window별 해제, RSS 로그.

★ v2.4 패치 (checkpoint/resume + NVML assert 자동 복구) ★
  window 9/9 의 94% 지점 NVML assert 로 수 시간 런이 전멸하는 문제 대응:
    1. window 완료마다 tracks 를 <out_root>/tracks_ckpt.pkl 에 저장. 재실행 시 자동으로
       완료된 window 를 건너뛰고 이어감(★같은 --stride/--window/--win_overlap 필수★, 자동 검증).
       정상 완료 시 ckpt 자동 삭제. --no_resume 으로 무시 가능.
    2. concept propagate 중 RuntimeError(NVML assert/OOM) 발생 시: 세션 폐기 → predictor 재빌드
       → 같은 window 세션 재시작 → 해당 concept 부터 재시도(같은 concept 최대 2회).
    3. 시작 시 PYTORCH_CUDA_ALLOC_CONF=expandable_segments 감지 경고 — 이 allocator 옵션이
       NVML_SUCCESS INTERNAL ASSERT 의 유발 혐의. unset 권장.
    4. 최종 저장 전 out_root 의 기존 숫자 폴더 제거 — 크래시 후 이전 런 잔재가 "N objects 성공"
       으로 위장하는 사고 방지 (run_full_pipeline.sh 는 exit code 를 확인하지 않고 폴더 수만 센다).

출력: <out_root>/<gid>/<stem>.png (마스크) + points3d.ply (depth voxel 센터, init) → prepare_folder 입력.

실행 (sam3 env):
    unset PYTORCH_CUDA_ALLOC_CONF
    LD_LIBRARY_PATH= \
    python sam3_relabel_video.py \
        --frames data/replica_room0_v2/images --img_ext .jpg \
        --colmap_dir data/replica_room0_v2/sparse_dense/0 \
        --depth_dir data/replica_room0_v2/images --depth_scale 6553.5 \
        --vocab_json /home/elicer/sam3/vocab.json \
        --bpe /home/elicer/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 2 --window 200 --min_area 0.0008 --min_track 2 \
        --vox 0.03 --sig_frac 0.25 --reid_th 0.3 \
        --exclude_concepts "door,blind,vent,window,wall,floor,ceiling,light switch,thermostat" \
        --out_root ~/relabel_video_room0
"""
import argparse, glob, json, os, gc, pickle, shutil, tempfile
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
    imgs=[]
    for ln in open(os.path.join(d,"images.txt")):
        if ln.startswith("#") or not ln.strip(): continue
        t=ln.split()
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


# ── ★ v2.3: bit-packed 마스크 (CPU RAM 8×↓) ★ ──
def pack_mask(m):
    """bool HxW → (packed bytes, shape). np.packbits: 0.8MB → ~0.1MB."""
    m = np.asarray(m, bool)
    return (np.packbits(m), m.shape)

def unpack_mask(p):
    b, shape = p
    return np.unpackbits(b, count=shape[0]*shape[1]).reshape(shape).astype(bool)

def or_masks(p1, p2):
    """packed OR packed → packed (같은 shape 가정; 다르면 unpack 경로)."""
    if p1[1] == p2[1] and len(p1[0]) == len(p2[0]):
        return (np.bitwise_or(p1[0], p2[0]), p1[1])
    return pack_mask(unpack_mask(p1) | unpack_mask(p2))

def rss_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS"): return int(ln.split()[1]) / 1048576.0
    except Exception: pass
    return float("nan")


# ── ★ depth-dense 3D signature ★ ──
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
    """track 의 모든 프레임 마스크(packed) → depth voxel 집계 → multi-view 일관 voxel만 sig.
    ★v2.2: 임계 분모 = cam+depth 유효 프레임 수만. ★v2.3: on-demand unpack."""
    vcount=Counter(); nvalid=0
    for stem,mp_ in masks.items():
        cam=cams.get(stem); D=load_depth(stem, dcfg, dcache)
        if cam is None or D is None: continue
        nvalid+=1
        for k in backproject_voxels(unpack_mask(mp_), cam, D, vox, max_px):
            vcount[k]+=1
    thr=max(2,int(sig_frac*max(nvalid,1)))
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
    ap.add_argument("--win_overlap",type=float,default=0.5,
                    help="★window 겹침 비율(0~0.9). 경계/짧은 관측 객체를 한 window에 온전히 담아 누락 방지.")
    ap.add_argument("--posed_only",action="store_true",default=True,
                    help="★v2.2: colmap pose 있는 프레임만 사용(기본 ON).")
    ap.add_argument("--no_posed_only",dest="posed_only",action="store_false")
    ap.add_argument("--offload_state",action="store_true",default=True,
                    help="★프레임별 state를 CPU로 offload. 기본 ON.")
    ap.add_argument("--no_offload_state",dest="offload_state",action="store_false")
    ap.add_argument("--offload_video",action="store_true",default=True,
                    help="★비디오 프레임 텐서를 CPU로 offload. 기본 ON.")
    ap.add_argument("--no_offload_video",dest="offload_video",action="store_false")
    ap.add_argument("--resume",action="store_true",default=True,
                    help="★v2.4: window 체크포인트에서 이어가기(기본 ON).")
    ap.add_argument("--no_resume",dest="resume",action="store_false")
    ap.add_argument("--prompt_frame",type=int,default=0,help="concept를 프롬프트할 window-로컬 프레임 인덱스")
    ap.add_argument("--min_area",type=float,default=0.0008,
                    help="프레임 마스크 최소 면적")
    ap.add_argument("--min_track",type=int,default=2,help="유효 객체 최소 관측 프레임 수")
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
                    help="voxel 을 객체로 인정할 최소 프레임 비율(유효 프레임 기준)")
    ap.add_argument("--reid_th",type=float,default=0.3,help="시간 배타 track re-id 병합 voxel-Jaccard 임계")
    ap.add_argument("--iou_th",type=float,default=0.5,help="공존 프레임 2D 마스크 IoU 임계(이상=synonym/중복 병합)")
    ap.add_argument("--cand_th",type=float,default=0.05,help="voxel-Jaccard 후보 하한")
    ap.add_argument("--exclude_concepts",default="")
    ap.add_argument("--out_root",required=True)
    args=ap.parse_args(); os.makedirs(args.out_root,exist_ok=True)

    # ── ★v2.4: allocator 옵션 경고 ──
    acc=os.environ.get("PYTORCH_CUDA_ALLOC_CONF","")
    if "expandable_segments" in acc:
        print(f"★★경고: PYTORCH_CUDA_ALLOC_CONF={acc}\n"
              "  expandable_segments 는 NVML_SUCCESS INTERNAL ASSERT(CUDACachingAllocator) 유발 혐의.\n"
              "  `unset PYTORCH_CUDA_ALLOC_CONF` 후 실행을 강력 권장.")

    VOCAB=(json.load(open(args.vocab_json))["vocab"] if args.vocab_json
           else [v.strip() for v in args.vocab.split(",")])
    cams=load_cams(args.colmap_dir)
    ddir=args.depth_dir or args.frames
    dcfg=(ddir,args.depth_from,args.depth_to,args.depth_ext,args.depth_scale)
    dcache={}
    print(f"vocab={len(VOCAB)} cams={len(cams)} depth_dir={ddir} vox={args.vox}m")

    # 전역 프레임 리스트 (stride 적용) — window 로 나눠 세션별 처리
    src=sorted(glob.glob(os.path.join(args.frames,f"*{args.img_ext}")))[::args.stride]
    stems_all=[os.path.splitext(os.path.basename(f))[0] for f in src]

    # ── ★v2.2: cam coverage 진단 + posed-only 필터 ──
    n_posed=sum(1 for s in stems_all if s in cams)
    cover=n_posed/max(len(stems_all),1)
    print(f"★cam coverage: {n_posed}/{len(stems_all)} = {cover:.1%} (colmap={args.colmap_dir})")
    if cover<0.9:
        print("★★경고: pose coverage <90% — colmap 이 프레임 서브셋만 커버. dense stride 를 줘도 "
              "유효 감독 뷰는 posed 프레임 수를 넘지 못함. make_dense_colmap.py 로 dense pose 생성 권장.")
    if args.posed_only:
        keep=[i for i,s in enumerate(stems_all) if s in cams]
        if len(keep)<len(stems_all):
            print(f"★posed_only: {len(stems_all)} → {len(keep)} 프레임 (unposed 제외)")
        src=[src[i] for i in keep]; stems_all=[stems_all[i] for i in keep]
    N=len(src)

    win = args.window if args.window>0 else N
    win = max(1, min(win, N)) if N else 1
    if N and args.window>0:
        step=max(1,int(round(win*(1.0-max(0.0,min(0.9,args.win_overlap))))))   # overlap → 경계 객체 온전 포착
        windows=[]
        for s in range(0, N, step):
            w=range(s, min(s+win, N))
            if windows and w.stop<=windows[-1].stop: break                     # 끝 도달 → 중복 window 방지
            windows.append(w)
    else:
        windows=[range(0, N)] if N else []
    # per-track min_track: window로 쪼갤 땐 완화(1). 최종 필터는 병합 객체 단위(아래).
    mt_track = 1 if args.window>0 else args.min_track
    print(f"frames={N}  window={win}  n_windows={len(windows)}  (single-prompt @local frame {args.prompt_frame}, streaming)")

    # ── ★v2.4: 체크포인트 로드 ──
    ckpt_path=os.path.join(args.out_root,"tracks_ckpt.pkl")
    ckpt_key=dict(N=N,stride=args.stride,window=args.window,win_overlap=args.win_overlap,
                  n_windows=len(windows),vocab=len(VOCAB))
    tracks=[]; done_windows=0
    if args.resume and os.path.isfile(ckpt_path):
        try:
            with open(ckpt_path,"rb") as f: ck=pickle.load(f)
            if ck.get("key")==ckpt_key:
                tracks=ck["tracks"]; done_windows=ck["done_windows"]
                print(f"★resume: window {done_windows}/{len(windows)} 완료 체크포인트 로드 (tracks={len(tracks)})")
            else:
                print(f"★ckpt 무시: 파라미터 불일치 {ck.get('key')} != {ckpt_key}")
        except Exception as e:
            print(f"★ckpt 로드 실패({e}) — 처음부터 실행")

    # depth 접근성 sanity check (첫 프레임)
    if N:
        _D=load_depth(stems_all[0],dcfg,dcache)
        print(f"depth probe [{stems_all[0]}]: "
              + (f"OK shape={_D.shape} range=[{_D[_D>0].min():.2f},{_D.max():.2f}]m" if _D is not None
                 else "★없음★ — --depth_dir/--depth_from/--depth_to 확인 필요"))

    from sam3.model_builder import build_sam3_video_predictor
    def build_predictor():
        try:
            return build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()),bpe_path=args.bpe) \
                   if args.bpe else build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
        except TypeError:
            return build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    predictor=build_predictor()

    # ── window별 세션 → concept별 single-prompt streaming → track 수집 ──
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        for wi,wr in enumerate(windows):
            if wi<done_windows: continue                     # ★v2.4: resume skip
            # window 프레임을 정수명으로 심링크 (로컬 idx→전역 stem)
            wdir=tempfile.mkdtemp(prefix=f"sam3relabel_w{wi}_"); local2stem=[]
            for li,gi in enumerate(wr):
                os.symlink(os.path.abspath(src[gi]),os.path.join(wdir,f"{li}.jpg"))
                local2stem.append(stems_all[gi])
            Nw=len(local2stem); pf=int(np.clip(args.prompt_frame,0,max(Nw-1,0)))
            def open_session():
                return predictor.handle_request(dict(type="start_session",resource_path=wdir,
                        offload_video_to_cpu=args.offload_video,
                        offload_state_to_cpu=args.offload_state))["session_id"]
            sid=open_session()
            wtracks=0; ci=0; retry=Counter()
            while ci<len(VOCAB):
                c=VOCAB[ci]
                try:
                    predictor.handle_request(dict(type="reset_session",session_id=sid))
                    predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=pf,text=c))
                    # ★v2.3: propagate 스트림을 모아두지 않고 즉시 필터 + bit-pack
                    byid={}
                    for r in predictor.handle_stream_request(dict(type="propagate_in_video",session_id=sid)):
                        o=r["outputs"]; stem=local2stem[r["frame_index"]]
                        ids=np.asarray(o["out_obj_ids"]).reshape(-1)
                        masks=np.asarray(o["out_binary_masks"]); probs=np.asarray(o["out_probs"]).reshape(-1)
                        for k,oid in enumerate(ids):
                            m=masks[k]
                            if m.mean()<args.min_area: continue
                            dd=byid.setdefault(int(oid),{"masks":{},"score":0.0})
                            dd["masks"][stem]=pack_mask(m>0); dd["score"]=max(dd["score"],float(probs[k]))
                except RuntimeError as e:
                    # ★v2.4: NVML assert / OOM → predictor 통째 재빌드 후 같은 concept 재시도
                    retry[ci]+=1
                    print(f"  ★RuntimeError @[{c}] window {wi+1}: {str(e).splitlines()[0]}")
                    if retry[ci]>2:
                        raise
                    print(f"  ★predictor 재빌드 후 [{c}] 재시도 ({retry[ci]}/2)")
                    try: predictor.handle_request(dict(type="close_session",session_id=sid))
                    except Exception: pass
                    try: predictor.shutdown()
                    except Exception: pass
                    del predictor; gc.collect(); torch.cuda.empty_cache()
                    predictor=build_predictor()
                    sid=open_session()
                    continue
                kept=0
                for oid,dd in byid.items():
                    if len(dd["masks"])<mt_track: continue        # window: 완화(1)
                    sig=compute_sig(dd["masks"],cams,dcfg,dcache,args.vox,args.max_px,args.sig_frac)
                    if len(sig)<args.min_sig: continue    # 안정 표면 voxel 부족 → 폐기
                    tracks.append(dict(concept=c,masks=dd["masks"],frames=set(dd["masks"].keys()),
                                       sig=sig,score=dd["score"]))
                    kept+=1; wtracks+=1
                print(f"  [{c}] SAM3 ids={len(byid)} → valid tracks={kept}"
                      + (f"  (window {wi+1}/{len(windows)})" if len(windows)>1 else ""))
                del byid
                torch.cuda.empty_cache()                  # ★v2.2: concept 축 누적 완화
                ci+=1
            # window 세션 해제 → GPU 메모리 반환 + ★v2.3: depth 캐시/RSS 관리
            predictor.handle_request(dict(type="close_session",session_id=sid))
            dcache.clear(); gc.collect(); torch.cuda.empty_cache()
            # ★v2.4: window 체크포인트 저장 (원자적 rename)
            with open(ckpt_path+".tmp","wb") as f:
                pickle.dump(dict(key=ckpt_key,done_windows=wi+1,tracks=tracks),f,protocol=4)
            os.replace(ckpt_path+".tmp",ckpt_path)
            print(f"  [window {wi+1}/{len(windows)}] frames {wr.start}..{wr.stop-1}  "
                  f"new tracks={wtracks}  total={len(tracks)}  RSS={rss_gb():.1f}GB  ckpt✓")
    try: predictor.shutdown()
    except Exception: pass
    print(f"\nnative tracks(전 concept·전 window): {len(tracks)}  RSS={rss_gb():.1f}GB")

    # ── co-occurrence 기반 instance unification (union-find) ──
    #     overlap window: 같은 객체의 인접-window track 은 프레임 공유 → mask-IoU(synonym) 경로.
    #     non-overlap 경계/재등장: 프레임 배타 → voxel-Jaccard re-id 경로.
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
            a=unpack_mask(A["masks"][s]); b=unpack_mask(B["masks"][s])
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

    # ── 그룹 → 객체 (masks OR — packed 상태 유지, sig union, concept 다수결) ──
    excl={c.strip() for c in args.exclude_concepts.split(",") if c.strip()}
    objs=[]
    for members in groups.values():
        masks={}; sig=set(); concepts=Counter()
        for mi in members:
            t=tracks[mi]; sig|=t["sig"]; concepts[t["concept"]]+=1
            for stem,mp_ in t["masks"].items():
                masks[stem]=or_masks(masks[stem],mp_) if stem in masks else mp_
        if concepts.most_common(1)[0][0] in excl: continue
        if len(masks)<args.min_track: continue      # ★최종 필터: 병합된 객체의 전체 관측 프레임 수
        objs.append(dict(masks=masks,sig=sig,concepts=concepts))
    objs.sort(key=lambda o:-len(o["masks"]))
    print(f"구조물 제외+min_track 후 유효 객체: {len(objs)}")

    # ── ★v2.4: 이전 런 잔재 제거 (숫자 폴더만) — 크래시 잔재의 '성공 위장' 방지 ──
    stale=[d for d in glob.glob(os.path.join(args.out_root,"[0-9]*")) if os.path.isdir(d)]
    if stale:
        print(f"★기존 객체 폴더 {len(stale)}개 제거 후 새로 저장")
        for d in stale: shutil.rmtree(d)

    # ── 저장 (sig voxel-key → 센터 점) ──
    for gid,o in enumerate(objs):
        od=os.path.join(args.out_root,str(gid)); os.makedirs(od,exist_ok=True)
        for stem,mp_ in o["masks"].items():
            m=unpack_mask(mp_)
            Image.fromarray((m*255).astype(np.uint8)).save(os.path.join(od,f"{stem}.png"))
        if o["sig"]:
            pts=(np.array(sorted(o["sig"]),dtype=np.float64)+0.5)*args.vox
        else:
            pts=np.zeros((0,3),np.float64)
        write_ply(os.path.join(od,"points3d.ply"),pts.astype(np.float32))
        print(f"  obj{gid}: frames={len(o['masks'])} init_pts={len(pts)} "
              f"concept~{o['concepts'].most_common(1)[0][0]}")
    # 정상 완료 → 체크포인트 삭제
    if os.path.isfile(ckpt_path): os.remove(ckpt_path)
    print(f"저장: {args.out_root}/<gid>/<stem>.png + points3d.ply")
    print("판정(v2.4): window 체크포인트 + NVML/OOM 자동 복구 — 크래시가 나도 완료 window 는 보존.")


if __name__=="__main__":
    main()
