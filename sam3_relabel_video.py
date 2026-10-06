#!/usr/bin/env python3
"""Axis 1, steps 1-2 (video) -- instance re-labeling with the SAM 3 video predictor.

Design, as verified by probing: SAM 3's concept-video mode is a streaming detector.
  1) One prompt per concept at frame 0, then propagate. SAM 3 finds and tracks the
     instances itself (out_obj_ids). Re-prompting at several keyframes was removed: it
     was the source of duplicate instances.
  2) 3-D signature (sig) = dense back-projection through GT depth.
     - Mask pixels are back-projected with the depth map, so only points on the object's
       front surface enter the signature and wall / floor background never does.
     - A voxel hash plus a multi-view consistency test keeps only voxels seen in a
       consistent fraction of frames, which removes the noise.
  3) Instance unification works off signals intrinsic to the video, so it depends on
     thresholds as little as possible:
     - Same concept, two tracks that CO-OCCUR in a frame: mask IoU decides synonym /
       duplicate versus two objects in contact.
     - Temporally exclusive tracks whose 3-D footprints (voxels) overlap: re-identified
       and merged.
  4) Structural concepts are excluded and a min_track filter is applied.

v2.1: --window N splits the frames into windows processed as separate sessions, which
      caps GPU memory; --win_overlap protects the boundaries.
v2.2: pose coverage -- compute_sig's denominator counts only usable frames, camera
      coverage is reported, --posed_only (on by default), per-concept empty_cache.
v2.3: CPU RAM -- masks are bit-packed (8x smaller), compressed as propagate streams,
      the depth cache is released per window, RSS is logged.

v2.4 (checkpoint / resume)
    1. tracks are written to <out_root>/tracks_ckpt.pkl after each window. A re-run skips
       the completed windows and continues (the SAME --stride / --window / --win_overlap
       are required, and this is checked). The checkpoint is deleted on normal completion.
       --no_resume ignores it.
    2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments is detected and warned about at start.
    3. Existing numeric directories under out_root are removed before the final save, so
       leftovers from a crashed run cannot masquerade as "N objects succeeded" (the caller
       counts directories and does not check the exit code).

v2.5 (per-concept resume -- in-process recovery abandoned)
  Measured: after an NVML assert, empty_cache frees 0 bytes and 20.6 GB stays reserved, so
  rebuilding the predictor hits the same failure on its first .to(device). Once that assert
  fires, the process's allocator cannot be recovered.
    1. On RuntimeError, instead of retrying in-process, the tracks completed so far are
       written to the checkpoint (partial: window wi, concept ci) and the process exits 3.
    2. A re-run resumes from the concept that died, without restarting the window.
    3. To run unattended, wrap it:
       until bash run_full_pipeline.sh relabel; do echo "=== restart ==="; sleep 5; done

v2.6 (per-concept window splitting -- for deterministic per-concept OOM)
  Measured: in a fresh process running window 9 alone, [cushion] still hit the NVML assert
  at frame 182/200 every time. A concept with many instances tracks many objects at once,
  so the peak over a 200-frame window exceeds the GPU deterministically and a restart dies
  in the same place forever.
  Fix: when the same (window, concept) fails again, split_lv is raised in the checkpoint, so
  the re-run processes THAT concept in 2^lv pieces (2, then 4) as separate sessions, cutting
  the peak to 1/2 and 1/4. The pieces are temporally exclusive, so the existing 3-D re-id
  puts them back together. If four pieces still fail, the concept is recorded in
  skipped_concepts.txt and the run moves to the next one.

Output: <out_root>/<gid>/<stem>.png (masks) + points3d.ply (depth voxel centres, used as
init) -> input to prepare_folder, and <out_root>/concepts.json, the vocabulary term each
object came from.

Run (sam3 env):
    unset PYTORCH_CUDA_ALLOC_CONF
    LD_LIBRARY_PATH= \
    python sam3_relabel_video.py \
        --frames data/replica_room0/images --img_ext .jpg \
        --colmap_dir data/replica_room0/sparse_dense/0 \
        --depth_dir data/replica_room0/images --depth_scale 6553.5 \
        --vocab_json "$HOME"/sam3/vocab.json \
        --bpe "$HOME"/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        --stride 2 --window 200 --min_area 0.0008 --min_track 2 \
        --vox 0.03 --sig_frac 0.25 --reid_th 0.3 \
        --exclude_concepts "door,blind,vent,window,wall,floor,ceiling,light switch,thermostat" \
        --out_root ~/relabel_video_room0
"""
import argparse, glob, json, os, gc, pickle, shutil, sys, tempfile
from collections import Counter, defaultdict
import numpy as np, torch
from PIL import Image


# -- COLMAP --
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
    """Per-object init points (depth voxel centres) as a binary PLY -> prepare_folder reads
    it as points3d.ply."""
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


# -- v2.3: bit-packed masks (8x less CPU RAM) --
def pack_mask(m):
    """bool HxW -> (packed bytes, shape). np.packbits: 0.8 MB -> ~0.1 MB."""
    m = np.asarray(m, bool)
    return (np.packbits(m), m.shape)

def unpack_mask(p):
    b, shape = p
    return np.unpackbits(b, count=shape[0]*shape[1]).reshape(shape).astype(bool)

def or_masks(p1, p2):
    """packed OR packed -> packed (assumes equal shape; falls back to unpacking)."""
    if p1[1] == p2[1] and len(p1[0]) == len(p2[0]):
        return (np.bitwise_or(p1[0], p2[0]), p1[1])
    return pack_mask(unpack_mask(p1) | unpack_mask(p2))

def rss_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS"): return int(ln.split()[1]) / 1048576.0
    except Exception: pass
    return float("nan")


# -- dense 3-D signature from depth --
def load_depth(stem, dcfg, cache):
    """stem (frameNNNNNN) -> its depth map in metres. dcfg=(dir,pfrom,pto,ext,scale). Cached."""
    if stem in cache: return cache[stem]
    ddir,pfrom,pto,ext,scale = dcfg
    dn = stem.replace(pfrom, pto) + ext
    path = os.path.join(ddir, dn)
    if not os.path.isfile(path):
        cache[stem]=None; return None
    D = np.asarray(Image.open(path)).astype(np.float32)
    if D.ndim==3: D=D[...,0]
    cache[stem]=D/scale          # metres
    return cache[stem]

def backproject_voxels(mask, cam, D, vox, max_px, zmin=0.05, zmax=20.0):
    """Back-project the mask pixels through the depth map -> world coordinates -> a set of
    voxel keys (tuples)."""
    if cam is None or D is None: return set()
    ys, xs = np.nonzero(mask)
    if len(xs)==0: return set()
    if len(xs)>max_px:
        sel=np.random.choice(len(xs),max_px,replace=False); xs,ys=xs[sel],ys[sel]
    Hd,Wd = D.shape
    sx=Wd/cam["W"]; sy=Hd/cam["H"]                 # the depth map may differ in resolution from the RGB
    xd=np.clip((xs*sx).astype(np.int64),0,Wd-1); yd=np.clip((ys*sy).astype(np.int64),0,Hd-1)
    d=D[yd,xd]
    ok=(d>zmin)&(d<zmax)
    if not ok.any(): return set()
    xs,ys,d=xs[ok].astype(np.float64),ys[ok].astype(np.float64),d[ok].astype(np.float64)
    Xc=np.stack([(xs-cam["cx"])/cam["fx"]*d, (ys-cam["cy"])/cam["fy"]*d, d],1)  # camera frame
    Xw=(Xc-cam["t"])@cam["R"]                       # world = R^T (Xc - t)
    keys=np.floor(Xw/vox).astype(np.int64)
    return set(map(tuple, keys.tolist()))

def compute_sig(masks, cams, dcfg, dcache, vox, max_px, sig_frac):
    """Every frame mask of a track (packed) -> depth voxels -> only the multi-view consistent
    voxels become the signature.
    v2.2: the threshold's denominator counts only frames with both a camera and a depth map.
    v2.3: unpack on demand."""
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
    ap.add_argument("--stride",type=int,default=10,help="frame subsample for SAM3 propagate (memory / speed)")
    ap.add_argument("--window",type=int,default=0,
                    help="split the frames into windows of this size, each its own session "
                         "(0 = one pass over everything). Caps GPU memory.")
    ap.add_argument("--win_overlap",type=float,default=0.5,
                    help="window overlap fraction (0-0.9). Keeps an object that only appears "
                         "near a boundary wholly inside one window.")
    ap.add_argument("--posed_only",action="store_true",default=True,
                    help="v2.2: use only frames that have a colmap pose (on by default).")
    ap.add_argument("--no_posed_only",dest="posed_only",action="store_false")
    ap.add_argument("--offload_state",action="store_true",default=True,
                    help="offload per-frame state to the CPU. On by default.")
    ap.add_argument("--no_offload_state",dest="offload_state",action="store_false")
    ap.add_argument("--offload_video",action="store_true",default=True,
                    help="offload the video frame tensors to the CPU. On by default.")
    ap.add_argument("--no_offload_video",dest="offload_video",action="store_false")
    ap.add_argument("--resume",action="store_true",default=True,
                    help="v2.4: continue from the window checkpoint (on by default).")
    ap.add_argument("--no_resume",dest="resume",action="store_false")
    ap.add_argument("--prompt_frame",type=int,default=0,help="window-local frame index at which to prompt each concept")
    ap.add_argument("--min_area",type=float,default=0.0008,
                    help="minimum per-frame mask area")
    ap.add_argument("--min_track",type=int,default=2,help="minimum observed frames for a valid object")
    # -- dense depth signature parameters --
    ap.add_argument("--depth_dir",default=None,help="GT depth directory (default: same as --frames)")
    ap.add_argument("--depth_from",default="frame",help="replace this prefix in the stem")
    ap.add_argument("--depth_to",default="depth",help="with this one, to form the depth file name")
    ap.add_argument("--depth_ext",default=".png")
    ap.add_argument("--depth_scale",type=float,default=6553.5,help="divisor converting uint16 to metres")
    ap.add_argument("--vox",type=float,default=0.03,help="voxel size (m). 3 cm by default")
    ap.add_argument("--max_px",type=int,default=3000,help="cap on back-projected pixels per frame (speed)")
    ap.add_argument("--min_sig",type=int,default=8,help="discard a track as noise below this many stable voxels")
    ap.add_argument("--sig_frac",type=float,default=0.25,
                    help="minimum fraction of frames (of the usable ones) in which a voxel must "
                         "appear to count as part of the object")
    ap.add_argument("--reid_th",type=float,default=0.3,help="voxel-Jaccard threshold to merge temporally exclusive tracks")
    ap.add_argument("--iou_th",type=float,default=0.5,help="2-D mask IoU threshold on co-occurring frames (above = synonym / duplicate)")
    ap.add_argument("--cand_th",type=float,default=0.05,help="voxel-Jaccard floor for a merge candidate")
    ap.add_argument("--exclude_concepts",default="")
    ap.add_argument("--out_root",required=True)
    args=ap.parse_args(); os.makedirs(args.out_root,exist_ok=True)

    # -- v2.4: warn about the allocator option --
    acc=os.environ.get("PYTORCH_CUDA_ALLOC_CONF","")
    if "expandable_segments" in acc:
        print(f"** WARNING: PYTORCH_CUDA_ALLOC_CONF={acc}\n"
              "  expandable_segments is the suspected cause of the NVML_SUCCESS INTERNAL ASSERT\n"
              "  in CUDACachingAllocator. Running after `unset PYTORCH_CUDA_ALLOC_CONF` is strongly advised.")

    VOCAB=(json.load(open(args.vocab_json))["vocab"] if args.vocab_json
           else [v.strip() for v in args.vocab.split(",")])
    cams=load_cams(args.colmap_dir)
    ddir=args.depth_dir or args.frames
    dcfg=(ddir,args.depth_from,args.depth_to,args.depth_ext,args.depth_scale)
    dcache={}
    print(f"vocab={len(VOCAB)} cams={len(cams)} depth_dir={ddir} vox={args.vox}m")

    # Global frame list (after --stride), split into windows processed as separate sessions.
    src=sorted(glob.glob(os.path.join(args.frames,f"*{args.img_ext}")))[::args.stride]
    stems_all=[os.path.splitext(os.path.basename(f))[0] for f in src]

    # -- v2.2: camera-coverage diagnostic and the posed-only filter --
    n_posed=sum(1 for s in stems_all if s in cams)
    cover=n_posed/max(len(stems_all),1)
    print(f"*cam coverage: {n_posed}/{len(stems_all)} = {cover:.1%} (colmap={args.colmap_dir})")
    if cover<0.9:
        print("** WARNING: pose coverage below 90% -- colmap covers only a subset of the frames. "
              "A dense stride cannot raise the number of usable supervision views above the "
              "number of posed frames. Generate dense poses with make_dense_colmap.py.")
    if args.posed_only:
        keep=[i for i,s in enumerate(stems_all) if s in cams]
        if len(keep)<len(stems_all):
            print(f"*posed_only: {len(stems_all)} -> {len(keep)} frames (unposed dropped)")
        src=[src[i] for i in keep]; stems_all=[stems_all[i] for i in keep]
    N=len(src)

    win = args.window if args.window>0 else N
    win = max(1, min(win, N)) if N else 1
    if N and args.window>0:
        step=max(1,int(round(win*(1.0-max(0.0,min(0.9,args.win_overlap))))))   # overlap -> boundary objects captured whole
        windows=[]
        for s in range(0, N, step):
            w=range(s, min(s+win, N))
            if windows and w.stop<=windows[-1].stop: break                     # reached the end -- no duplicate window
            windows.append(w)
    else:
        windows=[range(0, N)] if N else []
    # per-track min_track: relaxed to 1 when splitting into windows. The final filter is applied
    # per merged object, below.
    mt_track = 1 if args.window>0 else args.min_track
    print(f"frames={N}  window={win}  n_windows={len(windows)}  (single-prompt @local frame {args.prompt_frame}, streaming)")

    # -- v2.4: load the checkpoint --
    ckpt_path=os.path.join(args.out_root,"tracks_ckpt.pkl")
    ckpt_key=dict(N=N,stride=args.stride,window=args.window,win_overlap=args.win_overlap,
                  n_windows=len(windows),vocab=len(VOCAB))
    tracks=[]; done_windows=0; partial_ci=0; split_lv=0   # partial_ci / split_lv: resume point and split level
    if args.resume and os.path.isfile(ckpt_path):
        try:
            with open(ckpt_path,"rb") as f: ck=pickle.load(f)
            if ck.get("key")==ckpt_key:
                tracks=ck["tracks"]; done_windows=ck["done_windows"]
                partial_ci=ck.get("partial_ci",0); split_lv=ck.get("split_lv",0)
                print(f"*resume: loaded checkpoint at window {done_windows}/{len(windows)} "
                      f"+ partial concept {partial_ci}"
                      + (f" (split_lv={split_lv})" if split_lv else "")
                      + f" (tracks={len(tracks)})")
            else:
                print(f"*ckpt ignored: parameters differ {ck.get('key')} != {ckpt_key}")
        except Exception as e:
            print(f"*ckpt failed to load ({e}) -- running from the start")

    def save_ckpt(dw, pci=0, slv=0):
        with open(ckpt_path+".tmp","wb") as f:
            pickle.dump(dict(key=ckpt_key,done_windows=dw,partial_ci=pci,split_lv=slv,tracks=tracks),
                        f,protocol=4)
        os.replace(ckpt_path+".tmp",ckpt_path)

    # Depth sanity check on the first frame. Without depth every signature comes back empty and
    # every track is discarded below min_sig -- the run then reports zero objects and exits 0,
    # which reads as a successful no-op. Stop here instead.
    if N:
        _D=load_depth(stems_all[0],dcfg,dcache)
        if _D is None:
            print(f"[abort] depth probe [{stems_all[0]}] found nothing at "
                  f"{os.path.join(ddir, stems_all[0].replace(args.depth_from, args.depth_to) + args.depth_ext)}")
            print("        The 3-D signature is a back-projection through depth, so without it every")
            print("        track is discarded and this run would produce zero objects without failing.")
            print("        Check --depth_dir / --depth_from / --depth_to / --depth_ext.")
            sys.exit(1)
        print(f"depth probe [{stems_all[0]}]: OK shape={_D.shape} "
              f"range=[{_D[_D>0].min():.2f},{_D.max():.2f}]m")

    from sam3.model_builder import build_sam3_video_predictor
    def build_predictor():
        try:
            return build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()),bpe_path=args.bpe) \
                   if args.bpe else build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
        except TypeError:
            return build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    predictor=build_predictor()

    # -- per-window session -> one prompt per concept, streaming -> collect tracks --
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        for wi,wr in enumerate(windows):
            if wi<done_windows: continue                     # v2.4: resume skip
            # Symlink the window's frames under integer names (local idx -> global stem).
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
            wtracks=0
            ci=partial_ci if wi==done_windows else 0    # v2.5: resume from the concept that died
            slv=split_lv if wi==done_windows else 0     # v2.6: that concept's split level
            partial_ci=0; split_lv=0
            if ci>0: print(f"  *partial resume: window {wi+1} from concept {ci}({VOCAB[ci]})"
                           + (f" split_lv={slv}" if slv else ""))
            while ci<len(VOCAB):
                c=VOCAB[ci]
                cur_slv,slv = slv,0                     # the split level applies only to the resumed concept
                try:
                    # v2.3: do not accumulate the propagate stream -- filter and bit-pack at once
                    byid={}
                    def collect(stream, off):
                        for r in stream:
                            o=r["outputs"]; stem=local2stem[off+r["frame_index"]]
                            ids=np.asarray(o["out_obj_ids"]).reshape(-1)
                            masks=np.asarray(o["out_binary_masks"]); probs=np.asarray(o["out_probs"]).reshape(-1)
                            for k,oid in enumerate(ids):
                                m=masks[k]
                                if m.mean()<args.min_area: continue
                                dd=byid.setdefault((off,int(oid)),{"masks":{},"score":0.0})
                                dd["masks"][stem]=pack_mask(m>0); dd["score"]=max(dd["score"],float(probs[k]))
                    if cur_slv>0:
                        # v2.6: this concept hit OOM over the full window -> process it in 2^lv pieces.
                        #       The pieces are temporally exclusive, so the 3-D re-id reunites them.
                        nsub=2**cur_slv
                        print(f"  *[{c}] split mode: {Nw} frames -> {nsub} pieces")
                        try: predictor.handle_request(dict(type="close_session",session_id=sid))
                        except Exception: pass
                        gc.collect(); torch.cuda.empty_cache()
                        bounds=np.linspace(0,Nw,nsub+1).astype(int)
                        for si in range(nsub):
                            lo,hi=int(bounds[si]),int(bounds[si+1])
                            if hi<=lo: continue
                            sdir=tempfile.mkdtemp(prefix=f"sam3relabel_w{wi}s{si}_")
                            for lj,gj in enumerate(range(lo,hi)):
                                os.symlink(os.path.abspath(src[wr[gj]]),os.path.join(sdir,f"{lj}.jpg"))
                            ssid=predictor.handle_request(dict(type="start_session",resource_path=sdir,
                                    offload_video_to_cpu=args.offload_video,
                                    offload_state_to_cpu=args.offload_state))["session_id"]
                            predictor.handle_request(dict(type="add_prompt",session_id=ssid,frame_index=0,text=c))
                            collect(predictor.handle_stream_request(
                                dict(type="propagate_in_video",session_id=ssid)), lo)
                            predictor.handle_request(dict(type="close_session",session_id=ssid))
                            gc.collect(); torch.cuda.empty_cache()
                        sid=open_session()              # restore the main session for the next concept
                    else:
                        predictor.handle_request(dict(type="reset_session",session_id=sid))
                        predictor.handle_request(dict(type="add_prompt",session_id=sid,frame_index=pf,text=c))
                        collect(predictor.handle_stream_request(
                            dict(type="propagate_in_video",session_id=sid)), 0)
                except RuntimeError as e:
                    # v2.5 / v2.6: after an NVML assert this process cannot recover -> save the
                    #   checkpoint and exit. Repeated failure on the same concept raises the split
                    #   level (2 then 4 pieces); if that still fails, skip it.
                    print(f"  *RuntimeError @[{c}] window {wi+1} (split_lv={cur_slv}): {str(e).splitlines()[0]}")
                    if cur_slv>=2:
                        with open(os.path.join(args.out_root,"skipped_concepts.txt"),"a") as f:
                            f.write(f"window{wi} {c}\n")
                        print(f"  *[{c}] failed even in 4 pieces -> recorded in skipped_concepts.txt, "
                              f"resuming from the next concept")
                        save_ckpt(wi, ci+1, 0)
                    else:
                        save_ckpt(wi, ci, cur_slv+1)
                        print(f"  *ckpt saved (window {wi}, concept {ci}, split_lv={cur_slv+1}) -> exit 3. "
                              f"Re-running resumes [{c}] in split mode.")
                    print("    to run unattended: until bash run_full_pipeline.sh relabel; do sleep 5; done")
                    sys.exit(3)
                kept=0
                for oid,dd in byid.items():
                    if len(dd["masks"])<mt_track: continue        # windowed: relaxed to 1
                    sig=compute_sig(dd["masks"],cams,dcfg,dcache,args.vox,args.max_px,args.sig_frac)
                    if len(sig)<args.min_sig: continue    # too few stable surface voxels -> discard
                    tracks.append(dict(concept=c,masks=dd["masks"],frames=set(dd["masks"].keys()),
                                       sig=sig,score=dd["score"]))
                    kept+=1; wtracks+=1
                print(f"  [{c}] SAM3 ids={len(byid)} -> valid tracks={kept}"
                      + (f"  (window {wi+1}/{len(windows)})" if len(windows)>1 else ""))
                del byid
                torch.cuda.empty_cache()                  # v2.2: relieve the accumulation along the concept axis
                ci+=1
            # Close the window's session -> GPU memory returned. v2.3: manage the depth cache and RSS.
            predictor.handle_request(dict(type="close_session",session_id=sid))
            dcache.clear(); gc.collect(); torch.cuda.empty_cache()
            # v2.4 / v2.5: save the window checkpoint (atomic rename)
            save_ckpt(wi+1, 0)
            print(f"  [window {wi+1}/{len(windows)}] frames {wr.start}..{wr.stop-1}  "
                  f"new tracks={wtracks}  total={len(tracks)}  RSS={rss_gb():.1f}GB  ckpt ok")
    try: predictor.shutdown()
    except Exception: pass
    print(f"\nnative tracks (all concepts, all windows): {len(tracks)}  RSS={rss_gb():.1f}GB")

    # -- instance unification by co-occurrence (union-find) --
    #     With overlapping windows, two tracks of the same object in adjacent windows share
    #     frames -> the mask-IoU (synonym) path. Non-overlapping boundaries and re-appearances
    #     are frame-exclusive -> the voxel-Jaccard re-id path.
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
            j3=jac(A["sig"],B["sig"])               # voxel-Jaccard (dense, no background)
            if j3<args.cand_th: continue
            if A["frames"] & B["frames"]:           # co-occurring: mask IoU separates synonym from contact
                if mask_iou_shared(A,B)>args.iou_th:
                    union(i,j); n_syn+=1
            else:                                   # temporally exclusive (incl. across windows): same place = same object
                if j3>args.reid_th:
                    union(i,j); n_reid+=1
    groups=defaultdict(list)
    for i in range(len(tracks)): groups[find(i)].append(i)
    print(f"unification: synonym/dup merges={n_syn}, re-id merges={n_reid} -> {len(groups)} groups")

    # -- group -> object (masks OR'd while staying packed, sig unioned, concept by majority) --
    excl={c.strip() for c in args.exclude_concepts.split(",") if c.strip()}
    objs=[]
    for members in groups.values():
        masks={}; sig=set(); concepts=Counter()
        for mi in members:
            t=tracks[mi]; sig|=t["sig"]; concepts[t["concept"]]+=1
            for stem,mp_ in t["masks"].items():
                masks[stem]=or_masks(masks[stem],mp_) if stem in masks else mp_
        if concepts.most_common(1)[0][0] in excl: continue
        if len(masks)<args.min_track: continue      # final filter: total observed frames of the merged object
        objs.append(dict(masks=masks,sig=sig,concepts=concepts))
    objs.sort(key=lambda o:-len(o["masks"]))
    print(f"valid objects after structural exclusion + min_track: {len(objs)}")

    # -- v2.4: remove leftovers from a previous run (numeric directories only), so a crashed
    #    run's remains cannot pass for success. concepts.json goes with them: keeping an old
    #    one beside new gids would silently caption the wrong objects.
    stale=[d for d in glob.glob(os.path.join(args.out_root,"[0-9]*")) if os.path.isdir(d)]
    if stale:
        print(f"*removing {len(stale)} existing object directories before saving")
        for d in stale: shutil.rmtree(d)
    _cj=os.path.join(args.out_root,"concepts.json")
    if os.path.isfile(_cj): os.remove(_cj)

    # -- save (sig voxel keys -> centre points) --
    concept_rows={}
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
        top,votes=o["concepts"].most_common(1)[0]
        concept_rows[str(gid)]=dict(concept=top,votes=int(votes),
                                    votes_total=int(sum(o["concepts"].values())),
                                    frames=len(o["masks"]),
                                    all={k:int(v) for k,v in o["concepts"].items()})
        print(f"  obj{gid}: frames={len(o['masks'])} init_pts={len(pts)} concept~{top}")

    # The vocabulary term each object came from. Until this was written down it existed only in
    # this script's stdout, so the only object names available downstream came from the GT
    # semantic mesh -- which made the ShapeR captions depend on a Replica-only file, not just
    # the evaluation. gid here is the same gid every later stage uses.
    with open(_cj,"w") as f:
        json.dump(concept_rows,f,indent=1,ensure_ascii=False)

    # Normal completion -> delete the checkpoint
    if os.path.isfile(ckpt_path): os.remove(ckpt_path)
    print(f"saved: {args.out_root}/<gid>/<stem>.png + points3d.ply")
    print(f"concepts: {_cj} ({len(concept_rows)} objects)")
    print("v2.4 verdict: window checkpoints + automatic NVML/OOM recovery -- a crash preserves "
          "the completed windows.")


if __name__=="__main__":
    main()