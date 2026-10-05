#!/usr/bin/env python3
"""
relabel 출력(<relabel>/<gid>/<stem>.png + points3d.ply)에서 중복 객체 쌍을 탐지.

원리 — 중복은 두 형태로 나타난다:
  (1) 병합 실패(synonym miss): 같은 객체가 두 gid 로 분리, *같은 프레임*에 둘 다 마스크 존재
      → 공유 프레임 마스크 IoU 가 높음. 서로 다른 인스턴스라면 픽셀이 겹칠 수 없어 IoU≈0
      (인접 접촉 시 경계에서 소량 겹침 가능, ~0.1 이하).
  (2) re-id 실패: 같은 객체가 프레임 배타적으로 두 gid 로 분리 → 마스크 비교 불가.
      대신 points3d.ply(3D voxel footprint)의 Jaccard/centroid 거리로 탐지 —
      같은 위치에 두 객체가 앉아 있으면 중복 혐의.

판정 기준(기본값):
  - 공유 프레임 mean IoU > 0.30            → DUP(synonym miss) 혐의
  - 공유 프레임 없음 & 3D Jaccard > 0.15   → DUP(re-id miss) 혐의
  - 공유 프레임 있음 & IoU 낮음 & 3D Jaccard > 0.30 → PARTIAL(부분 마스크 분리) 혐의

출력:
  - 콘솔: 혐의 쌍 테이블 (IoU / 3D-Jac / centroid 거리)
  - <out>/report.csv: 전체 쌍 수치
  - <out>/pair_<A>_<B>_<stem>.png: 혐의 쌍 오버레이 (A=빨강, B=초록, 겹침=노랑) — 눈 검증용

실행 (RefineGS 루트, 아무 env):
    python check_duplicate_objects.py \
        --relabel ~/relabel_replica_room0_v2 \
        --frames data/replica_room0_v2/images \
        --out ~/dup_check
"""
import argparse, glob, os, struct
import numpy as np
from PIL import Image


def load_ply_xyz(path):
    """sam3_relabel_video.py 가 쓴 binary PLY (x,y,z float + rgb uchar) 읽기."""
    with open(path, "rb") as f:
        n = 0
        while True:
            ln = f.readline().decode(errors="ignore")
            if ln.startswith("element vertex"): n = int(ln.split()[-1])
            if ln.strip() == "end_header": break
        dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")])
        a = np.frombuffer(f.read(n*dt.itemsize), dt, count=n)
    return np.stack([a["x"],a["y"],a["z"]],1) if n else np.zeros((0,3),np.float32)


def vox_set(xyz, vox):
    if len(xyz)==0: return set()
    return set(map(tuple, np.floor(xyz/vox).astype(np.int64).tolist()))


def jac(a,b):
    if not a or not b: return 0.0
    i=len(a&b); return i/(len(a)+len(b)-i)


def load_mask(p):
    return np.asarray(Image.open(p))>127


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--relabel",required=True,help="relabel 출력 루트 (<gid>/ 폴더들)")
    ap.add_argument("--frames",default=None,help="RGB 프레임 폴더 (오버레이용; 없으면 마스크만 오버레이)")
    ap.add_argument("--img_ext",default=".jpg")
    ap.add_argument("--out",default="./dup_check")
    ap.add_argument("--vox",type=float,default=0.03)
    ap.add_argument("--iou_dup",type=float,default=0.30,help="공유 프레임 mean IoU 이 값 초과 → DUP")
    ap.add_argument("--jac_reid",type=float,default=0.15,help="프레임 배타 & 3D Jaccard 초과 → DUP")
    ap.add_argument("--jac_part",type=float,default=0.30,help="공유+저IoU & 3D Jaccard 초과 → PARTIAL")
    ap.add_argument("--max_shared",type=int,default=6,help="쌍당 IoU 계산 공유 프레임 샘플 수")
    ap.add_argument("--max_overlay",type=int,default=30)
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)

    gids=sorted([d for d in os.listdir(args.relabel)
                 if d.isdigit() and os.path.isdir(os.path.join(args.relabel,d))],key=int)
    objs={}
    for g in gids:
        od=os.path.join(args.relabel,g)
        stems={os.path.splitext(os.path.basename(p))[0]:p
               for p in glob.glob(os.path.join(od,"*.png"))}
        xyz=load_ply_xyz(os.path.join(od,"points3d.ply")) if os.path.isfile(os.path.join(od,"points3d.ply")) else np.zeros((0,3))
        objs[g]=dict(stems=stems,vox=vox_set(xyz,args.vox),
                     cen=xyz.mean(0) if len(xyz) else np.full(3,np.nan))
        print(f"obj{g}: frames={len(stems)} vox={len(objs[g]['vox'])}")

    rows=[]; flagged=[]
    G=list(objs.keys())
    for i in range(len(G)):
        for j in range(i+1,len(G)):
            A,B=objs[G[i]],objs[G[j]]
            shared=sorted(set(A["stems"])&set(B["stems"]))
            j3=jac(A["vox"],B["vox"])
            cd=float(np.linalg.norm(A["cen"]-B["cen"])) if not (np.isnan(A["cen"]).any() or np.isnan(B["cen"]).any()) else float("nan")
            miou=float("nan"); best=(None,-1.0)
            if shared:
                sel=[shared[k] for k in np.linspace(0,len(shared)-1,min(args.max_shared,len(shared))).astype(int)]
                v=[]
                for s in sel:
                    a=load_mask(A["stems"][s]); b=load_mask(B["stems"][s])
                    inter=int((a&b).sum()); uni=int((a|b).sum())
                    iou=inter/uni if uni else 0.0; v.append(iou)
                    if iou>best[1]: best=(s,iou)
                miou=float(np.mean(v))
            verdict=""
            if shared and miou>args.iou_dup: verdict="DUP(synonym miss)"
            elif not shared and j3>args.jac_reid: verdict="DUP(re-id miss)"
            elif shared and j3>args.jac_part and miou<=args.iou_dup: verdict="PARTIAL?"
            rows.append((G[i],G[j],len(shared),miou,j3,cd,verdict))
            if verdict:
                flagged.append((G[i],G[j],len(shared),miou,j3,cd,verdict,best[0]))

    with open(os.path.join(args.out,"report.csv"),"w") as f:
        f.write("objA,objB,shared_frames,mean_iou,jac3d,centroid_dist_m,verdict\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]}\n")

    print(f"\n총 {len(rows)}쌍 검사 → 혐의 {len(flagged)}쌍")
    if flagged:
        print(f"{'A':>4}{'B':>5}{'shared':>8}{'mIoU':>8}{'3dJac':>8}{'cdist':>8}  verdict")
        for a,b,ns,miou,j3,cd,vd,_ in sorted(flagged,key=lambda r:-(r[3] if r[3]==r[3] else r[4])):
            print(f"{a:>4}{b:>5}{ns:>8}{miou:>8.3f}{j3:>8.3f}{cd:>8.3f}  {vd}")
    else:
        print("중복 혐의 없음 — 인스턴스 분리가 깨끗함.")

    # ── 오버레이 (눈 검증) ──
    n_ov=0
    for a,b,ns,miou,j3,cd,vd,stem in flagged[:args.max_overlay]:
        A,B=objs[a],objs[b]
        if stem is None:                                  # 프레임 배타 쌍: 각자 최대 마스크 프레임
            stem=max(A["stems"],key=lambda s:load_mask(A["stems"][s]).sum())
        ma=load_mask(A["stems"][stem]) if stem in A["stems"] else None
        mb=load_mask(B["stems"][stem]) if stem in B["stems"] else None
        H,W=(ma if ma is not None else mb).shape
        if args.frames and os.path.isfile(os.path.join(args.frames,stem+args.img_ext)):
            img=np.asarray(Image.open(os.path.join(args.frames,stem+args.img_ext)).resize((W,H))).copy()
        else:
            img=np.zeros((H,W,3),np.uint8)
        img=(img*0.45).astype(np.uint8)
        if ma is not None: img[ma]=(img[ma]*0.3+np.array([255,40,40])*0.7).astype(np.uint8)
        if mb is not None: img[mb]=(img[mb]*0.3+np.array([40,255,40])*0.7).astype(np.uint8)
        if ma is not None and mb is not None:
            ov=ma&mb; img[ov]=(255,255,0)
        p=os.path.join(args.out,f"pair_{a}_{b}_{stem}_{vd.split('(')[0]}.png")
        Image.fromarray(img).save(p); n_ov+=1
    if n_ov: print(f"오버레이 {n_ov}장 저장 → {args.out}/pair_*.png (빨강=A, 초록=B, 노랑=겹침)")


if __name__=="__main__":
    main()
