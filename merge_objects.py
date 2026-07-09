#!/usr/bin/env python3
"""
relabel 객체 폴더 병합 — 중복 판정된 gid 들을 하나로 합친다 (마스크 OR + ply 합집합).

사용:
    python merge_objects.py --relabel ~/relabel_replica_room0_v2 --merge "3,13,30"
    # 여러 그룹: --merge "3,13,30;5,22"

동작:
  - 각 그룹의 첫 gid 가 대상. 나머지 gid 의 마스크를 프레임별 OR 로 흡수, ply 포인트 합집합(voxel dedup).
  - 흡수된 gid 폴더는 <gid>.merged_bak 으로 rename (검증 후 수동 삭제).
  - prepare_folder/recon 은 폴더 glob 기반이라 gid 번호 구멍은 문제 없음.
"""
import argparse, glob, os, struct
import numpy as np
from PIL import Image


def load_ply_xyz(path):
    with open(path,"rb") as f:
        n=0
        while True:
            ln=f.readline().decode(errors="ignore")
            if ln.startswith("element vertex"): n=int(ln.split()[-1])
            if ln.strip()=="end_header": break
        dt=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")])
        a=np.frombuffer(f.read(n*dt.itemsize),dt,count=n)
    return np.stack([a["x"],a["y"],a["z"]],1) if n else np.zeros((0,3),np.float32)


def write_ply(path,xyz):
    xyz=np.asarray(xyz,np.float32); n=len(xyz)
    with open(path,"wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        dt=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("r","u1"),("g","u1"),("b","u1")])
        a=np.empty(n,dt)
        if n: a["x"],a["y"],a["z"]=xyz[:,0],xyz[:,1],xyz[:,2]; a["r"]=a["g"]=a["b"]=180
        f.write(a.tobytes())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--relabel",required=True)
    ap.add_argument("--merge",required=True,help='병합 그룹. 예: "3,13,30;5,22"')
    ap.add_argument("--vox",type=float,default=0.03,help="ply 합집합 dedup voxel")
    args=ap.parse_args()

    for grp in args.merge.split(";"):
        gids=[g.strip() for g in grp.split(",") if g.strip()]
        assert len(gids)>=2, f"그룹에 gid 2개 이상 필요: {grp}"
        tgt=os.path.join(args.relabel,gids[0])
        assert os.path.isdir(tgt), f"{tgt} 없음"
        print(f"── merge {gids[1:]} → obj{gids[0]}")
        # 마스크 OR
        for g in gids[1:]:
            src=os.path.join(args.relabel,g)
            assert os.path.isdir(src), f"{src} 없음"
            for p in glob.glob(os.path.join(src,"*.png")):
                stem=os.path.basename(p); q=os.path.join(tgt,stem)
                m=np.asarray(Image.open(p))>127
                if os.path.isfile(q):
                    m=m|(np.asarray(Image.open(q))>127)
                Image.fromarray((m*255).astype(np.uint8)).save(q)
        # ply 합집합 (voxel dedup)
        pts=[load_ply_xyz(os.path.join(args.relabel,g,"points3d.ply"))
             for g in gids if os.path.isfile(os.path.join(args.relabel,g,"points3d.ply"))]
        if pts:
            P=np.concatenate(pts,0)
            keys=np.floor(P/args.vox).astype(np.int64)
            _,idx=np.unique(keys,axis=0,return_index=True)
            write_ply(os.path.join(tgt,"points3d.ply"),P[np.sort(idx)])
        # 흡수된 폴더 백업 rename
        for g in gids[1:]:
            os.rename(os.path.join(args.relabel,g),os.path.join(args.relabel,g+".merged_bak"))
        n=len(glob.glob(os.path.join(tgt,"*.png")))
        print(f"   obj{gids[0]}: frames={n}, ply pts={len(P) if pts else 0} (dedup 후 {len(idx) if pts else 0})")
    print("완료. 검증 후 *.merged_bak 폴더는 수동 삭제.")


if __name__=="__main__":
    main()
