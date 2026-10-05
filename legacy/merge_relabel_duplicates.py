#!/usr/bin/env python3
"""relabel 출력의 중복 객체를 point-Jaccard로 후처리 병합 (SAM3 재실행 불필요).

각 객체: <in_root>/<gid>/*.png (per-frame 마스크) + *.ply (init 3D 점).
객체쌍의 3D 점-집합 Jaccard > --jac 이면 같은 물리 객체로 보고 union-find 병합.
병합 그룹: 점 union(중복 제거) + 프레임별 마스크 OR.
출력: <out_root>/<newgid>/{*.png, points3d.ply}

실행:
  python merge_relabel_duplicates.py --in_root ~/relabel_replica_room0_v2 \
      --out_root ~/relabel_replica_room0_v2_merged --jac 0.5
"""
import argparse, glob, os, shutil
from collections import defaultdict
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def load_pts(f):
    v = PlyData.read(f)["vertex"].data
    return np.c_[v["x"], v["y"], v["z"]].astype(np.float32)


def write_ply(path, P):
    el = np.empty(len(P), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el["x"], el["y"], el["z"] = P[:, 0], P[:, 1], P[:, 2]
    PlyData([PlyElement.describe(el, "vertex")], text=False).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--jac", type=float, default=0.5, help="이 이상이면 같은 객체로 병합")
    ap.add_argument("--prec", type=int, default=2, help="점 격자 반올림 자리수(2=1cm)")
    a = ap.parse_args()

    gids = sorted([d for d in os.listdir(a.in_root)
                   if os.path.isdir(os.path.join(a.in_root, d))], key=lambda x: int(x))
    pts, keys = {}, {}
    for g in gids:
        plys = glob.glob(os.path.join(a.in_root, g, "*.ply"))
        P = load_pts(plys[0]) if plys else np.zeros((0, 3), np.float32)
        pts[g] = P
        keys[g] = set(map(tuple, np.round(P, a.prec)))

    # union-find by point-Jaccard
    parent = {g: g for g in gids}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y): parent[find(x)] = find(y)

    npair = 0
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            a_, b_ = keys[gids[i]], keys[gids[j]]
            u = len(a_ | b_)
            jac = len(a_ & b_) / u if u else 0.0
            if jac > a.jac:
                union(gids[i], gids[j]); npair += 1
                print(f"  merge obj{gids[i]} ~ obj{gids[j]}  pointJac={jac:.2f}")

    groups = defaultdict(list)
    for g in gids:
        groups[find(g)].append(g)

    if os.path.exists(a.out_root):
        shutil.rmtree(a.out_root)
    os.makedirs(a.out_root)
    for newid, members in enumerate(sorted(groups.values(), key=lambda m: -len(keys[m[0]]))):
        outd = os.path.join(a.out_root, str(newid))
        os.makedirs(outd)
        # 점 union (중복 제거)
        allP = np.vstack([pts[m] for m in members]) if members else np.zeros((0, 3), np.float32)
        if len(allP):
            allP = np.unique(np.round(allP, a.prec), axis=0)
        write_ply(os.path.join(outd, "points3d.ply"), allP.astype(np.float32))
        # 마스크 union (프레임별 OR)
        bystem = {}
        for m in members:
            for png in glob.glob(os.path.join(a.in_root, m, "*.png")):
                stem = os.path.splitext(os.path.basename(png))[0]
                arr = np.array(Image.open(png).convert("L")) > 127
                bystem[stem] = bystem[stem] | arr if stem in bystem else arr
        for stem, arr in bystem.items():
            Image.fromarray((arr * 255).astype(np.uint8)).save(os.path.join(outd, stem + ".png"))
        print(f"obj{newid}: merged {len(members)} ({','.join(members)})  "
              f"pts={len(allP)} frames={len(bystem)}")

    print(f"\n{len(gids)} -> {len(groups)} objects  ({npair} pairs merged)  -> {a.out_root}")


if __name__ == "__main__":
    main()
