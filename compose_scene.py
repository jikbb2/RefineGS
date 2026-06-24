#!/usr/bin/env python3
"""Composition: 홀리스틱 씬 베이스 + per-object 정제본 합성.

각 객체의 bbox 안에 있는 베이스 메시 면을 제거(이중표면/z-fight 방지)하고,
정제된 객체 메시를 삽입한다. 바닥·벽 등 객체 밖 영역은 베이스가 그대로 제공.

per-object 메시는 scene 좌표여야 함:
  - fuse_post.ply (depth recon, 올바른 위치)  ← 기본
  - fuse_genclean.ply (Amodal3R 완성)  ← 등록 견고화 후 권장

실행:
  python compose_scene.py \
    --base output/replica_room0/scene_base/train/ours_*/fuse_post.ply \
    --root output/replica_room0_v2/refinegs_full \
    --mesh_name fuse_post \
    --out output/replica_room0_v2/scene_composed.ply --pad 0.03
"""
import argparse, glob, os, re
import numpy as np
import trimesh


def best_obj(gid_dir, name):
    its = glob.glob(os.path.join(gid_dir, "train", "ours_*"))
    if not its:
        return None
    it = max(its, key=lambda p: int(re.search(r"ours_(\d+)", p).group(1)))
    cands = ([name + ".ply"] if name else []) + \
            ["fuse_genclean.ply", "fuse_completed.ply", "fuse_post.ply"]
    for n in cands:
        f = os.path.join(it, n)
        if os.path.exists(f):
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="홀리스틱 씬 메시 (glob 허용)")
    ap.add_argument("--root", required=True, help="refinegs_full")
    ap.add_argument("--mesh_name", default=None, help="fuse_post / fuse_genclean ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.03, help="bbox 여유(m)")
    ap.add_argument("--min_verts", type=int, default=200, help="이보다 작은 객체는 무시")
    ap.add_argument("--labels", nargs="*", default=None, help="특정 객체만(없으면 전체)")
    a = ap.parse_args()

    base_path = sorted(glob.glob(a.base))[-1] if glob.glob(a.base) else a.base
    base = trimesh.load(base_path, process=False)
    print(f"base: {base_path}  verts {len(base.vertices)} faces {len(base.faces)}")

    objs, boxes = [], []
    for d in sorted(glob.glob(os.path.join(a.root, "*"))):
        gid = os.path.basename(d)
        if a.labels and gid not in a.labels:
            continue
        f = best_obj(d, a.mesh_name)
        if not f:
            continue
        m = trimesh.load(f, process=False)
        if len(m.vertices) < a.min_verts:
            print(f"  skip {gid} (verts<{a.min_verts})"); continue
        objs.append(m)
        boxes.append((m.vertices.min(0) - a.pad, m.vertices.max(0) + a.pad))
        print(f"  obj {gid}: verts {len(m.vertices)}  ({os.path.basename(f)})")

    # 베이스에서 객체 bbox 안의 면 제거
    if boxes:
        fc = base.triangles.mean(axis=1)
        remove = np.zeros(len(fc), bool)
        for lo, hi in boxes:
            remove |= np.all((fc >= lo) & (fc <= hi), axis=1)
        base.update_faces(~remove)
        base.remove_unreferenced_vertices()
        print(f"base carved: removed {int(remove.sum())} faces -> kept {len(base.faces)}")

    scene = trimesh.util.concatenate([base] + objs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    scene.export(a.out)
    print(f"composed: base + {len(objs)} objects -> {a.out}  (verts {len(scene.vertices)})")


if __name__ == "__main__":
    main()
