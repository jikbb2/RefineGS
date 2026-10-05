#!/usr/bin/env python3
"""객체별 SDF 메쉬(world 좌표) + base 를 whole-scene 메쉬로 병합.

각 per-object SDF post 메쉬는 이미 world 좌표(sdf_distill_depth 가 verts*scale+center 복원)라
정합 불필요 — concat + base proximity carve + 후처리만.

  python merge_object_meshes.py \
    --obj_glob "output/replica_room0_v2/refinegs_full/*/train/ours_*/sdf_obj_post.ply" \
    --base output/replica_room0_v2/scene_mono_reg/train/ours_30000/fuse_post.ply \
    --carve_dist 0.04 \
    --out output/replica_room0_v2/whole_sdf_merged.ply

Deps: numpy, open3d, scipy.
"""
import os
import glob
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_glob", required=True)
    ap.add_argument("--base", default="", help="벽·바닥 base 메쉬(선택). 객체 근처는 carve")
    ap.add_argument("--carve_dist", type=float, default=0.04,
                    help="base 정점이 객체 표면 이 거리(m) 이내면 제거(중복 표면 방지)")
    ap.add_argument("--exclude", nargs="*", default=["merged_bak"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(args.obj_glob)) if not any(x in p for x in args.exclude)]
    assert paths, f"객체 메쉬 없음: {args.obj_glob}"
    print(f"객체 메쉬 {len(paths)}개")

    merged = o3d.geometry.TriangleMesh()
    obj_pts = []
    for p in paths:
        m = o3d.io.read_triangle_mesh(p)
        if len(m.triangles) == 0:
            print(f"  [skip 빈] {p}"); continue
        merged += m
        obj_pts.append(np.asarray(m.vertices))
        print(f"  + {os.path.basename(os.path.dirname(os.path.dirname(p)))}: "
              f"verts {len(m.vertices)}")
    obj_all = np.concatenate(obj_pts)

    if args.base and os.path.exists(args.base):
        base = o3d.io.read_triangle_mesh(args.base)
        bv = np.asarray(base.vertices)
        d, _ = cKDTree(obj_all).query(bv, workers=-1)
        keep = d > args.carve_dist            # 객체에서 먼 base 정점만(벽·바닥 몸통)
        base.remove_vertices_by_mask(~keep)
        base.remove_unreferenced_vertices()
        base.remove_degenerate_triangles()
        print(f"base carve: {(~keep).sum()}/{len(bv)} 제거 → {len(base.vertices)} 유지")
        merged += base

    merged.remove_duplicated_vertices()
    merged.remove_degenerate_triangles()
    merged.remove_unreferenced_vertices()
    merged.compute_vertex_normals()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(out, merged)
    print(f"\nwhole-scene 병합 메쉬: verts {len(merged.vertices)} faces {len(merged.triangles)} → {out}")


if __name__ == "__main__":
    main()
