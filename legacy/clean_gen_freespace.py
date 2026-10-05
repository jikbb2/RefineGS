#!/usr/bin/env python3
"""gen-centric 정리: Amodal3R(registered) 메시를 기준으로, 관측에 모순되는
free-space 부유물만 제거하고 출력. recon 과 합치지 않음(단일 표면).

- 관측 표면과 일치하는 gen 면 → 유지
- 관측 표면 '앞'(free-space)으로 튀어나온 면(부유물) → 제거
- 미관측(뒤/밑) 확장 → recon 광선이 없으니 테스트 불가 → 유지

루트(fuse_carve.py, eval_object_mesh.py 와 같은 폴더)에서 실행:
  python clean_gen_freespace.py \
    --gen   ~/Amodal3R/poc_output/0/seed_1/mesh_registered_clean.ply \
    --recon output/replica_room0_v2/iso0_depth/train/ours_7000/fuse_post.ply \
    --colmap_dir data/replica_room0_v2/masks/0/sparse/0 \
    --masks_dir  data/replica_room0_v2/masks/0/sil_amodal \
    --out output/replica_room0_v2/iso0_depth/train/ours_7000/gen_cleaned.ply \
    --margin 0.02
"""
import argparse
import numpy as np
import open3d as o3d
from eval_object_mesh import load_instance_cameras, load_instance_masks
from fuse_carve import freespace_violating_vertices, _build_raycast_scene, load_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--colmap_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.02)
    a = ap.parse_args()

    gen = load_mesh(a.gen)
    recon = load_mesh(a.recon)
    gv = np.asarray(gen.vertices)
    gf = np.asarray(gen.triangles)
    if len(gf) == 0:
        print("[ERROR] empty gen mesh"); return

    cameras = load_instance_cameras(str(a.colmap_dir))
    masks = load_instance_masks(str(a.masks_dir), cameras)
    scene = _build_raycast_scene(recon)

    viol = freespace_violating_vertices(gv, scene, cameras, masks, a.margin)
    keep_face = ~viol[gf].any(axis=1)

    sf = gf[keep_face]
    uv, ni = np.unique(sf, return_inverse=True)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(gv[uv])
    out.triangles = o3d.utility.Vector3iVector(ni.reshape(sf.shape))
    out.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, out)

    print(f"gen faces {len(gf)} -> kept {int(keep_face.sum())} "
          f"(removed {int((~keep_face).sum())} free-space 부유물)")
    bb = (np.asarray(out.vertices).max(0) - np.asarray(out.vertices).min(0)).round(2)
    print(f"verts {len(out.vertices)}  extent {bb}  -> {a.out}")


if __name__ == "__main__":
    main()
