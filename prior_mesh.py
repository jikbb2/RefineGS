#!/usr/bin/env python3
"""Extract a world-space mesh from a prior field (npz) -- a tool for looking at what
was generated.

Why this exists:
  Looking at the fusion output alone cannot tell you whether an unfilled region is the
  fusion's fault or the generator's. Measured on obj10 (the pot): the gate blocked the
  prior at unknown 6.4%, and forcing the gate open still did not fill the empty half
  -> the prior itself never generated that part. The gate measures "share of the prior
  surface that lies in unobserved space", so it reported correctly; the problem was in
  the generation stage.

  Extract the prior mesh with this script, open it next to fuse_post.ply, and the two
  cases separate immediately:
    - the prior is a complete object  -> fusion / gate problem
    - the prior is also half missing  -> generation problem (conditioning points,
      bounds, CFG)

Usage:
  python prior_mesh.py ~/prior/obj10_field.npz --out /tmp/obj10_prior.ply
  python prior_mesh.py ~/prior/obj10_field.npz --out /tmp/p.ply --level 0
"""
import argparse
import os

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=float, default=0.0, help="isolevel (m). 0 = surface")
    args = ap.parse_args()

    z = np.load(os.path.expanduser(args.npz))
    F = z["field"].astype(np.float32)
    G = F.shape[0]
    center, R, scale = z["center"], z["R_align"], float(z["scale"])
    vox = float(z["vox_world"])

    neg = float((F < args.level).mean())
    print(f"[prior] {os.path.basename(args.npz)}  G={G}  voxel={vox*1000:.2f}mm")
    print(f"  field range [{F.min():.4f}, {F.max():.4f}]m   "
          f"inside level {args.level}m: {neg*100:.2f}%")
    if not (F.min() < args.level < F.max()):
        raise SystemExit(f"isolevel {args.level} is outside the field range -- adjust --level")

    # Forward transform on the fusion side (sdf_distill_depth.py `_sd`):
    #   n = (q - center) @ R.T * scale
    # R is orthonormal, so the inverse is  q = (n / scale) @ R + center.
    # scale MULTIPLIES on the way in, so divide on the way out.
    #
    # gradient_direction: skimage's default is "descent", which assumes the object has
    # HIGHER values than the exterior. An SDF is the opposite -- negative inside -- so the
    # default orients every face inward and the mesh renders as a hollow black shell under
    # backface culling. Point-sampled metrics do not care, but this script exists to be
    # looked at, and an inside-out mesh reads as "the prior is missing", which is exactly
    # the wrong conclusion.
    v, f, _, _ = marching_cubes(F, level=args.level, spacing=(2.0 / (G - 1),) * 3,
                                gradient_direction="ascent")
    v = (v - 1.0) / scale                      # [-1,1] -> aligned coordinates (metres)
    v = v @ R + center                         # -> world

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    m.compute_vertex_normals()
    p = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    assert o3d.io.write_triangle_mesh(p, m), f"write failed: {p}"

    V = np.asarray(m.vertices)
    ext = V.max(0) - V.min(0)
    print(f"  vertices {len(V):,}  faces {len(np.asarray(m.triangles)):,}")
    print(f"  world bbox  min {np.round(V.min(0), 3)}  max {np.round(V.max(0), 3)}")
    print(f"              size {np.round(ext, 3)} m")
    # Self-check on the transform: the span the grid covers must be vox_world x G.
    # This checks SIZE only -- a permuted axis mapping passes it, so compare the size
    # above against the object's measured extent (audit_objects.py) as well.
    span = vox * (G - 1)
    if ext.max() > span * 1.05:
        print(f"  WARNING bbox ({ext.max():.3f}m) is larger than the grid span "
              f"({span:.3f}m) -- suspect the coordinate transform (scale multiply/divide)")
    else:
        print(f"  (grid span {span:.3f}m -- a bbox inside it means the transform is sane)")

    # Per-axis occupancy: see WHICH side is empty as a number, not just by eye.
    lv = " ▁▂▃▄▅▆▇█"
    for ax, nm in enumerate("xyz"):
        h, _ = np.histogram(V[:, ax], bins=16)
        bar = "".join("·" if c == 0 else lv[max(1, min(8, int(8 * c / h.max())))] for c in h)
        print(f"  {nm} axis |{bar}|  (· = no vertices)")
    print(f"\n-> {p}\n  Open it in MeshLab next to fuse_post.ply.")
    print("  A complete prior means a fusion problem; a half prior means a generation problem.")


if __name__ == "__main__":
    main()