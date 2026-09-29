#!/usr/bin/env python3
"""Render side A and side B of the same object from identical cameras, for the paper figure.

Matching viewpoints by hand is not reproducible, and a reader cannot tell whether a
difference between two panels came from the method or from the camera. Here both meshes are
rendered through the same intrinsics and the same extrinsics, so every pixel difference is
the method.

Rendering is done by ray casting on the CPU (open3d's RaycastingScene), not through OpenGL:
a headless server has no display, and an offscreen GL context is one more thing that fails
for a reason unrelated to the figure. Shading is Lambertian on the hit normals, which is
enough to read a shape and keeps the two panels comparable.

The BACK view is the point of the figure. Every camera in the trajectory saw the front, so
that is where A and B agree; what the fusion added is on the side no camera reached.

  python render_ab.py \\
      --recon  output/replica_room2_v2/objects_voted/6/train/ours_30000/fuse_post.ply \\
      --recon2 output/replica_room2_v2/objects_voted/6/train/ours_30000/fused_0927_1517_post.ply \\
      --colmap data/replica_room2_v2/sparse/0 \\
      --out /tmp/fig_room2_obj6.png --tag "room2 obj6 (chair)"

Without --colmap the front direction is the world +X axis; with it, the front is the
trajectory camera that sees side A best, so the "front" panel matches what was actually
observed.
"""

import argparse
import os
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("[abort] open3d is required: pip install open3d")


# ----------------------------------------------------------------- COLMAP text poses
def read_colmap_images(path):
    """[(name, R_wc, t_wc)] from images.txt. Binary models are not read here."""
    f = os.path.join(os.path.expanduser(path), "images.txt")
    if not os.path.isfile(f):
        return []
    out, expect_pose = [], True
    for line in open(f):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if expect_pose:
            t = line.split()
            if len(t) >= 10:
                qw, qx, qy, qz = (float(x) for x in t[1:5])
                tx, ty, tz = (float(x) for x in t[5:8])
                # COLMAP stores world->camera as a quaternion.
                R = np.array([
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ])
                out.append((t[9], R, np.array([tx, ty, tz])))
            expect_pose = False
        else:
            expect_pose = True  # the points2D line
    return out


# ----------------------------------------------------------------- geometry helpers
def load_mesh(path):
    m = o3d.io.read_triangle_mesh(os.path.expanduser(path))
    if len(m.triangles) == 0:
        sys.exit(f"[abort] no triangles in {path}")
    m.compute_vertex_normals()
    return m


def load_gt(path, ids):
    """The GT mesh, restricted to the given object_id values.

    Read with plyfile, never with open3d. Replica's mesh_semantic.ply carries a per-face
    `object_id` property, and open3d's RPly parser rejects the header outright ("Invalid file
    format") -- which is also why every other script in this pipeline uses plyfile for it.
    Reading it here and handing open3d a plain vertex/triangle mesh sidesteps the parser
    entirely, and selecting the faces first means the renderer never sees the whole room.
    """
    from plyfile import PlyData
    p = PlyData.read(os.path.expanduser(path))
    fe = p["face"]
    key = "vertex_indices" if "vertex_indices" in fe.data.dtype.names else "vertex_index"
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)

    keep = {int(x) for x in ids.replace(",", " ").split()} if ids else None
    if keep is None:
        print("[warn] --gt without --gt_ids draws the whole scene mesh; pass the GT id "
              "from names.tsv")
    T = []
    for face, oid in zip(fe[key], fe["object_id"]):
        if keep is not None and int(oid) not in keep:
            continue
        for k in range(1, len(face) - 1):
            T.append((face[0], face[k], face[k + 1]))
    if not T:
        have = sorted({int(o) for o in fe["object_id"]})[:20]
        sys.exit(f"[abort] no GT faces with object_id in {sorted(keep)}. "
                 f"ids present (first 20): {have}")

    # Drop the vertices no kept face uses, so the mesh's bounds are the object's bounds.
    T = np.asarray(T, np.int64)
    used = np.unique(T)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    gt = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V[used]),
        o3d.utility.Vector3iVector(remap[T].astype(np.int32)))
    gt.compute_vertex_normals()
    print(f"[gt] {len(T)} triangles for object_id {sorted(keep) if keep else 'ALL'}")
    return gt


def gt_ids_near(path, mesh, tol=0.05, min_frac=0.05, top=10):
    """Which GT instances this reconstruction actually covers, by nearest-point vote.

    A figure needs the GT instance the object corresponds to, and nothing in the output tree
    records it: the voted gid is an index into OUR labels, not into Replica's object_id. The
    numbers were read off by hand for the two objects that got figures, which does not scale
    and cannot be checked. Here every vertex of `mesh` votes for the GT instance whose
    surface is nearest, votes further than `tol` are discarded, and the instances above
    `min_frac` of the surviving vote are returned.

    A reconstruction that spans two GT instances -- a merged label, which this pipeline
    produces -- shows up as two ids with substantial shares rather than as one id silently
    chosen, so the vote is printed and not only used.
    """
    from plyfile import PlyData
    p = PlyData.read(os.path.expanduser(path))
    fe = p["face"]
    key = "vertex_indices" if "vertex_indices" in fe.data.dtype.names else "vertex_index"
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float64)

    q = np.asarray(mesh.vertices)
    if len(q) > 20000:                       # the vote is a proportion; 20k resolves it
        q = q[np.random.default_rng(0).choice(len(q), 20000, replace=False)]
    pad = tol * 4.0
    lo, hi = q.min(0) - pad, q.max(0) + pad

    # Distance to the GT SURFACE, not to its vertices. Replica's instance meshes have large
    # flat triangles -- a tabletop is a handful of them -- so a point sitting in the middle of
    # one is far from every corner: the vertex version of this test reported "no GT instance
    # within 50 mm" for a reconstruction lying exactly on the surface.
    tris, tri_oid = [], []
    for face, oid in zip(fe[key], fe["object_id"]):
        idx = np.asarray(face, np.int64)
        c = V[idx]
        if (c.max(0) < lo).any() or (c.min(0) > hi).any():
            continue                          # instance is nowhere near this object
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
            tri_oid.append(int(oid))
    if not tris:
        print(f"[gt-auto] no GT face within {pad:.2f} m of the reconstruction")
        return []
    tri_oid = np.asarray(tri_oid, np.int64)
    near = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V),
        o3d.utility.Vector3iVector(np.asarray(tris, np.int32)))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(near))
    res = sc.compute_closest_points(o3d.core.Tensor(q.astype(np.float32)))
    hit = res["points"].numpy().astype(np.float64)
    prim = res["primitive_ids"].numpy().astype(np.int64)
    d = np.linalg.norm(hit - q, axis=1)

    ok = d <= tol
    if not ok.any():
        print(f"[gt-auto] no vertex within {tol:.3f} m of any GT surface "
              f"(nearest {d.min() * 1000:.0f} mm) -- the object may be misplaced entirely")
        return []
    votes, dist = {}, {}
    for oid, dd in zip(tri_oid[prim[ok]], d[ok]):
        votes[int(oid)] = votes.get(int(oid), 0) + 1
        dist[int(oid)] = dist.get(int(oid), 0.0) + float(dd)
    n = int(ok.sum())
    order = sorted(votes, key=lambda o: -votes[o])
    print(f"[gt-auto] {n}/{len(q)} vertices within {tol * 1000:.0f} mm of a GT instance")
    for oid in order[:top]:
        share = votes[oid] / n
        print(f"          object_id {oid:>5}  {share * 100:5.1f}%  "
              f"mean {dist[oid] / votes[oid] * 1000:5.1f} mm"
              f"{'   <== used' if share >= min_frac else ''}")
    chosen = [oid for oid in order if votes[oid] / n >= min_frac]
    print(f"          --gt_ids {','.join(str(o) for o in chosen)}")
    return chosen


def look_at(eye, target, up=(0, 0, 1)):
    """World->camera extrinsic (4x4) for a camera at `eye` pointing at `target`."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    n = np.linalg.norm(f)
    if n < 1e-9:
        sys.exit("[abort] camera and target coincide")
    f /= n
    up = np.asarray(up, float)
    if abs(float(np.dot(f, up / np.linalg.norm(up)))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])  # looking straight along up; pick another
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, -u, f])          # camera looks down +z, y down (open3d convention)
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ np.asarray(eye, float)
    return E


def render(scene, K, E, w, h, up_world):
    """Lambertian grey image of whatever the rays hit. Background stays white."""
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float64),
        extrinsic_matrix=o3d.core.Tensor(E, dtype=o3d.core.Dtype.Float64),
        width_px=w, height_px=h)
    ans = scene.cast_rays(rays)
    hit = ans["t_hit"].numpy()
    nrm = ans["primitive_normals"].numpy()
    img = np.ones((h, w, 3), np.float32)
    m = np.isfinite(hit)
    if m.any():
        # Two lights so a surface facing away from the key light is shaded, not black --
        # a single headlight makes the added back surface read as a hole.
        L1 = np.array([0.3, 0.4, 0.85]); L1 /= np.linalg.norm(L1)
        L2 = np.array([-0.6, -0.2, 0.4]); L2 /= np.linalg.norm(L2)
        n = nrm[m]
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        lam = 0.55 * np.abs(n @ L1) + 0.30 * np.abs(n @ L2) + 0.15
        img[m] = np.clip(lam, 0, 1)[:, None]
    return img


def strip(panels, pad=8):
    """Lay images out left to right with white gutters."""
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + pad * (len(panels) - 1)
    out = np.ones((h, w, 3), np.float32)
    x = 0
    for p in panels:
        out[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + pad
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", default="", help="side A: observation only (fuse_post.ply)")
    ap.add_argument("--recon2", default="", help="side B: fused (fused_<RUN>_post.ply)")
    # Comparing more than two settings by running this twice does not work: the camera is
    # fitted to the meshes of ONE call, so two output images have different framing and the
    # reader cannot tell a real difference from a different distance. Every mesh that will be
    # compared has to be drawn in the same call, through the same camera.
    ap.add_argument("--recons", default="",
                    help="two or more meshes to draw as rows, comma or space separated, "
                         "in place of --recon/--recon2. The FIRST one defines the front view "
                         "and should be side A. Row labels default to the file names")
    ap.add_argument("--gt", default="", help="optional GT mesh, drawn as a third row")
    ap.add_argument("--gt_ids", default="",
                    help="object_id values to keep from --gt, comma separated. Without this "
                         "the whole scene mesh is drawn, which is never what you want")
    ap.add_argument("--gt_auto", action="store_true",
                    help="work out --gt_ids from the first mesh by nearest-point vote, and "
                         "print the vote. Use when the GT id is not written down anywhere")
    ap.add_argument("--gt_tol", type=float, default=0.05,
                    help="--gt_auto: a vertex further than this from every GT instance casts "
                         "no vote (metres)")
    ap.add_argument("--colmap", default="",
                    help="sparse/0 with images.txt; the front view is taken from the "
                         "trajectory camera that sees A best")
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--tag", default="", help="printed to stdout with the numbers")
    # The rows used to be labelled "A: observation only" / "B: fused" whatever was passed.
    # Comparing two fusion settings then produced a picture whose own caption said the top
    # row was the observation -- a figure that lies about what it shows is worse than no
    # figure, so the labels follow the meshes.
    ap.add_argument("--labels", default="",
                    help="comma-separated row labels, in the order GT (if any), recon, "
                         "recon2. Default: 'A: observation only,B: fused'")
    ap.add_argument("--size", type=int, default=640, help="pixels per panel")
    ap.add_argument("--views", type=int, default=2,
                    help="azimuths, evenly spaced from the front. 2 = front and back")
    ap.add_argument("--elev", type=float, default=15.0, help="camera elevation, degrees")
    ap.add_argument("--fit", type=float, default=1.5,
                    help="camera distance as a multiple of the object radius")
    args = ap.parse_args()

    if args.recons:
        paths = [s.strip() for s in args.recons.replace(",", " ").split() if s.strip()]
        if len(paths) < 2:
            sys.exit("[abort] --recons needs at least two meshes")
        meshes = [(os.path.basename(p), load_mesh(p)) for p in paths]
    else:
        if not (args.recon and args.recon2):
            sys.exit("[abort] pass --recon and --recon2, or --recons with a list of meshes")
        meshes = [("A: observation only", load_mesh(args.recon)),
                  ("B: fused", load_mesh(args.recon2))]

    # The front view is chosen from whatever the first row is, which is why --recons asks for
    # side A first: "front" has to mean the direction the cameras actually observed.
    A = meshes[0][1]

    rows, labels = [], []
    if args.gt:
        ids = args.gt_ids
        if args.gt_auto and not ids:
            found = gt_ids_near(args.gt, A, tol=args.gt_tol)
            if not found:
                sys.exit("[abort] --gt_auto found no GT instance under this reconstruction. "
                         "Raise --gt_tol, or pass --gt_ids by hand.")
            ids = ",".join(str(o) for o in found)
        meshes.insert(0, ("GT", load_gt(args.gt, ids)))
    if args.labels:
        given = [s.strip() for s in args.labels.split(",")]
        if len(given) != len(meshes):
            sys.exit(f"[abort] --labels has {len(given)} entries but {len(meshes)} rows "
                     f"are drawn ({', '.join(n for n, _ in meshes)})")
        meshes = [(given[i], m) for i, (_, m) in enumerate(meshes)]

    # One frame for every row. The camera must be fitted to ALL the meshes drawn, not just
    # A and B: the GT instance is usually a little larger (it has the legs and the parts the
    # reconstruction missed), so fitting to A and B alone crops the GT row -- which is exactly
    # the row a reader checks the other two against.
    pts = np.vstack([np.asarray(m.vertices) for _, m in meshes])
    ctr = 0.5 * (pts.min(0) + pts.max(0))
    radius = float(np.linalg.norm(pts.max(0) - pts.min(0))) * 0.5
    dist = max(radius * args.fit, 1e-3)

    # Up is whichever world axis the object is thinnest in... no: up is the axis the scene
    # stands in, and for Replica that is +z. Deriving it from the object would tilt every
    # panel differently.
    up = np.array([0.0, 0.0, 1.0])

    front = np.array([1.0, 0.0, 0.0])
    if args.colmap:
        cams = read_colmap_images(args.colmap)
        if not cams:
            print(f"[warn] no images.txt under {args.colmap}; using world +X as front")
        else:
            sc = o3d.t.geometry.RaycastingScene()
            sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(A))
            best, best_hits = None, -1
            K_probe = np.array([[160, 0, 80], [0, 160, 80], [0, 0, 1]], float)
            for name, R, t in cams[::max(1, len(cams) // 120)]:
                E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
                cam_pos = -R.T @ t
                if np.linalg.norm(cam_pos - ctr) > radius * 12:
                    continue
                img = render(sc, K_probe, E, 160, 160, up)
                hits = int((img[:, :, 0] < 0.999).sum())
                if hits > best_hits:
                    best_hits, best = hits, cam_pos
            if best is not None and best_hits > 0:
                d = best - ctr
                d[2] = 0.0
                if np.linalg.norm(d) > 1e-6:
                    front = d / np.linalg.norm(d)
                print(f"[view] front taken from the trajectory camera with {best_hits} hits")

    S = args.size
    f = S * 1.1
    K = np.array([[f, 0, S / 2.0], [0, f, S / 2.0], [0, 0, 1]], float)

    az0 = np.arctan2(front[1], front[0])
    el = np.radians(args.elev)
    for label, mesh in meshes:
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        panels = []
        for i in range(args.views):
            az = az0 + 2 * np.pi * i / args.views
            eye = ctr + dist * np.array([np.cos(az) * np.cos(el),
                                         np.sin(az) * np.cos(el),
                                         np.sin(el)])
            panels.append(render(sc, K, look_at(eye, ctr, up), S, S, up))
        rows.append(strip(panels))
        labels.append(label)

    w = max(r.shape[1] for r in rows)
    grid = np.ones((sum(r.shape[0] for r in rows) + 8 * (len(rows) - 1), w, 3), np.float32)
    y = 0
    for r in rows:
        grid[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + 8

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    o3d.io.write_image(out, o3d.geometry.Image((grid * 255).astype(np.uint8)))

    print(f"[out] {out}")
    print(f"      rows top to bottom: {', '.join(labels)}")
    print(f"      columns: view 1 = front (observed), "
          f"{'view 2 = back (unobserved)' if args.views == 2 else f'{args.views} azimuths'}")
    if args.tag:
        print(f"      {args.tag}")


if __name__ == "__main__":
    main()