"""Re-colour a fused mesh: project the training images onto what was observed, fill the
rest by mirror symmetry.

The fusion colours every vertex from the nearest point of the oriented cloud. Adjacent
vertices grab different source points, which shows up as grid-aligned blotches, and the
global statistics match then flattens whatever texture survives. Neither step looks at
the images.

Observed vertices: average the images over the views where the vertex is the first
surface, weighted by the viewing angle. Self-occlusion is decided by raycasting the mesh
itself, so no depth files are needed.

Unobserved vertices carry no information -- any colour there is invented. The fill is
continuous at the seam and flat away from it:
  mirror     where a reflection plane maps the vertex onto observed surface, take that
             colour. This works for a chair (left-right) but NOT for a table seen only
             from above: no plane sends the underside to the top, so most of it falls
             through to the next rule.
  seam blend near the seam, the observed colour of the nearest vertex ALONG THE SURFACE.
             Straight-line nearest jumps through 2mm of tabletop and paints the underside
             with the top.
  base       far from the seam, the median colour of the observed vertices FACING THE
             SAME WAY. The shading is baked into the images, so a downward face is dark
             and an upward face is light; a single global median makes the invented
             underside light tan while the parts of it that were actually observed stay
             dark, and the two tones meet in the middle of the surface.

  python color_unseen.py --mesh OUT/objects_voted/6/train/ours_30000/fused_X_post.ply \\
      --colmap DATA/sparse/0 --images DATA/images --masks_root DATA/masks --gid 6 \\
      --out /tmp/obj6_coloured.ply
"""
import os, argparse, functools
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from PIL import Image
from scipy.ndimage import binary_erosion

print = functools.partial(print, flush=True)

try:
    from warp_gt_to_pose import read_colmap
except Exception:
    read_colmap = None


def load_image(images, stem):
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        p = os.path.join(os.path.expanduser(images), stem + ext)
        if os.path.exists(p):
            return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
    return None


def load_mask(masks_root, gid, stem, shape):
    if not (masks_root and gid):
        return None
    d = os.path.join(os.path.expanduser(masks_root), str(gid), "masks")
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(d, stem + ext)
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        a = np.asarray(im)
        if a.ndim == 3:                       # RGBA carries the mask in alpha
            a = a[..., 3] if a.shape[2] == 4 else np.asarray(im.convert("L"))
        m = (a == 188) if (a == 188).any() else (a > (0 if a.max() <= 1 else 127))
        if m.shape != shape:
            m = np.asarray(Image.fromarray(m.astype(np.uint8))
                           .resize((shape[1], shape[0]), Image.NEAREST)) > 0
        return m
    return None


def depth_buffer(rc, cam, H, W, ds):
    """The mesh's own depth from a camera. Self-occlusion is what decides whether a
    vertex was observed. An unnormalised ray direction makes t_hit the camera z."""
    fx, fy, cx, cy = cam["fx"] / ds, cam["fy"] / ds, cam["cx"] / ds, cam["cy"] / ds
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, np.float64)], -1)
    o = np.broadcast_to(-cam["R"].T @ cam["t"], d.shape)
    rays = np.concatenate([o, d @ cam["R"]], -1).astype(np.float32)
    z = rc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return np.where(np.isfinite(z), z, 0.0)


def project_colours(mesh, V, N, cams, stems, args):
    """Weighted mean of the images over the views that see each vertex.

    Accumulated online: keeping every view's contribution would be views x verts x 3
    floats, gigabytes on a mesh this size.
    """
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    num = np.zeros((len(V), 3), np.float64)
    den = np.zeros(len(V), np.float64)
    cnt = np.zeros(len(V), np.int32)
    n_used = 0

    for s in stems:
        c = cams[s]
        img = load_image(args.images, s)
        if img is None:
            continue
        hh, ww = img.shape[:2]
        H, W = hh // args.ds, ww // args.ds
        dbuf = depth_buffer(rc, c, H, W, args.ds)
        # A vertex one pixel inside the silhouette still samples the background, which is
        # what turns a tabletop rim white. Erode the hit region instead of trusting it.
        valid = dbuf > 0
        if args.silhouette_erode > 0:
            valid = binary_erosion(valid, iterations=args.silhouette_erode)

        Xc = V @ c["R"].T + c["t"]
        z = Xc[:, 2]
        zz = np.maximum(z, 1e-6)
        uf = c["fx"] * Xc[:, 0] / zz + c["cx"]            # full-resolution pixel
        vf = c["fy"] * Xc[:, 1] / zz + c["cy"]
        iu = np.round(uf / args.ds).astype(int)
        iv = np.round(vf / args.ds).astype(int)
        ok = (z > 0.05) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
        if not ok.any():
            continue
        ok[ok] &= valid[iv[ok], iu[ok]]
        if ok.any():
            ok[ok] &= np.abs(dbuf[iv[ok], iu[ok]] - z[ok]) < args.margin  # first surface

        msk = load_mask(args.masks_root, args.gid, s, (H, W))
        if msk is not None and args.silhouette_erode > 0:
            msk = binary_erosion(msk, iterations=args.silhouette_erode)
        if msk is not None and ok.any():
            ok[ok] &= msk[iv[ok], iu[ok]]
        if not ok.any():
            continue

        # a grazing view carries the least reliable colour and the most bleed
        view = (-c["R"].T @ c["t"])[None] - V
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
        w = np.abs((N * view).sum(1))
        ok &= w > args.min_cos
        if not ok.any():
            continue

        pu = np.clip(np.round(uf[ok]).astype(int), 0, ww - 1)
        pv = np.clip(np.round(vf[ok]).astype(int), 0, hh - 1)
        ws = (w[ok] ** 2)[:, None]                       # sharpen toward face-on views
        num[ok] += img[pv, pu] * ws
        den[ok] += ws[:, 0]
        cnt[ok] += 1
        n_used += 1

    assert n_used, "no training image was found or projected; check --images and --colmap"
    seen = cnt >= args.min_views
    C = np.zeros((len(V), 3), np.float32)
    C[seen] = (num[seen] / den[seen, None]).astype(np.float32)
    print(f"[project] {n_used} views, {int(seen.sum()):,}/{len(V):,} vertices observed "
          f"({seen.mean()*100:.1f}%)")
    return C, seen


def mirror_plane(P):
    """Best reflection plane among the PCA axes through the centroid.

    The residual, against the object's diagonal, says whether the mirror fill is safe.
    """
    c = P.mean(0)
    Q = P - c
    axes = np.linalg.svd(Q, full_matrices=False)[2]
    tree = cKDTree(P)
    best_n, best_r = None, np.inf
    for n in axes:
        M = P - 2.0 * ((Q @ n)[:, None]) * n[None]
        r = float(np.median(tree.query(M, workers=-1)[0]))
        if r < best_r:
            best_n, best_r = n, r
    return best_n, c, best_r


def base_colour(N, C, seen, unseen, k, fallback):
    """Median colour of the observed vertices pointing the same way.

    Orientation is the strongest predictor of shading here, and the shading is baked
    into the images, so this is what keeps an invented underside the same tone as the
    parts of the underside that happened to be observed.
    """
    ns = N[seen]
    if len(ns) < k:
        return np.broadcast_to(fallback, (int(unseen.sum()), 3)).copy()
    _, j = cKDTree(ns).query(N[unseen], k=min(k, len(ns)), workers=-1)
    return np.median(C[seen][j], axis=1)


def geodesic_source(V, F, seen):
    """Nearest observed vertex along the surface, and how far it is."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    a = np.concatenate([e[:, 0], e[:, 1]])
    b = np.concatenate([e[:, 1], e[:, 0]])
    g = coo_matrix((np.concatenate([w, w]), (a, b)), shape=(len(V), len(V))).tocsr()
    dist, _, src = dijkstra(g, directed=False, indices=np.where(seen)[0],
                            min_only=True, return_predecessors=True)
    return dist, src


def smooth(V, F, C, mask, iters):
    """Laplacian passes on the filled vertices only; they also blend across the seam."""
    if iters <= 0 or not mask.any():
        return C
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    a = np.concatenate([e[:, 0], e[:, 1]])
    b = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.maximum(np.bincount(a, minlength=len(V)), 1).astype(np.float32)
    for _ in range(iters):
        s = np.zeros_like(C)
        np.add.at(s, a, C[b])
        C = np.where(mask[:, None], s / deg[:, None], C)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks_root", default="")
    ap.add_argument("--gid", default="")
    ap.add_argument("--stems", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", default=200, type=int, help="0 = every view")
    ap.add_argument("--ds", default=2, type=int, help="depth buffer downscale")
    ap.add_argument("--margin", default=0.02, type=float,
                    help="|z - mesh depth| for counting a vertex as the first surface")
    ap.add_argument("--min_cos", default=0.3, type=float,
                    help="skip views seeing the vertex more than ~72 deg off its normal")
    ap.add_argument("--min_views", default=2, type=int)
    ap.add_argument("--mirror_max_res", default=0.06, type=float,
                    help="reject the mirror fill above this plane residual, as a fraction "
                         "of the object diagonal")
    ap.add_argument("--mirror_tol", default=0.02, type=float,
                    help="a reflected vertex must land this close (m) to observed surface")
    ap.add_argument("--silhouette_erode", default=3, type=int,
                    help="pixels to erode the visible region by before sampling, so a "
                         "vertex near the outline cannot pick up the background")
    ap.add_argument("--blend_dist", default=0.08, type=float,
                    help="distance along the surface (m) over which the fill goes from "
                         "the neighbouring observed colour to the base colour")
    ap.add_argument("--base_mode", default="normal", choices=["normal", "global"],
                    help="normal: median of the observed vertices facing the same way, "
                         "which keeps the invented underside as dark as the observed "
                         "underside. global: one median for the whole object")
    ap.add_argument("--base_k", default=200, type=int,
                    help="observed vertices entering the per-orientation median")
    ap.add_argument("--smooth", default=3, type=int, help="Laplacian passes on the fill")
    ap.add_argument("--tint", default=0.0, type=float,
                    help="blend the filled region toward magenta, for a figure that shows "
                         "which part is invented")
    args = ap.parse_args()

    assert read_colmap is not None, "could not import warp_gt_to_pose; run from the repo root"
    mesh = o3d.io.read_triangle_mesh(os.path.expanduser(args.mesh))
    assert len(mesh.triangles), f"no triangles in {args.mesh}"
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    N = np.asarray(mesh.vertex_normals)
    print(f"[mesh] {len(V):,} verts  {len(F):,} tris")

    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    if args.stems and os.path.exists(os.path.expanduser(args.stems)):
        stems = [l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()]
    else:
        stems = sorted(cams)
    stems = [s for s in stems if s in cams]
    if args.n_views > 0 and len(stems) > args.n_views:
        k = np.linspace(0, len(stems) - 1, args.n_views).round().astype(int)
        stems = [stems[i] for i in np.unique(k)]

    C, seen = project_colours(mesh, V, N, cams, stems, args)
    unseen = ~seen
    print(f"[fill] {int(unseen.sum()):,} vertices to fill ({unseen.mean()*100:.1f}%)")

    filled = np.zeros(len(V), bool)
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    if unseen.any() and seen.sum() > 100:
        n, c0, res = mirror_plane(V[seen])
        print(f"[mirror] residual {res*1000:.1f}mm ({res/ext*100:.1f}% of the diagonal)")
        if res / ext <= args.mirror_max_res:
            Q = V[unseen] - c0
            M = V[unseen] - 2.0 * ((Q @ n)[:, None]) * n[None]
            d, j = cKDTree(V[seen]).query(M, workers=-1)
            hit = d < args.mirror_tol
            si = np.where(seen)[0]
            ui = np.where(unseen)[0]
            C[ui[hit]] = C[si[j[hit]]]
            filled[ui[hit]] = True
            print(f"[mirror] filled {int(hit.sum()):,}/{int(unseen.sum()):,}")
        else:
            print(f"[mirror] rejected (> {args.mirror_max_res*100:.0f}%); geodesic only")

    rest = unseen & ~filled
    if rest.any() and seen.any():
        dist, src = geodesic_source(V, F, seen)
        i = np.where(rest)[0]
        ok = src[i] >= 0
        gm = np.median(C[seen], axis=0)
        if args.base_mode == "normal":
            B = base_colour(N, C, seen, rest, args.base_k, gm)
        else:
            B = np.broadcast_to(gm, (int(rest.sum()), 3)).copy()
        w = np.clip(1.0 - dist[i] / max(args.blend_dist, 1e-6), 0.0, 1.0)[:, None]
        C[i[ok]] = (w[ok] * C[src[i][ok]] + (1.0 - w[ok]) * B[ok]).astype(np.float32)
        filled[i[ok]] = True
        print(f"[fill] base={args.base_mode}  global median {np.round(gm, 3)}  "
              f"spread {np.round(B.std(0), 3)}  blend {args.blend_dist*100:.0f}cm")
    if (unseen & ~filled).any():
        print(f"[fill] {int((unseen & ~filled).sum()):,} vertices left black "
              f"(disconnected from any observed surface)")

    C = smooth(V, F, C, unseen, args.smooth)
    if args.tint > 0:
        C[unseen] = (1 - args.tint) * C[unseen] + args.tint * np.array([1.0, 0.0, 1.0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    assert o3d.io.write_triangle_mesh(out, mesh), f"failed to write {out}"
    print(f"[out] {out}")


if __name__ == "__main__":
    main()