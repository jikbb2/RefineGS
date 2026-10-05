#!/usr/bin/env python3
"""Seen / unseen split evaluation -- does the prior fill the unobserved WITHOUT
disturbing the observed?

A single chamfer or F-score cannot separate "the observation was corrupted" from
"the generated part is inaccurate". A visibility oracle labels GT points and recon
points by the SAME rule, and the metrics are then split by region. The oracle is a
function of 3-D position alone (is this point visible from an input camera), so it
applies identically to any point set, and the GT labels are fixed independently of
the method -- which is what makes a baseline/ours comparison valid.

  Default oracle = ray casting against the GT scene mesh (--vis_source gt_mesh):
    visibility is defined purely as "is the GT scene visible from this input camera
    pose", with no dependence on a depth-file format or scale convention. That makes
    it dataset independent and easy to state as a paper protocol.
    (--vis_source gt_depth selects a depth-map oracle instead.)

  Labels (decided per view, then combined by min_views consensus):
    seen      |z - d_gt| < margin   -> first surface in that view = genuinely observed
    free      z < d_gt - margin     -> observed empty space (recon points only; a
                                       definite error)
    unseen    neither (occluded / outside the frustum) -> no information = the region
                                       generation is supposed to fill

  Metrics:
    accuracy   (recon->GT)  : the seen region should match the baseline (observation
                              preserved)
    completion (GT->recon)  : the unseen region should improve (the prior's contribution)
    F@thr                   : precision over recon points, recall over GT points, per region

Claim form for the paper: "seen accuracy preserved + unseen completion improved".

  python eval_seen_unseen.py --gt_mesh gt_obj1.ply \
    --gt_scene_mesh "${REPLICA_ROOT}"/room_0/habitat/mesh_semantic.ply \
    --recon output/.../1/train/ours_7000/fuse_post.ply \
    --recon2 output/.../1/train/ours_7000/fused_prior.ply \
    --colmap data/replica_room0_v2/sparse/0 --gid 1 \
    --stems "${STEMS_DIR}"/1.txt
"""
import os
import glob
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image

try:
    from warp_gt_to_pose import read_colmap, cam_center
except Exception:                                    # running from outside the repo
    read_colmap = None

    def cam_center(R, t):
        return -R.T @ t


# --------------------------------------------------------------------------
def load_gt_depth(depth_dir, stem, scale):
    """Load a GT depth map (same convention as sdf_distill_depth.load_gt_depth)."""
    for c in (stem.replace("frame", "depth"), stem, stem + "_depth"):
        for ext in (".png", ".npy"):
            p = os.path.join(os.path.expanduser(depth_dir), c + ext)
            if not os.path.exists(p):
                continue
            if ext == ".npy":
                return np.load(p).astype(np.float32)
            return np.array(Image.open(p)).astype(np.float32) / scale
    return None


def load_mesh_labeled(path):
    """PLY -> (V, T, L). L = per-face object_id, or None when absent.

    Replica's mesh_semantic.ply carries an object_id face property AND polygon (quad)
    faces, which makes Open3D's RPly reader fail on the header. Read it with plyfile
    instead and fan-triangulate.
    """
    path = os.path.expanduser(path)
    try:
        from plyfile import PlyData
    except ImportError:
        PlyData = None
        print("[mesh] plyfile not installed -- cannot extract object_id. pip install plyfile")
    if PlyData is not None:
        try:
            p = PlyData.read(path)
            v = p["vertex"].data
            V = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
            fe = p["face"].data
            names = fe.dtype.names
            key = "vertex_indices" if "vertex_indices" in names else "vertex_index"
            polys = fe[key]
            oid = fe["object_id"].astype(np.int64) if "object_id" in names else None
            lens = np.fromiter((len(x) for x in polys), int, len(polys))
            tris, labs = [], []
            for L_ in np.unique(lens):                 # vectorised fan triangulation, per face size
                sel = np.where(lens == L_)[0]
                arr = np.stack([np.asarray(polys[i]) for i in sel]).astype(np.int64)
                for k in range(1, int(L_) - 1):
                    tris.append(arr[:, [0, k, k + 1]])
                    if oid is not None:
                        labs.append(oid[sel])
            T = np.vstack(tris)
            Lb = np.concatenate(labs) if oid is not None else None
            print(f"[mesh] {os.path.basename(path)}: verts {len(V)}, tris {len(T)}"
                  + (f", {len(np.unique(Lb))} distinct object_id" if Lb is not None else ""))
            return V, T, Lb
        except Exception as e:
            print(f"[mesh] plyfile load failed ({e}) -- falling back to Open3D")
    m = o3d.io.read_triangle_mesh(path)
    assert len(m.triangles), f"mesh load failed (no triangles): {path}"
    return (np.asarray(m.vertices), np.asarray(m.triangles).astype(np.int64), None)


def sample_tris(V, T, n, seed=0):
    """Area-weighted uniform sample -> (points, owning triangle index).

    Implemented here rather than via Open3D so the per-face labels can be carried along.
    """
    rng = np.random.default_rng(seed)
    e1 = V[T[:, 1]] - V[T[:, 0]]; e2 = V[T[:, 2]] - V[T[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    tot = area.sum()
    assert tot > 0, "mesh has zero area"
    idx = rng.choice(len(T), n, p=area / tot)
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    P = ((1 - r1)[:, None] * V[T[idx, 0]]
         + (r1 * (1 - r2))[:, None] * V[T[idx, 1]]
         + (r1 * r2)[:, None] * V[T[idx, 2]])
    return P, idx


def _aabb(P):
    return P.min(0), P.max(0)


def _aabb_gap(a, b):
    """Separation between two axis-aligned boxes in metres; 0 when they overlap."""
    (alo, ahi), (blo, bhi) = a, b
    return float(np.linalg.norm(np.maximum(np.maximum(blo - ahi, alo - bhi), 0.0)))


def auto_match_labels(V, T, L, ref_pts, min_share=0.10, n=300000, max_dist=0.05,
                      max_gap=0.15, min_cover=0.30):
    """Pick, by voting, the set of object_ids that overlap the reconstruction.

    A SAM3 instance is not 1:1 with a dataset semantic id -- one instance may span
    several GT objects, or the reverse. Choosing a single label makes GT too small or
    too large, so every label holding at least min_share of the vote is accepted, and
    the composition is printed so a human can check it.

    WARNING: the vote cutoff (80th percentile) is RELATIVE, so even a reconstruction
      that overlaps no GT object at all produces a landslide for whichever label is
      nearest. Measured: obj21 matched id25 with 95.3% of the vote, yet its unseen
      completion was 3278 mm (i.e. GT was 3.3 m away). The absolute distance is
      therefore reported alongside. When it is large, do not interpret that row --
      the problem is the reconstruction, not the matching.
    """
    P, idx = sample_tris(V, T, min(n, 20 * len(T) + 1000))
    lab = L[idx]
    d, j = cKDTree(P).query(ref_pts, workers=-1)
    med = float(np.median(d))
    print(f"[auto-match] recon->GT median distance {med*1000:.1f}mm "
          f"(80th pct {np.percentile(d, 80)*1000:.1f}mm, threshold {max_dist*1000:.0f}mm)")
    if med > max_dist:
        print(f"[auto-match] !! this reconstruction overlaps NO GT object "
              f"({med*1000:.0f}mm away). The vote below is merely the NEAREST label; "
              f"do not interpret this object's metrics.")
        print(f"[auto-match]    the cause is the reconstruction, not the matching -- "
              f"check whether the mask points at a different object, or whether "
              f"per-object training failed.")
    keep = d < max(np.percentile(d, 80), 1e-6)        # drop far points (floaters)
    vals, cnt = np.unique(lab[j][keep], return_counts=True)
    share = cnt / max(keep.sum(), 1)
    order = np.argsort(-share)
    print("[auto-match] vote composition: " + ", ".join(
        f"id{int(vals[i])} {share[i]*100:.1f}%" for i in order[:6]))
    # A share threshold alone cannot tell "one object split across several ids" from "our
    # instance leaked onto the neighbour": both look like a second label holding ~20% of
    # the vote. The difference is spatial. A split part sits ON the reconstruction; a
    # neighbouring chair sits a chair-width away, and unioning it makes GT->recon report
    # the gap between two objects as unseen completion (measured: 1408mm median on obj2).
    # So a non-dominant label is accepted only if its own box touches the reconstruction's.
    # Unobserved parts of the SAME object stay inside that box, so this does not penalise
    # the very geometry the method is meant to complete.
    # The bbox test cannot do its job for objects that TOUCH. Measured 0929 on room0: every
    # non-dominant label on every object reported a 0mm gap, because a cushion sits on a sofa
    # and a lamp sits on a table -- so a label our reconstruction merely leaked 5% of its
    # points onto was unioned into the GT exactly like a genuine part. The consequence is not
    # subtle: unseen completion tracked the number of unioned labels almost perfectly --
    # obj6 [11] 25mm, obj18 [71,9] 1040mm, obj15 [5,13,77] 790mm, obj20 [27,7,60,20,25]
    # 2634mm. GT->recon was reporting the distance from a NEIGHBOUR's surface to our object.
    #
    # Coverage is the test the gap was standing in for, and it runs the other way round. The
    # vote asks "where do OUR points land"; this asks "how much of THAT instance did we
    # reconstruct". A GT part we genuinely recovered is mostly covered by us; a neighbour we
    # brushed against is not, however firmly the two boxes touch. An unobserved part of the
    # same object is NOT penalised: coverage is measured against the reconstruction we are
    # about to evaluate, and side A already spans the object's observed extent.
    rtree = cKDTree(ref_pts)
    rb = _aabb(ref_pts)
    sel, dropped, cov = [], [], 0.0
    print(f"{'  label':>8}{'share':>8}{'bbox gap':>10}{'covered':>10}   verdict")
    for i in order:
        if share[i] < min_share:
            continue
        lid = int(vals[i])
        lp = P[lab == lid]
        gap = _aabb_gap(_aabb(lp), rb)
        dc, _ = rtree.query(lp, workers=-1)
        cover = float((dc <= max_dist).mean())
        why = ""
        if not sel:
            take = True                               # the top label is always kept
        elif gap > max_gap:
            take, why = False, f"DROP (> {max_gap*1000:.0f}mm away)"
        elif cover < min_cover:
            take, why = False, f"DROP (only {cover*100:.0f}% of it reconstructed)"
        else:
            take = True
        print(f"  id{lid:<5}{share[i]*100:7.1f}%{gap*1000:9.0f}mm{cover*100:9.1f}%   "
              + (why or "accept"))
        (sel if take else dropped).append(lid)
        if take:
            cov += float(share[i])
    if not sel:
        sel = [int(vals[order[0]])]
    print(f"  -> accepted labels {sel} (total {cov*100:.1f}%)"
          + (f"   dropped {dropped}" if dropped else ""))
    if dropped:
        print("  (a dropped label is a separate object away from the reconstruction -- "
              "unioning it makes unseen completion measure the distance between two "
              "objects. Tune with --match_max_gap)")
    if cov < 0.7 and not dropped:
        print("  ! low coverage -- set --gt_labels explicitly or adjust min_share")
    return sel


def raycast_depth(scene, cam, ds):
    """Ray cast the GT scene mesh from a camera pose -> z-depth (the recommended oracle).

    Visibility is defined only as "is the GT scene visible from this input camera pose",
    with no dependence on depth-file format or scale conventions, so it holds for any
    dataset -- which makes it easy to state as a paper protocol.
    Leaving the direction vectors unnormalised (z component = 1) makes t_hit equal the
    camera z-depth, so the convention matches the rest of the pipeline.
    NOTE: occlusion is decided by the WHOLE scene, so `scene` must be the full GT mesh,
    not the object mesh.
    """
    W = int(np.ceil(cam["W"] / ds)); H = int(np.ceil(cam["H"] / ds))
    fx, fy = cam["fx"] / ds, cam["fy"] / ds
    cx, cy = cam["cx"] / ds, cam["cy"] / ds
    R, t = cam["R"], cam["t"]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dcam = np.stack([(uu - cx) / fx, (vv - cy) / fy,
                     np.ones_like(uu, float)], -1).reshape(-1, 3)
    dwn = (R.T @ dcam.T).T                             # world directions (unnormalised)
    C = cam_center(R, t)
    rays = np.concatenate([np.broadcast_to(C.astype(np.float32), dwn.shape),
                           dwn.astype(np.float32)], 1)
    th = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)
    return np.where(np.isfinite(th), th, 0.0).astype(np.float32)


def load_mask(masks_root, gid, stem):
    p = os.path.join(masks_root, str(gid), "masks", stem + ".png")
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p))
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., 3]
    elif a.ndim == 3:
        a = np.array(Image.open(p).convert("L"))
    if a.max() <= 1:
        return a > 0
    if (a == 188).any():
        return a == 188                              # amodal convention: 188 = visible
    return a > 127


def build_views(args, scene_mesh=None):
    """Camera + visibility depth (+ mask) view buffers. Downscaled for memory and speed.

    scene_mesh: the WHOLE-scene Open3D mesh, for the gt_mesh oracle.
    """
    assert read_colmap is not None, \
        "failed to import warp_gt_to_pose -- run this from the repository root"
    cams = {c["stem"]: c for c in read_colmap(args.colmap)}
    if args.stems and os.path.exists(os.path.expanduser(args.stems)):
        stems = [l.strip() for l in open(os.path.expanduser(args.stems)) if l.strip()]
    else:
        stems = sorted(cams)
    stems = [s for s in stems if s in cams]
    if args.n_views > 0 and len(stems) > args.n_views:      # uniform-interval subsample
        idx = np.linspace(0, len(stems) - 1, args.n_views).round().astype(int)
        stems = [stems[i] for i in np.unique(idx)]

    ds = max(1, args.ds)
    rc_scene = None
    if args.vis_source == "gt_mesh":
        assert scene_mesh is not None, "the gt_mesh oracle needs a scene mesh"
        rc_scene = o3d.t.geometry.RaycastingScene()
        rc_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene_mesh))
        print(f"[oracle] GT scene mesh ray casting (tri {len(scene_mesh.triangles)})")
    else:
        print("[oracle] GT depth maps (--vis_source gt_depth)")

    views, n_mask = [], 0
    for s in stems:
        c = cams[s]
        if rc_scene is not None:
            dg = raycast_depth(rc_scene, c, ds)
        else:
            dg = load_gt_depth(args.gt_depth_dir, s, args.gt_depth_scale)
            dg = dg[::ds, ::ds] if dg is not None else None
        if dg is None:
            continue
        m = (load_mask(args.masks_root, args.gid, s)
             if (args.masks_root and args.gid) else None)
        if m is not None and m.shape != dg.shape:
            m = np.array(Image.fromarray(m.astype(np.uint8))
                         .resize((dg.shape[1], dg.shape[0]), Image.NEAREST)) > 0
        v = {"R": c["R"], "t": c["t"],
             "fx": c["fx"] / ds, "fy": c["fy"] / ds,
             "cx": c["cx"] / ds, "cy": c["cy"] / ds,
             "dgt": dg}
        v["H"], v["W"] = v["dgt"].shape
        if m is not None:
            v["mask"] = m if m.shape == dg.shape else m[::ds, ::ds]
            n_mask += 1
        views.append(v)
    print(f"[views] {len(views)} views, {n_mask} with masks, ds={ds}")
    assert views, "0 visibility views -- check --vis_source and the path / filename convention"
    return views


def classify(P, views, margin, min_views, use_mask):
    """Point labels: seen (matches the first surface) / free (observed empty space) /
    unseen (no information).

    Returns (seen: bool, free: bool) -- free is True only on points that are not seen.
    """
    n_seen = np.zeros(len(P), np.int32)
    n_free = np.zeros(len(P), np.int32)
    for v in views:
        Xc = P @ v["R"].T + v["t"]
        z = Xc[:, 2]
        zz = np.maximum(z, 1e-6)
        u = v["fx"] * Xc[:, 0] / zz + v["cx"]
        w = v["fy"] * Xc[:, 1] / zz + v["cy"]
        infr = (z > 0.05) & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
        ui = np.clip(u, 0, v["W"] - 1).astype(int)
        wi = np.clip(w, 0, v["H"] - 1).astype(int)
        d = v["dgt"][wi, ui]
        ok = infr & (d > 0.01)
        if use_mask and "mask" in v:
            ok = ok & v["mask"][wi, ui]
        n_seen += (ok & (np.abs(z - d) < margin)).astype(np.int32)
        n_free += (ok & (z < d - margin)).astype(np.int32)
    seen = n_seen >= min_views
    free = (~seen) & (n_free >= min_views)
    return seen, free


def sample(mesh_path, n, seed=0):
    """Uniform mesh sampling. Settings must be compared under the SAME seed.

    There are two sources of noise, and the interpretation threshold is dominated by
    evaluation sampling.

    (1) Evaluation sampling -- mesh fixed, seed varied (eval_noise.sh, obj6, 5 runs)
        seen F@1cm  +/-0.0001    seen acc      +/-0.022mm
        seen P@1cm  +/-0.0003    unseen acc    +/-0.26mm
        seen R@1cm  +/-0.0004    unseen P@2cm  +/-0.0033
        free viol.  +/-0.085%p   unseen R@2cm  +/-0.0005

    (2) Fusion -- same settings run twice, evaluation seed fixed (measured on the
        outputs of verify_gpu_fuse.sh)
        unseen F@2cm +/-0.0002   free +/-0.1%p   seen composition +/-0.1%p
        The two meshes differ by 0.0497mm Chamfer (2DGS rendering is non-deterministic),
        but the metrics average over 200,000 points, so local differences cancel and
        this source is 16x smaller than (1).

    -> Interpretation threshold: do not interpret a difference below 0.007 (2 sigma) in
      unseen F@2cm, or below 0.2%p in free violation. In practice the unseen F@2 spread
      across wcap 8/16/32 (0.005-0.006) and the obj22 gt_edge_thr difference (0.0033)
      both fell inside that band, and both decisions were made on other axes (seen
      accuracy, free violation) where the differences were far larger than the noise.
    """
    m = o3d.io.read_triangle_mesh(os.path.expanduser(mesh_path))
    assert len(m.vertices), f"mesh load failed: {mesh_path}"
    if len(m.triangles) == 0:
        return np.asarray(m.vertices), None
    try:                                    # global RNG, Open3D >= 0.16
        o3d.utility.random.seed(int(seed))
    except Exception:
        pass
    m.compute_vertex_normals()              # for NC -- normals ride along on the samples
    try:                                    # some versions accept a seed argument
        pc = m.sample_points_uniformly(number_of_points=n, seed=int(seed))
    except TypeError:
        pc = m.sample_points_uniformly(number_of_points=n)
    N = np.asarray(pc.normals) if pc.has_normals() else None
    return np.asarray(pc.points), N


# --------------------------------------------------------------------------
# Completion-side metrics. Chamfer and F-score are SURFACE measures: a hollow
# shell and a solid of the same outline score alike, and an inflated surface is
# only penalised where it drifts past the threshold. Design C failed by
# inflating, so volume and closedness are measured directly.

def mesh_watertight(path):
    """(open boundary edges, is_watertight). An unfilled region leaves a hole, so
    this is the completion counterpart to a distance metric."""
    m = o3d.io.read_triangle_mesh(os.path.expanduser(path))
    T = np.asarray(m.triangles)
    if len(T) == 0:
        return -1, False
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    n_open = int((cnt == 1).sum())
    return n_open, bool(m.is_watertight())


def occupancy_iou(Vg, Tg, recon_path, voxel):
    """Solid IoU over a shared voxel grid, via ray-parity occupancy.

    0.01 m, not the fusion's 0.005: this measures gross volume (hollow, inflated),
    which the surface metrics already cover at fine scale, and halving the voxel
    costs 8x the queries for no change in what it detects.
    Occupancy is only meaningful on closed meshes; returns nan when either side is
    open rather than reporting a number that means nothing.
    """
    mr = o3d.io.read_triangle_mesh(os.path.expanduser(recon_path))
    if len(mr.triangles) == 0:
        return float("nan")
    mg = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vg),
                                   o3d.utility.Vector3iVector(Tg.astype(np.int32)))
    if not (mr.is_watertight() and mg.is_watertight()):
        return float("nan")
    lo = np.minimum(np.asarray(mr.vertices).min(0), Vg.min(0)) - 2 * voxel
    hi = np.maximum(np.asarray(mr.vertices).max(0), Vg.max(0)) + 2 * voxel
    g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i], voxel) for i in range(3)],
                             indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    if len(g) > 40_000_000:
        print(f"[iou] grid {len(g):,} too large at voxel {voxel} -- skipped")
        return float("nan")
    q = o3d.core.Tensor(g)
    occ = []
    for m in (mr, mg):
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        occ.append(sc.compute_occupancy(q).numpy().astype(bool))
    inter = float((occ[0] & occ[1]).sum()); union = float((occ[0] | occ[1]).sum())
    return inter / union if union > 0 else float("nan")


def preservation(A, B, thr):
    """Share of A's surface that survives in B, within thr.

    The claim is that fusion fills the unobserved WITHOUT disturbing the observed,
    and seen_acc only shows that indirectly: a method can hold its accuracy while
    replacing the surface with a different one of similar quality. This is
    one-directional on purpose -- B's new material is the contribution, not an error.
    """
    d, _ = cKDTree(B).query(A, workers=-1)
    return float((d < thr).mean())


def _stat(d):
    return (float(d.mean() * 1000), float(np.median(d) * 1000)) if len(d) else (float("nan"),) * 2


def report(name, RN, G, gs, gf, thresholds, views, args, GN=None,
           mesh_path="", gt_VT=None):
    """Print and return per-region metrics for recon (points, normals) RN against GT
    point cloud G with GT labels gs (seen)."""
    R, RNn = RN if isinstance(RN, tuple) else (RN, None)
    rs, rf = classify(R, views, args.margin, args.min_views, args.use_mask)
    dR, jR = cKDTree(G).query(R, workers=-1)         # for accuracy (recon->GT)
    dG, jG = cKDTree(R).query(G, workers=-1)         # for completion (GT->recon)

    # [NC] Normal Consistency -- mean |cos| between corresponding normals (0-1, higher
    #   is better). Chamfer distance sees POSITION only: a bumpy or flipped surface
    #   passes as long as the positions line up. DP-Recon and others report it, so the
    #   same axis is provided here.
    #   The sign is ignored (|cos|) because mesh winding differs between pipelines.
    ncR = ncG = None
    if RNn is not None and GN is not None and len(RNn) == len(R):
        ncR = np.abs((RNn * GN[jR]).sum(1))          # per recon point
        ncG = np.abs((GN * RNn[jG]).sum(1))          # per GT point

    print(f"\n===== {name} =====")
    print(f"  composition  recon: seen {rs.mean()*100:5.1f}%  free-violation "
          f"{rf.mean()*100:5.1f}%  unseen {(~rs & ~rf).mean()*100:5.1f}%   |  "
          f"GT: seen {gs.mean()*100:.1f}%")
    M = {"mesh": os.path.basename(name.split(": ")[-1]),
         "free_pct": float(rf.mean() * 100),
         "recon_seen_pct": float(rs.mean() * 100),
         "recon_unseen_pct": float((~rs & ~rf).mean() * 100),
         "gt_seen_pct": float(gs.mean() * 100)}
    for lab, key, rm, gm in (("ALL   ", "all", np.ones(len(R), bool), np.ones(len(G), bool)),
                             ("SEEN  ", "seen", rs, gs),
                             ("UNSEEN", "unseen", ~rs, ~gs)):
        am, amd = _stat(dR[rm]); cm, cmd = _stat(dG[gm])
        M[f"{key}_acc"] = am; M[f"{key}_acc_med"] = amd
        M[f"{key}_comp"] = cm; M[f"{key}_comp_med"] = cmd
        nc = float("nan")
        if ncR is not None:
            v = [x for x in (ncR[rm], ncG[gm]) if len(x)]
            nc = float(np.mean([x.mean() for x in v])) if v else float("nan")
        M[f"{key}_NC"] = nc
        print(f"  [{lab}] accuracy {am:7.2f}mm (med {amd:6.2f})   "
              f"completion {cm:7.2f}mm (med {cmd:6.2f})"
              + (f"   NC {nc:.4f}" if nc == nc else ""))
        for thr in thresholds:
            p = float((dR[rm] < thr).mean()) if rm.sum() else float("nan")
            r = float((dG[gm] < thr).mean()) if gm.sum() else float("nan")
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            tn = f"{thr*100:.1f}"
            M[f"{key}_F{tn}"] = f; M[f"{key}_P{tn}"] = p; M[f"{key}_R{tn}"] = r
            print(f"        F@{tn}cm {f:.4f} (P {p:.4f} / R {r:.4f})")
    if rf.any():
        am, _ = _stat(dR[rf])
        print(f"  [FREE violation] {int(rf.sum())} points ({rf.mean()*100:.1f}%) -- "
              f"surface in observed empty space, accuracy {am:.2f}mm  NOTE: a definite error")
    if mesh_path and not args.no_extra_metrics:
        n_open, wt = mesh_watertight(mesh_path)
        M["open_edges"] = n_open
        M["watertight"] = int(wt)
        M["vol_iou"] = (occupancy_iou(gt_VT[0], gt_VT[1], mesh_path, args.iou_voxel)
                        if gt_VT is not None else float("nan"))
        print(f"  [closure] open boundary edges {n_open}  watertight {wt}"
              + (f"   volumetric IoU {M['vol_iou']:.4f}" if M["vol_iou"] == M["vol_iou"]
                 else "   volumetric IoU n/a (a mesh is open)"))
    M["seen_acc_mm"] = M["seen_acc"]; M["unseen_comp_mm"] = M["unseen_comp"]
    return M


def write_csv(path, rows, tag):
    """Append rows for sweep comparison (header written automatically on a new file)."""
    import csv
    cols = ["tag", "mesh", "gt_seen_pct", "recon_seen_pct", "recon_unseen_pct", "free_pct",
            "seen_acc", "seen_acc_med", "seen_comp", "seen_comp_med",
            "seen_F1.0", "seen_P1.0", "seen_R1.0",
            "unseen_acc", "unseen_acc_med", "unseen_comp", "unseen_comp_med",
            "unseen_F2.0", "unseen_P2.0", "unseen_R2.0",
            # NC was computed in report() but never reached the CSV
            "all_NC", "seen_NC", "unseen_NC",
            "open_edges", "watertight", "vol_iou", "preservation"]
    path = os.path.expanduser(path)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        for r in rows:
            w.writerow([tag] + [f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, "")
                                for c in cols[1:]])
    print(f"[csv] -> {path}")


def main():
    ap = argparse.ArgumentParser(description="seen / unseen split evaluation")
    ap.add_argument("--gt_mesh", required=True,
                    help="GT mesh. Pass Replica's mesh_semantic.ply directly and the "
                         "target object is extracted by object_id (or name it with "
                         "--gt_labels)")
    ap.add_argument("--gt_labels", default="",
                    help="comma-separated object_ids to extract. Leave empty to match "
                         "automatically by overlap voting -- this also covers a SAM3 "
                         "instance that spans several GT objects. 'all' skips label "
                         "selection and compares against the whole scene (scene-level "
                         "evaluation)")
    ap.add_argument("--match_max_dist", type=float, default=0.05,
                    help="warn that the match is meaningless when the median recon->GT "
                         "distance exceeds this (m). The vote cutoff is relative, so a "
                         "reconstruction 3 m away still draws 95%% of the vote (measured "
                         "on obj21). Use it as a warning, never as a decision")
    ap.add_argument("--match_min_share", type=float, default=0.10,
                    help="minimum vote share for a label to be accepted during automatic "
                         "matching")
    ap.add_argument("--match_min_cover", type=float, default=0.30,
                    help="to accept a non-dominant label, at least this share of that GT "
                         "instance's surface must lie within --match_max_dist of the "
                         "reconstruction. A bbox gap alone cannot reject a TOUCHING "
                         "neighbour such as a cushion on a sofa -- measured 0929 on room0, "
                         "every non-dominant label was accepted at a 0mm gap, GT then "
                         "included the neighbour, and unseen completion grew with the "
                         "number of unioned labels to 25/790/1040/2634mm")
    ap.add_argument("--match_max_gap", type=float, default=0.15,
                    help="maximum bbox gap (m) for accepting a non-dominant label. Vote "
                         "share alone cannot separate 'GT split across several ids' from "
                         "'our instance leaked onto the neighbour'. The former sits on the "
                         "reconstruction; the latter sits away from it")
    ap.add_argument("--recon", required=True, help="side A (usually fuse_post.ply)")
    ap.add_argument("--recon2", default="", help="side B (usually fused_prior.ply)")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--gid", default="",
                    help="object id. Needed only when masks are used "
                         "(--masks_root / --use_mask). Leave unset for scene-level "
                         "evaluation")
    ap.add_argument("--masks_root", default="")
    ap.add_argument("--stems", default="")
    ap.add_argument("--vis_source", default="gt_mesh", choices=["gt_mesh", "gt_depth"],
                    help="visibility oracle. gt_mesh = ray cast the GT scene mesh "
                         "(recommended, needs no depth files) / gt_depth = depth map files")
    ap.add_argument("--gt_scene_mesh", default="",
                    help="[gt_mesh] the WHOLE-scene GT mesh, for occlusion. Not needed when "
                         "--gt_mesh is already the scene mesh (mesh_semantic.ply)")
    ap.add_argument("--gt_depth_dir", default="", help="[gt_depth] depth map directory")
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--n_views", type=int, default=120,
                    help="number of views to use (uniform subsample, 0 = all)")
    ap.add_argument("--ds", type=int, default=2, help="depth / mask downscale factor")
    ap.add_argument("--n_sample", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0,
                    help="mesh sampling seed. Settings must be compared under the same "
                         "seed. Re-evaluating one mesh with seeds 0..4 reveals this "
                         "evaluation's own noise band (do not interpret differences "
                         "smaller than that)")
    ap.add_argument("--margin", type=float, default=0.015,
                    help="visibility tolerance (m) -- headroom for GT depth noise and "
                         "discretisation")
    ap.add_argument("--min_views", type=int, default=1,
                    help="minimum views for a seen verdict (1 = observed if any view saw it)")
    ap.add_argument("--use_mask", action="store_true",
                    help="include the object mask in the visibility test (keeps a "
                         "neighbouring object's depth out)")
    ap.add_argument("--thresholds", default="0.005,0.01,0.02")
    ap.add_argument("--csv", default="", help="path to append sweep-comparison rows to")
    ap.add_argument("--tag", default="", help="tag for the CSV row (e.g. d0.010)")
    ap.add_argument("--csv_all", action="store_true",
                    help="also write the A row to the CSV (B only by default)")
    # Completion-side metrics. Surface distances cannot separate "filled the hole"
    # from "left it open", nor "solid" from "inflated shell".
    ap.add_argument("--iou_voxel", type=float, default=0.01,
                    help="volumetric IoU voxel (m). Coarser than the fusion's 0.005 on "
                         "purpose: this measures gross volume, and halving it costs 8x "
                         "the queries for the same verdict")
    ap.add_argument("--preserve_thr", type=float, default=0.01,
                    help="A->B preservation radius (m). Share of the observed surface "
                         "that survives fusion -- the direct form of the claim that "
                         "seen_acc only supports indirectly")
    ap.add_argument("--no_extra_metrics", action="store_true",
                    help="skip watertightness / volumetric IoU / preservation")
    args = ap.parse_args()

    thr = [float(x) for x in args.thresholds.split(",")]
    if args.vis_source == "gt_depth" and not args.gt_depth_dir:
        ap.error("--vis_source gt_depth requires --gt_depth_dir")
    # Scene-level evaluation has no gid. Masks require one, so refuse explicitly rather
    # than silently running without masks.
    if args.use_mask and not args.gid:
        ap.error("--use_mask requires --gid -- drop --use_mask for scene-level evaluation")
    if args.gt_labels.strip().lower() == "all" and args.gid:
        print("[warn] --gt_labels all was given together with --gid -- gid is ignored")

    # --- load GT and extract the target object ---
    V, T, L = load_mesh_labeled(args.gt_mesh)
    Vs, Ts = (V, T)
    if args.gt_scene_mesh:                            # a separate scene mesh was supplied
        Vs, Ts, _ = load_mesh_labeled(args.gt_scene_mesh)
    if L is not None and args.gt_labels.strip().lower() == "all":
        # [scene evaluation] Use the whole GT with no label selection. This path exists
        # for cases where per-object evaluation does not hold (a SAM3 instance spanning
        # several GT objects, or a generative prior completing an object beyond the
        # instance boundary) -- at scene level that geometry is not an error.
        print(f"[GT] using the whole scene: tri {len(T)}  (no label selection)")
    elif L is not None:
        if args.gt_labels:
            labs = [int(x) for x in args.gt_labels.split(",")]
        else:
            ref, _ = sample(args.recon, min(args.n_sample, 100000), args.seed)
            labs = auto_match_labels(V, T, L, ref, args.match_min_share,
                                     max_dist=args.match_max_dist,
                                     max_gap=args.match_max_gap,
                                     min_cover=args.match_min_cover)
        sel = np.isin(L, labs)
        assert sel.any(), f"no face has object_id={labs}"
        T = T[sel]
        print(f"[GT] extracted object_id={labs}: tri {int(sel.sum())}")
    elif not args.gt_scene_mesh:
        print("  ! no object_id -- treating --gt_mesh as an object mesh. Supply "
              "--gt_scene_mesh so occlusion by other objects is accounted for")

    scene_mesh = None
    if args.vis_source == "gt_mesh":
        scene_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(Vs),
            o3d.utility.Vector3iVector(Ts.astype(np.int32)))
    views = build_views(args, scene_mesh)

    G, gidx = sample_tris(V, T, args.n_sample)
    # [NC] Face normals for the GT points -- Chamfer cannot catch a surface that is in
    # the right place but bumpy.
    _e1 = V[T[gidx, 1]] - V[T[gidx, 0]]; _e2 = V[T[gidx, 2]] - V[T[gidx, 0]]
    GN = np.cross(_e1, _e2)
    GN /= np.maximum(np.linalg.norm(GN, axis=1, keepdims=True), 1e-12)
    gs, _ = classify(G, views, args.margin, args.min_views, args.use_mask)
    print(f"[GT] {len(G)} points -- seen {gs.mean()*100:.1f}% / "
          f"unseen {(~gs).mean()*100:.1f}%")
    if gs.mean() > 0.98:
        print("  ! almost nothing is unseen -- margin may be too large, or check the "
              "view / depth correspondence")

    rows = []
    gt_VT = (V, T)
    SA = sample(args.recon, args.n_sample, args.seed)
    a = report("A: " + args.recon, SA, G, gs, None, thr, views, args, GN,
               mesh_path=args.recon, gt_VT=gt_VT)
    rows.append(a)
    if args.recon2:
        SB = sample(args.recon2, args.n_sample, args.seed)
        b = report("B: " + args.recon2, SB, G, gs, None, thr, views, args, GN,
                   mesh_path=args.recon2, gt_VT=gt_VT)
        # A -> B, one-directional: what B adds is the contribution, not an error
        if not args.no_extra_metrics:
            b["preservation"] = preservation(SA[0], SB[0], args.preserve_thr)
            a["preservation"] = 1.0
        rows.append(b)
        print("\n===== A -> B change (desired: seen acc held, unseen comp down) =====")
        print(f"  seen accuracy      {a['seen_acc']:7.2f} -> {b['seen_acc']:7.2f} mm  "
              f"({b['seen_acc']-a['seen_acc']:+.2f}; closer to 0 = observation preserved)")
        print(f"  unseen completion  {a['unseen_comp']:7.2f} -> {b['unseen_comp']:7.2f} mm  "
              f"({b['unseen_comp']-a['unseen_comp']:+.2f}; negative = the prior contributed)")
        print(f"  unseen F@2cm       {a['unseen_F2.0']:7.4f} -> {b['unseen_F2.0']:7.4f}  "
              f"(P {a['unseen_P2.0']:.3f}->{b['unseen_P2.0']:.3f}, "
              f"R {a['unseen_R2.0']:.3f}->{b['unseen_R2.0']:.3f})")
        print(f"  free violation     {a['free_pct']:6.2f}% -> {b['free_pct']:6.2f}%  "
              f"({b['free_pct']-a['free_pct']:+.2f}%p, lower is better)")
        if not args.no_extra_metrics:
            print(f"  observation kept   {b.get('preservation', float('nan')):7.4f}        "
                  f"(share of A's surface still within {args.preserve_thr*100:.0f}cm; "
                  f"1 = untouched)")
            if "open_edges" in a and "open_edges" in b:
                print(f"  open boundary edges{a['open_edges']:7d} -> {b['open_edges']:7d}  "
                      f"(falls as the mesh closes)")
            if a.get("vol_iou", float("nan")) == a.get("vol_iou", float("nan")):
                print(f"  volumetric IoU     {a['vol_iou']:7.4f} -> {b['vol_iou']:7.4f}  "
                      f"(by volume; catches inflation and hollowness)")
            print(f"  generated region   recon unseen {a['recon_unseen_pct']:.1f}% -> "
                  f"{b['recon_unseen_pct']:.1f}%   |  GT unseen {100-a['gt_seen_pct']:.1f}%")
    if args.csv:
        write_csv(args.csv, rows if args.csv_all else rows[-1:], args.tag or "")


if __name__ == "__main__":
    main()