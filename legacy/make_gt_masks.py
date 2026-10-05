#!/usr/bin/env python3
"""Render per-view instance masks (and depth) by raycasting the GT semantic mesh.

Preparing a new room otherwise needs a SAM3 pass, which is the step this project already
found unreliable: on room0, 8 of 31 objects were dropped because their instance label was
wrong or spanned two GT objects. Raycasting the GT mesh gives exact masks in minutes, so
several rooms become available at once.

This is a CONTROL, not the main pipeline. Use SAM3 masks for the headline table and these
for experiments where instance quality must be held fixed -- learning the fusion weights,
for instance, is independent of where the instance boundaries came from.

Writes, in the layout the rest of the pipeline expects:
  <data>/masks/<oid>/masks/<stem>.png     binary 0/255 per object
  <data>/labels_scene/labels/<stem>.png   uint16, 0 = background, 1..K instances
  <data>/labels_scene/union/<stem>.png    foreground union
  <data>/labels_scene/id_map.json
  <data>/images/depth<NNNNNN>.png         GT depth, uint16 / gt_depth_scale  (--write_depth)

  python make_gt_masks.py --gt_mesh ~/room_0/habitat/mesh_semantic.ply \
      --gt_info ~/room_0/habitat/info_semantic.json \
      --colmap DATA/sparse_dense/0 --out DATA --stride 1
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warp_gt_to_pose import read_colmap                        # noqa: E402

IGNORE = 65535
DEFAULT_EXCLUDE = ("wall,floor,ceiling,door,window,blind,vent,switch,thermostat,"
                   "rug,stair,beam,panel,pillar,wall-plug,outlet")


def tok(name):
    import re
    return set(t for t in re.split(r"[^a-z0-9]+", name.lower()) if t) | {name.lower()}


def excluded(name, terms):
    tk = tok(name)
    for t in terms:
        t = t.strip().lower()
        if not t:
            continue
        if " " in t or "-" in t:
            if t.replace("-", " ") in name.lower().replace("-", " "):
                return True
        elif t in tk or t + "s" in tk or (t.endswith("s") and t[:-1] in tk):
            return True
    return False


def load_class_names(path):
    if not path or not os.path.isfile(os.path.expanduser(path)):
        return {}
    m = json.load(open(os.path.expanduser(path)))
    by = {int(c["id"]): str(c.get("name", "")).lower()
          for c in m.get("classes", []) if isinstance(c, dict) and "id" in c}
    out = {}
    for o in m.get("objects", []):
        if isinstance(o, dict) and "id" in o:
            out[int(o["id"])] = str(o.get("class_name")
                                    or by.get(int(o.get("class_id", -1)), "")).lower()
    if not out and "id_to_label" in m:
        out = {i: by.get(int(c), "") for i, c in enumerate(m["id_to_label"])}
    return out


def load_semantic(path):
    p = PlyData.read(os.path.expanduser(path))
    V = np.stack([p["vertex"][k] for k in ("x", "y", "z")], 1).astype(np.float32)
    fe = p["face"]
    assert "object_id" in fe.data.dtype.names, "the GT mesh has no object_id"
    T, L = [], []
    for f, o in zip(fe["vertex_indices"], fe["object_id"]):
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1])); L.append(o)
    return V, np.asarray(T, np.uint32), np.asarray(L, np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gt_info", default="")
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--out", required=True, help="the room's data dir")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth view")
    # A total-pixel threshold depends on how many views were cast, so --stride silently
    # changes which objects survive. Both criteria below are per-view and stride-free.
    ap.add_argument("--min_px_per_view", type=float, default=50.0,
                    help="mean pixels per cast view an object needs")
    ap.add_argument("--min_views", type=int, default=20,
                    help="views in which the object covers at least 20 px")
    ap.add_argument("--min_mask_px", type=int, default=20,
                    help="do not write a per-view mask below this many pixels")
    ap.add_argument("--drop_unnamed", action="store_true",
                    help="also drop GT objects whose class name is empty ('undefined'). "
                         "They are real geometry, so they are kept by default")
    ap.add_argument("--exclude_classes", default=DEFAULT_EXCLUDE,
                    help="classes never treated as objects; 'none' keeps them")
    ap.add_argument("--write_depth", action="store_true",
                    help="also write GT depth next to the frames")
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--dry_run", action="store_true", help="count pixels, write nothing")
    args = ap.parse_args()

    V, T, L = load_semantic(args.gt_mesh)
    names = load_class_names(args.gt_info)
    terms = [] if args.exclude_classes.strip().lower() in ("none", "") else \
        args.exclude_classes.split(",")
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(T))

    cams = read_colmap(args.colmap)[::args.stride]
    print(f"[gt-mask] {len(T):,} tris, {len(np.unique(L))} object ids, {len(cams)} views")

    def cast(c):
        """object_id per pixel (-1 where the ray misses), plus hit distance."""
        K = o3d.core.Tensor([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]],
                            dtype=o3d.core.Dtype.Float64)
        E = np.eye(4); E[:3, :3] = c["R"]; E[:3, 3] = c["t"]
        r = sc.cast_rays(sc.create_rays_pinhole(K, o3d.core.Tensor(E),
                                                int(c["W"]), int(c["H"])))
        pid = r["primitive_ids"].numpy()
        ok = pid != o3d.t.geometry.RaycastingScene.INVALID_ID
        return np.where(ok, L[np.where(ok, pid, 0)], -1), r["t_hit"].numpy(), ok

    # Pass 1 counts pixels; pass 2 writes. The results are NOT cached between them: at
    # stride 1 that is 2000 views x ~1.8MB = 3.7GB. Casting twice is cheaper than that.
    px, nv = {}, {}
    for i, c in enumerate(cams):
        oid, _, ok = cast(c)
        u, n = np.unique(oid[ok], return_counts=True)
        for a, b in zip(u.tolist(), n.tolist()):
            px[a] = px.get(a, 0) + b
            if b >= 20:
                nv[a] = nv.get(a, 0) + 1
        if (i + 1) % 200 == 0:
            print(f"  cast {i + 1}/{len(cams)}")

    nc = max(len(cams), 1)
    keep, dropped = [], []
    for o in sorted(px):
        o = int(o)
        nm = names.get(o, "")
        ppv, views = px[o] / nc, nv.get(o, 0)
        why = ""
        if terms and nm and excluded(nm, terms):
            why = "class"
        elif args.drop_unnamed and not nm:
            why = "unnamed"
        elif ppv < args.min_px_per_view:
            why = "too small"
        elif views < args.min_views:
            why = f"only {views} views"
        (dropped if why else keep).append((o, nm, ppv, views, why))
    print(f"\n[gt-mask] keeping {len(keep)}, dropping {len(dropped)}"
          f"   (>= {args.min_px_per_view:.0f} px/view and >= {args.min_views} views)")
    print(f"{'oid':>5}  {'class':<16}{'px/view':>9}{'views':>7}")
    for o, nm, ppv, views, _ in sorted(keep, key=lambda x: -x[2]):
        print(f"{o:>5}  {nm or 'undefined':<16}{ppv:>9.0f}{views:>7}"
              + ("   <- no class name" if not nm else ""))
    if dropped:
        print("\ndropped:")
        for o, nm, ppv, views, why in sorted(dropped, key=lambda x: -x[2])[:20]:
            print(f"{o:>5}  {nm or 'undefined':<16}{ppv:>9.0f}{views:>7}   {why}")
    keep = sorted(o for o, *_ in keep)
    label_of = {o: i + 1 for i, o in enumerate(keep)}
    if args.dry_run:
        return

    out = os.path.expanduser(args.out)
    lab_d = os.path.join(out, "labels_scene", "labels")
    uni_d = os.path.join(out, "labels_scene", "union")
    for d in (lab_d, uni_d):
        os.makedirs(d, exist_ok=True)
    for o in keep:
        os.makedirs(os.path.join(out, "masks", str(o), "masks"), exist_ok=True)

    for i, c in enumerate(cams):
        oid, t, ok = cast(c)
        lab = np.zeros(oid.shape, np.uint16)
        for o in keep:
            m = ok & (oid == o)
            n = int(m.sum())
            if n:
                lab[m] = label_of[o]
            # A mask with a handful of pixels is noise, and a missing file simply means
            # "not visible here", which require_mask already handles.
            if n >= args.min_mask_px:
                Image.fromarray((m * 255).astype(np.uint8)).save(
                    os.path.join(out, "masks", str(o), "masks", c["stem"] + ".png"))
        Image.fromarray(lab).save(os.path.join(lab_d, c["stem"] + ".png"))
        Image.fromarray(((lab > 0) * 255).astype(np.uint8)).save(
            os.path.join(uni_d, c["stem"] + ".png"))
        if args.write_depth:
            d = np.where(ok & np.isfinite(t), t, 0.0) * args.gt_depth_scale
            nm = c["stem"].replace("frame", "depth")
            Image.fromarray(np.clip(d, 0, 65535).astype(np.uint16)).save(
                os.path.join(out, "images", nm + ".png"))
        if (i + 1) % 200 == 0:
            print(f"  wrote {i + 1}/{len(cams)}")

    json.dump({"gids": keep, "label_of_gid": {str(o): label_of[o] for o in keep},
               "K": len(keep), "source": "gt_raycast", "ignore": IGNORE,
               "min_px_per_view": args.min_px_per_view, "min_views": args.min_views,
               "n_views_cast": len(cams), "stride": args.stride,
               "class_of_gid": {str(o): names.get(o, "") for o in keep}},
              open(os.path.join(out, "labels_scene", "id_map.json"), "w"), indent=1)
    print(f"[gt-mask] -> {out}/masks/<oid>/masks  and  {out}/labels_scene")


if __name__ == "__main__":
    main()