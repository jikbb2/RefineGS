#!/usr/bin/env python3
"""Turn <data>/masks/<oid>/masks into a per-object COLMAP dataset, as render.py expects.

The per-object pipeline gives every object its own dataset restricted to the views where
it is visible:
    masks/<oid>/{images,depths,masks}/  and  sparse/0/{cameras,images,points3D}.txt
make_gt_masks.py only writes the masks, so render.py fails with "Could not recognize scene
type". This fills in the rest by symlinking the room's frames and filtering images.txt.

The view list is taken from the mask filenames, which make_gt_masks.py already wrote only
where the object covers at least --min_mask_px pixels. Integrating views where the object
is not visible would feed empty depth into its TSDF.

  python make_object_dirs.py --data DATA_GT --images DATA/images \
      --colmap DATA/sparse_dense/0
"""
import argparse
import os
import shutil


def read_colmap_images(path):
    """[(name, line1, line2)] from a COLMAP images.txt, plus the header lines."""
    head, recs = [], []
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        head.append(lines[i]); i += 1
    while i < len(lines):
        l1 = lines[i].strip()
        if not l1:
            i += 1; continue
        l2 = lines[i + 1] if i + 1 < len(lines) else ""
        recs.append((l1.split()[-1], lines[i], l2))
        i += 2
    return head, recs


def link(src, dst):
    if os.path.lexists(dst):
        return True
    if not os.path.exists(src):
        return False
    os.symlink(os.path.abspath(src), dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="room dir holding masks/<oid>/masks")
    ap.add_argument("--images", required=True, help="the room's frames")
    ap.add_argument("--colmap", required=True, help="the room's sparse/0")
    ap.add_argument("--depths", default="", help="folder with depth*.png (default --images)")
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--min_views", type=int, default=20,
                    help="skip an object seen in fewer views; it cannot be reconstructed")
    args = ap.parse_args()

    data = os.path.expanduser(args.data)
    imgs = os.path.expanduser(args.images)
    dep = os.path.expanduser(args.depths or args.images)
    cm = os.path.expanduser(args.colmap)
    head, recs = read_colmap_images(os.path.join(cm, "images.txt"))
    by_name = {n: (a, b) for n, a, b in recs}
    print(f"[colmap] {len(recs)} images in {cm}")

    root = os.path.join(data, "masks")
    made = skipped = 0
    for oid in sorted((d for d in os.listdir(root) if d.isdigit()), key=int):
        md = os.path.join(root, oid, "masks")
        if not os.path.isdir(md):
            continue
        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(md) if f.endswith(".png"))
        if len(stems) < args.min_views:
            print(f"  [{oid}] {len(stems)} views -- skipped"); skipped += 1; continue

        od = os.path.join(root, oid)
        for sub in ("images", "depths", "sparse/0"):
            os.makedirs(os.path.join(od, sub), exist_ok=True)

        n_img = n_dep = 0
        for st in stems:
            n_img += link(os.path.join(imgs, st + args.ext),
                          os.path.join(od, "images", st + args.ext))
            dn = st.replace("frame", "depth") + ".png"
            n_dep += link(os.path.join(dep, dn), os.path.join(od, "depths", dn))

        # cameras and the 3D points are shared; only images.txt is filtered
        for f in ("cameras.txt", "points3D.txt", "points3D.ply", "points3D.bin"):
            src = os.path.join(cm, f)
            if os.path.exists(src):
                link(src, os.path.join(od, "sparse/0", f))
        keep = [by_name[st + args.ext] for st in stems if st + args.ext in by_name]
        with open(os.path.join(od, "sparse/0", "images.txt"), "w") as f:
            f.write("\n".join(head) + "\n" if head else "")
            for a, b in keep:
                f.write(a + "\n" + b + "\n")
        print(f"  [{oid}] {len(stems)} views  images {n_img}  depths {n_dep}  "
              f"poses {len(keep)}" + ("   <- POSE MISMATCH" if len(keep) < len(stems) else ""))
        made += 1

    print(f"\n[object-dirs] built {made}, skipped {skipped}  under {root}")
    if made:
        print("next: the mesh stage can now run (render.py -s masks/<oid>)")


if __name__ == "__main__":
    main()
