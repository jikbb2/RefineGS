#!/usr/bin/env python3
"""Make the per-object folders trainable, right after prepare_folder. Idempotent.

For each data/<scene>/masks/<gid>/:
  (1)  rebuild images/ as REAL COPIES of only the frames that have a mask. The source is
       always the scene-level data/<scene>/images (symlinks into nice-slam, or real files).
       Required, because filterPLY's threshold (len(images)/2) and the training views have
       to agree with the number of masked frames.
  (1b) copy the GT depth to depths/<frame-stem>.png, for depth supervision. nice-slam names
       them frameNNN.jpg <-> depthNNN.png.
  (2)  delete sparse/0/points3D.ply (and any older points3d.ply) so the loader builds the
       OBJECT's init through filterPLY instead of reusing the scene's.

Assumes RGBA masks with the object in alpha, and the cameras.py / dataset_readers.py
patches. Reading from the scene level rather than from the per-object folder is what makes
re-running this safe.

Run: python setup_instance_folders.py <scene>
"""
import os, sys, glob, shutil


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "replica_room0_v2"
    root = f"data/{scene}/masks"
    scene_img = os.path.realpath(f"data/{scene}/images")   # the source (frame*.jpg + depth*.png)
    if not os.path.isdir(scene_img):
        print(f"[ERROR] no scene image source: data/{scene}/images (broken symlink?)"); sys.exit(1)

    gdirs = sorted(d for d in glob.glob(root + "/*/") if os.path.isdir(d))
    n_obj = 0
    for gd in gdirs:
        masks = glob.glob(os.path.join(gd, "masks", "*.png"))
        if not masks:
            continue
        stems = sorted(os.path.splitext(os.path.basename(m))[0] for m in masks)

        # (1) rebuild images/ -- real copies of the masked frames only, from the scene source
        imgdir = os.path.join(gd, "images")
        if os.path.islink(imgdir):
            os.unlink(imgdir)
        elif os.path.isdir(imgdir):
            shutil.rmtree(imgdir)
        os.makedirs(imgdir)
        # (1b) rebuild depths/
        depthdir = os.path.join(gd, "depths")
        shutil.rmtree(depthdir, ignore_errors=True); os.makedirs(depthdir)

        copied = dcopied = 0
        for st in stems:
            isrc = os.path.join(scene_img, st + ".jpg")
            if os.path.exists(isrc):
                shutil.copy(os.path.realpath(isrc), os.path.join(imgdir, st + ".jpg")); copied += 1
            dsrc = os.path.join(scene_img, st.replace("frame", "depth") + ".png")  # nice-slam naming
            if os.path.exists(dsrc):
                shutil.copy(os.path.realpath(dsrc), os.path.join(depthdir, st + ".png")); dcopied += 1

        # (2) drop the scene ply and any earlier object init, so filterPLY runs again
        for p in (os.path.join(gd, "sparse", "0", "points3D.ply"), os.path.join(gd, "points3d.ply")):
            if os.path.exists(p):
                os.remove(p)

        gid = os.path.basename(gd.rstrip("/"))
        print(f"  obj {gid:>3}: masks={len(stems)} images={copied} depths={dcopied}")
        n_obj += 1
    print(f"setup done: {n_obj} objects ({scene})  [scene_img={scene_img}]")


if __name__ == "__main__":
    main()