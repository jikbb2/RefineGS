#!/usr/bin/env python3
"""
Amodal3R Generation PoC — RefineGS Axis 3

Runs Amodal3R on 3 representative instances (chair/table/sofa) with K seeds,
saves per-seed mesh PLY files (for later registration + fusion), and exports
multi-view renders for visual inspection.

Usage (run from ~/Amodal3R with conda activate amodal3r):
    python run_amodal3r_poc.py
    python run_amodal3r_poc.py --labels 97 98 75 --seeds 1 2 3 --n_views 3
    python run_amodal3r_poc.py --labels 97 --seeds 1 2 3 4 5 --n_views 1

Output:
    ./poc_output/<label>/
        seed_<k>/
            mesh.ply           ← trimesh PLY, canonical coords
            mesh.glb           ← optional, for blender preview
            multiview_gs/      ← 8 Gaussian splat renders
            multiview_mesh/    ← 8 mesh normal renders
        best_view_rgb.png      ← input image (copy)
        best_view_mask.png     ← input mask (copy)
        meta.json              ← run info

Notes:
  - Input is loaded from ./input/<label>/ (prepared by prepare_amodal3r_inputs.py)
  - n_views controls how many of the top-k views to feed (multi-image mode)
  - Canonical coords: Amodal3R outputs in its own canonical frame.
    Registration to world coords is a separate step (A-1 DINOv2 + ICP).
"""
import os
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh
from PIL import Image

from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils


# ── helpers ──────────────────────────────────────────────────────────────────

def save_mesh_ply(mesh_result, path: Path):
    """Save Amodal3R mesh result to PLY via trimesh."""
    verts = mesh_result.vertices
    faces = mesh_result.faces
    if hasattr(verts, 'cpu'):
        verts = verts.cpu().numpy()
    if hasattr(faces, 'cpu'):
        faces = faces.cpu().numpy()
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs
        if hasattr(attrs, 'cpu'):
            attrs = attrs.cpu().numpy()
        m.visual.vertex_colors = attrs
    m.export(str(path))


def save_multiview(result, out_dir: Path, prefix: str, mode: str = 'gs', nviews: int = 8):
    """Render nviews and save as PNG images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == 'gs':
        mv, _, _ = render_utils.render_multiview(result, nviews=nviews, bg_color=(1, 1, 1))
        frames = mv['color']
    else:
        mv, _, _ = render_utils.render_multiview(result, nviews=nviews, bg_color=(1, 1, 1))
        frames = mv['normal']
    for i, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{prefix}_{i:03d}.png"), bgr)


def load_views(input_dir: Path, n_views: int):
    """Load up to n_views RGB + mask pairs from input dir."""
    images, masks = [], []
    for rank in range(n_views):
        rgb_p = input_dir / f"rgb_{rank}.png"
        mask_p = input_dir / f"mask_{rank}.png"
        if not rgb_p.exists() or not mask_p.exists():
            break
        images.append(Image.open(rgb_p).convert("RGB"))
        masks.append(Image.open(mask_p).convert("L"))
    return images, masks


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["97", "98", "75"],
                    help="Instance labels (must match input/<label>/ dirs)")
    ap.add_argument("--input_dir", default="./input",
                    help="Root input dir (from prepare_amodal3r_inputs.py)")
    ap.add_argument("--out_dir", default="./poc_output",
                    help="Root output dir")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                    help="Random seeds (one generation per seed)")
    ap.add_argument("--n_views", type=int, default=3,
                    help="Number of input views to feed (multi-image mode)")
    ap.add_argument("--sparse_steps", type=int, default=12)
    ap.add_argument("--sparse_cfg", type=float, default=7.5)
    ap.add_argument("--slat_steps", type=int, default=12)
    ap.add_argument("--slat_cfg", type=float, default=3.0)
    ap.add_argument("--skip_glb", action="store_true",
                    help="Skip GLB export (faster, saves disk)")
    args = ap.parse_args()

    input_root = Path(args.input_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── load pipeline once ──────────────────────────────────────────────────
    print("Loading Amodal3R pipeline …")
    t0 = time.time()
    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.cuda()
    print(f"  pipeline ready in {time.time()-t0:.1f}s\n")

    summary = []

    for label in args.labels:
        inst_in = input_root / str(label)
        inst_out = out_root / str(label)
        inst_out.mkdir(parents=True, exist_ok=True)

        # load views
        images, masks = load_views(inst_in, args.n_views)
        if not images:
            print(f"[SKIP] {label}: no input views found in {inst_in}")
            continue
        print(f"[{label}] {len(images)} view(s) loaded")

        # copy best view for reference
        shutil.copy(inst_in / "rgb_0.png",  inst_out / "best_view_rgb.png")
        shutil.copy(inst_in / "mask_0.png", inst_out / "best_view_mask.png")

        label_summary = {"label": label, "n_views": len(images), "seeds": []}

        for seed in args.seeds:
            seed_out = inst_out / f"seed_{seed}"
            mesh_ply = seed_out / "mesh.ply"

            if mesh_ply.exists():
                print(f"  seed {seed}: already exists, skipping")
                label_summary["seeds"].append({"seed": seed, "status": "skipped"})
                continue

            seed_out.mkdir(parents=True, exist_ok=True)
            print(f"  seed {seed}: generating …", end="", flush=True)
            t1 = time.time()

            try:
                outputs = pipeline.run_multi_image(
                    images,
                    masks,
                    seed=seed,
                    sparse_structure_sampler_params={
                        "steps": args.sparse_steps,
                        "cfg_strength": args.sparse_cfg,
                    },
                    slat_sampler_params={
                        "steps": args.slat_steps,
                        "cfg_strength": args.slat_cfg,
                    },
                )
            except Exception as e:
                print(f" FAILED: {e}")
                label_summary["seeds"].append({"seed": seed, "status": "failed", "error": str(e)})
                continue

            elapsed = time.time() - t1
            print(f" done ({elapsed:.1f}s)")

            # save mesh PLY  ← main output for registration PoC
            save_mesh_ply(outputs['mesh'][0], mesh_ply)
            print(f"    mesh → {mesh_ply}")

            # save multiview renders for visual inspection
            save_multiview(outputs['gaussian'][0], seed_out / "multiview_gs",
                           prefix="gs", mode='gs', nviews=8)
            save_multiview(outputs['mesh'][0], seed_out / "multiview_mesh",
                           prefix="mesh", mode='mesh', nviews=8)

            # save preview GIF
            vid_gs   = render_utils.render_video(outputs['gaussian'][0], bg_color=(1,1,1))['color']
            vid_mesh = render_utils.render_video(outputs['mesh'][0],     bg_color=(1,1,1))['normal']
            video = [np.concatenate([a, b], axis=1) for a, b in zip(vid_gs, vid_mesh)]
            imageio.mimsave(str(seed_out / "preview.gif"), video, fps=30)

            # optional GLB
            if not args.skip_glb:
                try:
                    glb = postprocessing_utils.to_glb(
                        outputs['gaussian'][0], outputs['mesh'][0],
                        simplify=0.95, texture_size=512, verbose=False)
                    glb.export(str(seed_out / "mesh.glb"))
                except Exception as e:
                    print(f"    [warn] GLB export failed: {e}")

            label_summary["seeds"].append({
                "seed": seed, "status": "ok", "elapsed_s": round(elapsed, 1),
                "mesh_ply": str(mesh_ply),
            })

        # write meta
        (inst_out / "meta.json").write_text(json.dumps(label_summary, indent=2))
        summary.append(label_summary)

    # ── final summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PoC SUMMARY")
    print("=" * 60)
    for s in summary:
        ok = [x for x in s["seeds"] if x.get("status") == "ok"]
        skip = [x for x in s["seeds"] if x.get("status") == "skipped"]
        fail = [x for x in s["seeds"] if x.get("status") == "failed"]
        times = [x["elapsed_s"] for x in ok]
        avg_t = sum(times)/len(times) if times else 0
        print(f"  label {s['label']}: {len(ok)} generated, {len(skip)} skipped, "
              f"{len(fail)} failed | avg {avg_t:.1f}s/seed")
        for x in ok:
            print(f"    seed {x['seed']} → {x['mesh_ply']}")

    print(f"\nMeshes saved to: {out_root}")
    print("Next step: registration PoC (A-1) — DINOv2 feature lifting + ICP")
    print("  Input:  poc_output/<label>/seed_<k>/mesh.ply  (canonical)")
    print("  Target: /home/elicer/RefineGS/output/replica_room0/axis3_sweep/reg_strong/<label>/train/ours_7000/fuse_post.ply")


if __name__ == "__main__":
    main()
