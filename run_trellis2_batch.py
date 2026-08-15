#!/usr/bin/env python3
"""TRELLIS-2 전 객체 배치 생성 — 모델을 1회만 로드하고 모든 객체 cutout → glb.

객체마다 run_trellis2_obj.py 를 부르면 4B 모델을 매번 재로드(수 분 낭비)하므로,
파이프라인을 한 번 로드하고 gen_input/obj*/views 를 순회한다.

  conda activate trellis2
  python run_trellis2_batch.py --ckpt /home/elicer/TRELLIS.2/TRELLIS.2-4B \
    --gen_input ~/gen_input --out_dir ~/gen_out

각 ~/gen_input/obj{gid}/views/*.png(문맥 cutout) → ~/gen_out/obj{gid}.glb.
이미 있는 glb 는 skip. (DINOv3 extract_features 패치 선행 필요)
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


def best_view(views_dir):
    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
    if not paths:
        return None
    imgs = [Image.open(p).convert("RGBA") for p in paths]
    areas = [np.asarray(im)[..., 3].astype(bool).sum() if np.asarray(im).shape[-1] == 4
             else np.asarray(im).size for im in imgs]
    return imgs[int(np.argmax(areas))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/elicer/TRELLIS.2/TRELLIS.2-4B")
    ap.add_argument("--gen_input", default=os.path.expanduser("~/gen_input"))
    ap.add_argument("--out_dir", default=os.path.expanduser("~/gen_out"))
    ap.add_argument("--pipeline_type", default="1024_cascade")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--texture_size", type=int, default=4096)
    ap.add_argument("--decimation", type=int, default=1000000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    obj_dirs = sorted(glob.glob(os.path.join(os.path.expanduser(args.gen_input), "obj*")))
    print(f"대상 객체 {len(obj_dirs)}개")

    print("파이프라인 로드(1회)...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt)
    pipeline.cuda()

    ok, skip, fail = 0, 0, 0
    for od in obj_dirs:
        gid = os.path.basename(od).replace("obj", "")
        out = os.path.join(os.path.expanduser(args.out_dir), f"obj{gid}.glb")
        if os.path.exists(out):
            print(f"[skip] obj{gid}: 이미 있음"); skip += 1; continue
        img = best_view(os.path.join(od, "views"))
        if img is None:
            print(f"[skip] obj{gid}: view 없음"); skip += 1; continue
        try:
            mesh = pipeline.run(img, seed=args.seed, pipeline_type=args.pipeline_type)[0]
            mesh.simplify(16777216)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.decimation, texture_size=args.texture_size,
                remesh=True, remesh_band=1, remesh_project=0, verbose=False)
            glb.export(out)
            print(f"[ok] obj{gid} → {out}"); ok += 1
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[fail] obj{gid}: {e}"); fail += 1
    print(f"\n생성 완료: ok {ok}, skip {skip}, fail {fail}  → {args.out_dir}")


if __name__ == "__main__":
    main()
