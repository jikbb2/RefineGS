#!/usr/bin/env python3
"""객체 masked cutout → TRELLIS-2 → GLB (fuse_generated_mesh.py 입력용).

export_object_views.py 로 뽑은 RGBA cutout(배경 투명=우리 마스크)을 조건으로 사용.
TRELLIS.2 preprocess 는 RGBA alpha 를 그대로 쓰므로 소파 등 인접 객체가 섞이지 않음.

기본: 단일 best-view 생성(공식 경로).
--views 로 폴더를 주면 면적 최대 뷰 1장 자동 선택.
--multiview 실험 플래그: get_cond 에 여러 뷰를 함께 넣어 조건화(모델이 지원할 때만).

  # 단일 뷰
  python run_trellis2_obj.py --ckpt /home/elicer/TRELLIS.2/TRELLIS.2-4B \
    --views ~/gen_input/obj6/views --out ~/gen_out/obj6.glb

Deps: trellis2, o_voxel (TRELLIS-2 env).
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


def load_cutouts(views_dir, single=True):
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(views_dir), "*.png")))
    assert paths, f"뷰 없음: {views_dir}"
    imgs = [Image.open(p).convert("RGBA") for p in paths]
    if single:
        # alpha 면적 최대 뷰
        areas = [np.asarray(im)[..., 3].astype(bool).sum() for im in imgs]
        i = int(np.argmax(areas))
        print(f"단일 조건 뷰: {os.path.basename(paths[i])}")
        return [imgs[i]]
    print(f"multi 조건 뷰 {len(imgs)}장")
    return imgs


def multiview_cond(pipeline, imgs, resolution):
    """실험: 전처리된 여러 뷰의 cond 토큰을 concat (모델 지원 시)."""
    pre = [pipeline.preprocess_image(im) for im in imgs]
    return pipeline.get_cond(pre, resolution)     # image_cond_model([img1,...])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/elicer/TRELLIS.2/TRELLIS.2-4B")
    ap.add_argument("--views", required=True, help="export_object_views 출력 폴더/views")
    ap.add_argument("--out", required=True, help="출력 .glb")
    ap.add_argument("--pipeline_type", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--multiview", action="store_true", help="실험적 multi-view 조건")
    ap.add_argument("--texture_size", type=int, default=4096)
    ap.add_argument("--decimation", type=int, default=1000000)
    args = ap.parse_args()

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt)
    pipeline.cuda()

    imgs = load_cutouts(args.views, single=not args.multiview)

    if args.multiview:
        # 커스텀 run: multi-view cond → 단일 이미지 run 과 동일한 이후 단계
        try:
            torch.manual_seed(args.seed)
            c512 = multiview_cond(pipeline, imgs, 512)
            c1024 = multiview_cond(pipeline, imgs, 1024) if args.pipeline_type != "512" else None
            coords = pipeline.sample_sparse_structure(
                c512, {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[args.pipeline_type], 1)
            shape_slat, res = pipeline.sample_shape_slat_cascade(
                c512, c1024,
                pipeline.models['shape_slat_flow_model_512'],
                pipeline.models['shape_slat_flow_model_1024'],
                512, 1024 if args.pipeline_type == "1024_cascade" else 1536,
                coords, {}, 49152)
            tex_slat = pipeline.sample_tex_slat(
                c1024, pipeline.models['tex_slat_flow_model_1024'], shape_slat, {})
            torch.cuda.empty_cache()
            mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
            print("multi-view 조건 성공")
        except Exception as e:
            print(f"[경고] multi-view 조건 실패({e}) → 단일 뷰로 fallback")
            mesh = pipeline.run(imgs[0], seed=args.seed, pipeline_type=args.pipeline_type)[0]
    else:
        mesh = pipeline.run(imgs[0], seed=args.seed, pipeline_type=args.pipeline_type)[0]

    mesh.simplify(16777216)
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation, texture_size=args.texture_size,
        remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    glb.export(os.path.expanduser(args.out), extension_webp=True)
    print(f"→ GLB 저장: {args.out}  (aabb 단위정규화, Y-up)")


if __name__ == "__main__":
    main()
