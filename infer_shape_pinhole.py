#!/usr/bin/env python3
"""ShapeR 추론 래퍼 — 핀홀 카메라 지원(비침습 몽키패치).

문제: ShapeR `preprocessing/helper.rectify_images()` 는 `type_str="Fisheye624"` 하드코딩.
      Aria 어안을 핀홀로 펴는 단계라, 이미 핀홀인 COLMAP 이미지를 넣으면
      arctan 매핑을 잘못 되돌려 왜곡시킨다.
해결: pkl 에 `pinhole: True` 가 있으면 rectify 를 항등 통과로 대체하고
      `camera_params`(3x3 K)를 4x4 로 확장해 그대로 쓴다.
      ※ 코드에 이미 선례가 있다 — `get_image_data_dav3_workaround` 도 rectify 를
        건너뛰고 `convert_to_4x4(camera_params)` 를 직접 사용한다.

ShapeR 저장소 루트에 두고 실행:
  python infer_shape_pinhole.py --input_pkl refinegs_obj1.pkl --config balance \
      --do_transform_to_world --output_dir output
(인자는 infer_shape.py 와 동일하게 전달됨)
"""
import io
import os
import pickle
import runpy
import sys

import numpy as np
import torch

import dataset.image_processor as ip


def _to_4x4(params):
    """3x3 K(또는 이미 4x4) → 4x4."""
    out = []
    for p in params:
        p = np.asarray(p, np.float32)
        m = np.eye(4, dtype=np.float32)
        if p.shape == (4, 4):
            m = p
        elif p.shape == (3, 3):
            m[:3, :3] = p
        else:                                  # [fx, fy, cx, cy]
            m[0, 0], m[1, 1] = p[0], p[1]
            m[0, 2], m[1, 2] = p[2], p[3]
        out.append(m)
    return np.stack(out)


def rectify_passthrough(images, masks, camera_params):
    """핀홀 입력용 항등 rectify. 원본과 동일한 (images, masks, 4x4 params) 반환."""
    imgs = images.numpy() if torch.is_tensor(images) else np.asarray(images)
    msks = masks.numpy() if torch.is_tensor(masks) else np.asarray(masks)
    if msks.ndim == 4 and msks.shape[-1] == 1:         # (N,H,W,1) → (N,H,W)
        msks = msks[..., 0]
    cps = camera_params.numpy() if torch.is_tensor(camera_params) else np.asarray(camera_params)
    return imgs.astype(np.uint8), msks.astype(np.uint8), _to_4x4(cps)


_orig_rectify = ip.rectify_images
_orig_get = ip.get_image_data_based_on_strategy


def get_image_data_patched(pkl_sample, num_views, scale, is_rgb, strategy="cluster"):
    """pkl 에 pinhole 플래그가 있으면 rectify 를 항등으로 바꿔 원 함수를 호출."""
    if pkl_sample.get("pinhole", False):
        ip.rectify_images = rectify_passthrough
        try:
            return _orig_get(pkl_sample, num_views, scale, is_rgb, strategy)
        finally:
            ip.rectify_images = _orig_rectify
    return _orig_get(pkl_sample, num_views, scale, is_rgb, strategy)


ip.get_image_data_based_on_strategy = get_image_data_patched
# shaper_dataset 이 from-import 로 이미 바인딩했을 수 있으므로 그쪽도 교체
try:
    import dataset.shaper_dataset as sd
    sd.get_image_data_based_on_strategy = get_image_data_patched
except Exception as e:                                  # pragma: no cover
    print(f"[patch] shaper_dataset 바인딩 교체 실패({e}) — import 순서 확인")

print("[patch] 핀홀 rectify 우회 활성 (pkl 의 pinhole=True 인 샘플에만 적용)")

if __name__ == "__main__":
    sys.argv[0] = "infer_shape.py"
    runpy.run_path("infer_shape.py", run_name="__main__")
