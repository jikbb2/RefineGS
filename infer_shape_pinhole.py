#!/usr/bin/env python3
"""ShapeR inference wrapper with pinhole-camera support (a non-invasive monkeypatch).

The problem: ShapeR's `preprocessing/helper.rectify_images()` hardcodes
      `type_str="Fisheye624"`. That step unwarps an Aria fisheye capture into a pinhole
      image, so feeding it COLMAP images that are ALREADY pinhole inverts an arctan
      mapping that was never applied, and distorts them.
The fix: when the pkl carries `pinhole: True`, replace rectify with an identity pass and
      expand `camera_params` (a 3x3 K) to 4x4, used as it is.
      There is precedent for this in the upstream code: `get_image_data_dav3_workaround`
      likewise skips rectify and uses `convert_to_4x4(camera_params)` directly.

Imported by shaper_field.py, which applies the patch by importing this module. It can also
be run on its own, from the ShapeR checkout (it hands off to infer_shape.py in the current
directory):

  python infer_shape_pinhole.py --input_pkl refinegs_obj1.pkl --config balance \
      --do_transform_to_world --output_dir output

(arguments are passed through to infer_shape.py unchanged)
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
    """A 3x3 K (or something already 4x4) -> 4x4."""
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
    """Identity rectify for pinhole input. Returns the same (images, masks, 4x4 params)
    shapes as the original."""
    imgs = images.numpy() if torch.is_tensor(images) else np.asarray(images)
    msks = masks.numpy() if torch.is_tensor(masks) else np.asarray(masks)
    if msks.ndim == 4 and msks.shape[-1] == 1:         # (N,H,W,1) -> (N,H,W)
        msks = msks[..., 0]
    cps = camera_params.numpy() if torch.is_tensor(camera_params) else np.asarray(camera_params)
    return imgs.astype(np.uint8), msks.astype(np.uint8), _to_4x4(cps)


_orig_rectify = ip.rectify_images
_orig_get = ip.get_image_data_based_on_strategy


def get_image_data_patched(pkl_sample, num_views, scale, is_rgb, strategy="cluster"):
    """When the pkl carries the pinhole flag, swap rectify for the identity and call the
    original function."""
    if pkl_sample.get("pinhole", False):
        ip.rectify_images = rectify_passthrough
        try:
            return _orig_get(pkl_sample, num_views, scale, is_rgb, strategy)
        finally:
            ip.rectify_images = _orig_rectify
    return _orig_get(pkl_sample, num_views, scale, is_rgb, strategy)


ip.get_image_data_based_on_strategy = get_image_data_patched
# shaper_dataset may already hold its own from-import binding, so replace that one too.
try:
    import dataset.shaper_dataset as sd
    sd.get_image_data_based_on_strategy = get_image_data_patched
except Exception as e:                                  # pragma: no cover
    print(f"[patch] could not replace the shaper_dataset binding ({e}) -- check import order")

print("[patch] pinhole rectify bypass active (applies only to samples with pinhole=True)")

if __name__ == "__main__":
    sys.argv[0] = "infer_shape.py"
    runpy.run_path("infer_shape.py", run_name="__main__")
