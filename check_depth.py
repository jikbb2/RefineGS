#!/usr/bin/env python3
"""Rendered depth vs GT depth, for one or more trained scene models.

The cheapest way to tell whether a training change helped: it needs no fusion, no
extraction and no evaluation, just a few renders. The same quantity appears as [gt-check]
in the fusion logs, where it is only measured over one object's mask; here it covers the
whole frame.

Measured so far (Replica room0, -r 2, over the fusion logs):
  lambda_dist 0,   lambda_normal 0      median 3.1mm  p90  5.7mm
  lambda_dist 300, lambda_normal 0.05   median 6.2mm  p90 14.4mm

  python check_depth.py -m OUT/scene OUT/scene_n --gt_depth_dir GTD --iteration 30000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def stats(model, args):
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import render
    from scene import Scene, GaussianModel
    from sdf_distill_depth import load_gt_depth

    p = ArgumentParser()
    mp = ModelParams(p, sentinel=True)
    pp = PipelineParams(p)
    p.add_argument("--iteration", default=args.iteration, type=int)
    argv, sys.argv = sys.argv, ["x", "-m", model]
    a = get_combined_args(p)
    sys.argv = argv
    a.depth_ratio = args.depth_ratio                      # match render.py's mesh call
    a.data_device = "cpu"

    g = GaussianModel(mp.extract(a).sh_degree)
    sc = Scene(mp.extract(a), g, load_iteration=args.iteration, shuffle=False)
    bg = torch.zeros(3, device="cuda")
    cams = sc.getTrainCameras()
    step = max(1, len(cams) // args.n_views)

    err, nfin, npx = [], 0, 0
    pipe = pp.extract(a)
    with torch.no_grad():                      # render() returns a tensor with a graph
        for c in cams[::step][:args.n_views]:
            d = render(c, g, pipe, bg)["surf_depth"][0]
            fin = torch.isfinite(d)
            nfin += int((~fin).sum()); npx += d.numel()
            dg = load_gt_depth(args.gt_depth_dir, c.image_name,
                               d.shape[0], d.shape[1], args.gt_depth_scale)
            if dg is None:
                continue
            dg = torch.as_tensor(np.asarray(dg), device=d.device, dtype=d.dtype)
            if dg.shape != d.shape:
                dg = torch.nn.functional.interpolate(
                    dg[None, None], size=d.shape, mode="nearest")[0, 0]
            ok = fin & (dg > 0.01) & (d > 0.01)
            if ok.any():
                err.append((d[ok] - dg[ok]).abs().cpu().numpy())
    if not err:
        return None
    e = np.concatenate(err)
    return dict(n=len(e), med=np.median(e) * 1000,
                p90=np.percentile(e, 90) * 1000, p99=np.percentile(e, 99) * 1000,
                mean=e.mean() * 1000, nonfinite=nfin / max(npx, 1) * 100,
                gauss=int(g.get_xyz.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", nargs="+", required=True)
    ap.add_argument("--gt_depth_dir", required=True)
    ap.add_argument("--gt_depth_scale", type=float, default=6553.5)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--depth_ratio", type=float, default=1.0)
    ap.add_argument("--n_views", type=int, default=40)
    args = ap.parse_args()

    res = []
    for m in args.models:
        res.append((m, stats(os.path.expanduser(m), args)))
    print(f"\n{'model':<34}{'gauss':>10}{'med':>8}{'p90':>8}{'p99':>8}{'mean':>8}  nonfin%")
    for m, s in res:
        nm = os.path.basename(os.path.normpath(m))
        if s is None:
            print(f"{nm:<34}  no GT depth matched -- check --gt_depth_dir")
            continue
        print(f"{nm:<34}{s['gauss']:>10,}{s['med']:>8.1f}{s['p90']:>8.1f}"
              f"{s['p99']:>8.1f}{s['mean']:>8.1f}{s['nonfinite']:>9.3f}")
    print("\nmm, over pixels where both depths are valid. Lower is better; a heavy p90/p99")
    print("tail means large errors where a ray legitimately crosses two surfaces (edges).")


if __name__ == "__main__":
    main()