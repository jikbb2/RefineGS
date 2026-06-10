#
# RefineGS - smoke_test.py
# ---------------------------------------------------------------------------
# 머지된 스택(2DGS base + S&S graft)을 per-object 1개로 빠르게 점검.
# 품질 측정 X — "크래시 없이 도는가 + 출력이 sane한가"만 확인.
#
# 실행(서버, GPU 필요):
#   python smoke_test.py -s data/<scene>/masks/<id> -m output/<scene>/smoke/<id> \
#          --is_instance --disable_viewer
#
# 점검 항목:
#   [1] GaussianModel: 2D scaling(=2), id(=3)/desc(=384) 차원
#   [2] render(): render/mask/depth/rend_normal/rend_dist/surf_normal 키·shape·NaN
#   [3] 손실 5종이 유한값인지
#   [4] backward + densify stat + optimizer step 1회
#   [5] save_ply -> load_ply 왕복 (id/desc/2-scale 보존)
# ---------------------------------------------------------------------------

import os
import sys
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


def check_tensor(name, v):
    if torch.is_tensor(v):
        nan = bool(torch.isnan(v).any().item())
        print(f"    {name:18} shape={tuple(v.shape)} min={v.min().item():.4f} "
              f"max={v.max().item():.4f} nan={nan}")
        assert not nan, f"{name} contains NaN!"
    else:
        print(f"    {name:18} = {v}")


if __name__ == "__main__":
    parser = ArgumentParser("RefineGS smoke test")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    args = parser.parse_args(sys.argv[1:])
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    print("[1] build GaussianModel + Scene ...")
    gaussians = GaussianModel(dataset.sh_degree, 0, opt.optimizer_type)   # per-object: active_sh_degree=0
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    n0 = gaussians.get_xyz.shape[0]
    print(f"    #gaussians init      = {n0}")
    print(f"    scaling dim (exp 2)  = {gaussians._scaling.shape[1]}")
    print(f"    id dim (exp 3)       = {gaussians._id.shape[1]}")
    print(f"    desc dim (exp 384)   = {gaussians._desc_test.shape[1]}")
    assert gaussians._scaling.shape[1] == 2, "scaling 이 2D(surfel)가 아님!"
    assert gaussians._id.shape[1] == 3, "id 차원 != 3"

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    cam = scene.getTrainCameras()[0]
    print(f"    test camera          = {cam.image_name}  ({cam.image_width}x{cam.image_height})")

    print("[2] forward render ...")
    pkg = render(cam, gaussians, pipe, bg)
    for k in ["render", "mask", "depth", "rend_alpha", "rend_normal",
              "rend_dist", "surf_normal", "radii"]:
        check_tensor(k, pkg[k])
    assert pkg["mask"] is not None, "mask=None (get_id_color 실패) — _id 미초기화?"

    print("[3] losses ...")
    gt = cam.original_image.cuda()
    gtm = cam.original_mask.cuda()
    Ll1 = l1_loss(pkg["render"], gt)
    Lssim = 1.0 - ssim(pkg["render"], gt)
    Lmask = l1_loss(pkg["mask"], gtm)
    normal_error = (1 - (pkg["rend_normal"] * pkg["surf_normal"]).sum(dim=0))[None]
    Lnormal = normal_error.mean()
    Ldist = pkg["rend_dist"].mean()
    loss = ((1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim + Lmask * 0.25
            + opt.lambda_normal * Lnormal + opt.lambda_dist * Ldist)
    print(f"    Ll1={Ll1.item():.4f} ssim={Lssim.item():.4f} mask={Lmask.item():.4f} "
          f"normal={Lnormal.item():.4f} dist={Ldist.item():.4f}  total={loss.item():.4f}")
    assert torch.isfinite(loss), "loss 가 유한값이 아님!"

    print("[4] backward + densify-stat + optimizer step ...")
    loss.backward()
    vis = pkg["visibility_filter"]
    radii = pkg["radii"]
    gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
    gaussians.add_densification_stats(pkg["viewspace_points"], vis)
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    print("    grad/step ok")

    print("[5] save_ply -> load_ply roundtrip ...")
    out_dir = args.model_path if args.model_path else "."
    os.makedirs(out_dir, exist_ok=True)
    ply = os.path.join(out_dir, "smoke.ply")
    gaussians.save_ply(ply)
    g2 = GaussianModel(dataset.sh_degree, 0, opt.optimizer_type)
    g2.load_ply(ply)
    print(f"    reload #gaussians={g2.get_xyz.shape[0]} scaling={g2._scaling.shape[1]} "
          f"id={g2._id.shape[1]} desc={g2._desc_test.shape[1]}")
    assert g2._scaling.shape[1] == 2 and g2._id.shape[1] == 3

    print("\nSMOKE TEST PASSED ✅  (per-object 스택 통합 정상)")
