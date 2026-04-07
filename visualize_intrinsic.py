# visualize_intrinsic.py
import torch
import torchvision
import os
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from argparse import Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
import sys

# ── 설정 ──
PLY_PATH = "output/hotdog_chromaticity_2/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT_DIR = "output/hotdog_chromaticity_2/visualize"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 모델 로드 ──
gaussians = GaussianModel(0)
gaussians.load_ply(PLY_PATH)
gaussians.shading_mlp = gaussians.shading_mlp.cuda()
print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")

# ── Albedo 분포 확인 ──
albedo = gaussians.get_albedo.detach()
print(f"\n[Albedo 분포]")
print(f"  mean: {albedo.mean():.4f}")
print(f"  std:  {albedo.std():.4f}")
print(f"  min:  {albedo.min():.4f}")
print(f"  max:  {albedo.max():.4f}")

# ── Scene 로드 후 카메라 가져오기 ──
parser = ArgumentParser()
lp = ModelParams(parser)
pp = PipelineParams(parser)

# Scene 로드
args = parser.parse_args(["-s", "data/hotdog",
                          "--model_path", "output/hotdog_chromaticity"])
dataset = lp.extract(args)
pipe = pp.extract(args)

scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
test_cameras = scene.getTrainCameras()

bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

print(f"\n{len(test_cameras)}개 테스트 카메라로 렌더링 중...")

for idx, cam in enumerate(test_cameras[:5]):  # 5장만
    with torch.no_grad():
        # 1) 합성 색상 (albedo × shading)
        pkg = render(cam, gaussians, pipe, bg)
        color = torch.clamp(pkg["render"], 0, 1)

        # 2) Albedo-only 렌더링
        albedo_color = render(cam, gaussians, pipe, bg,
                              override_color=gaussians.get_albedo)
        albedo_img = torch.clamp(albedo_color["render"], 0, 1)

        # 3) Shading-only 렌더링 (shading을 3채널로 확장)
        normals = gaussians.get_normal
        dir_pp = gaussians.get_xyz - cam.camera_center.repeat(
            gaussians.get_xyz.shape[0], 1)
        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
        shading = gaussians.get_shading(normals, gaussians.get_xyz,
                                        dir_pp_normalized)
        shading_rgb = shading.repeat(1, 3)
        shading_pkg = render(cam, gaussians, pipe, bg,
                             override_color=shading_rgb)
        shading_img = torch.clamp(shading_pkg["render"], 0, 1)

    torchvision.utils.save_image(color,
        f"{OUTPUT_DIR}/color_{idx:03d}.png")
    torchvision.utils.save_image(albedo_img,
        f"{OUTPUT_DIR}/albedo_{idx:03d}.png")
    torchvision.utils.save_image(shading_img,
        f"{OUTPUT_DIR}/shading_{idx:03d}.png")

    print(f"  [{idx+1}/5] 저장 완료")

print(f"\n결과 저장 위치: {OUTPUT_DIR}/")
print("확인 포인트:")
print("  albedo_*.png  → 조명 없이 균일한 색상이어야 함")
print("  shading_*.png → 그림자/밝기 변화만 보여야 함")
print("  color_*.png   → GT와 유사해야 함")