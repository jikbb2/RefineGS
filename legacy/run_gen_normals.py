#!/usr/bin/env python3
"""gen_*.jpg → normal_%04d.png (단안 normal 추정, Metric3D v2 torch.hub).

NV normal 감독([v2] train.py --nv_lambda_normal)용 전처리.
출력: camera-space normal, png ((n*0.5+0.5)*255). train.py 쪽 loss가 1-|cos| 라
부호/축 규약 차이에 강건하지만, 첫 실행 후 normal_*.png 하나를 눈으로 확인 권장
(테이블 윗면이 한 색으로 균일하면 정상).

  conda activate mono
  python run_gen_normals.py --gen_dir ~/See3D/dataset/obj24_v2/gen_traj --fov_deg 60

fov_deg: poses.npz의 FoVx(라디안)를 도로 변환해 넣으면 정확 (없으면 60 근사).
"""
import os
import glob
import argparse
import numpy as np
import torch
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--model", default="metric3d_vit_small",
                    choices=["metric3d_vit_small", "metric3d_vit_large"])
    ap.add_argument("--fov_deg", type=float, default=60.0, help="수평 FoV(도) — intrinsic 근사용")
    args = ap.parse_args()

    gen_dir = os.path.expanduser(args.gen_dir)
    files = sorted(glob.glob(os.path.join(gen_dir, "gen_*.jpg")))
    assert files, f"gen_*.jpg 없음: {gen_dir}"

    model = torch.hub.load("yvanyin/metric3d", args.model, pretrain=True).cuda().eval()

    # Metric3D 권장 전처리 상수 (vit: 616x1064 캔버스)
    input_size = (616, 1064)
    mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

    for f in files:
        i = os.path.basename(f)[4:8]
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        fx = W / (2 * np.tan(np.deg2rad(args.fov_deg) / 2))
        intrinsic = [fx, fx, W / 2, H / 2]

        scale = min(input_size[0] / H, input_size[1] / W)
        rs = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
        h, w = rs.shape[:2]
        pad_h, pad_w = input_size[0] - h, input_size[1] - w
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        canvas = cv2.copyMakeBorder(rs, pt, pb, pl, pr, cv2.BORDER_CONSTANT,
                                    value=[123.675, 116.28, 103.53])
        t = torch.from_numpy(canvas.transpose(2, 0, 1)).float()
        t = ((t - mean) / std)[None].cuda()

        with torch.no_grad():
            _, _, out = model.inference({"input": t})
        assert "prediction_normal" in out, "이 모델은 normal 미출력 — vit 계열 사용"
        n = out["prediction_normal"][0, :3]                     # (3,616,1064) camera-space
        n = n[:, pt:input_size[0] - pb, pl:input_size[1] - pr]  # un-pad
        n = torch.nn.functional.interpolate(n[None], (H, W), mode="bilinear")[0]
        n = torch.nn.functional.normalize(n, dim=0).cpu().numpy()
        png = ((n * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite(os.path.join(gen_dir, f"normal_{i}.png"), cv2.cvtColor(png, cv2.COLOR_RGB2BGR))
        print(f"normal_{i}.png")

    print(f"done → {gen_dir} (train.py --nv_lambda_normal 0.2 로 사용)")


if __name__ == "__main__":
    main()
