#!/usr/bin/env python3
"""ShapeR → φ_prior (부호 있는 SDF 그리드) 추출. ShapeR 저장소 루트에서 실행.

왜 메쉬가 아니라 필드인가
-------------------------
`AutoEncoder.extract_mesh` 를 보면 디코더 출력은 원래 **signed** 다:
    if use_udf_extraction:  marching_cubes(np.abs(grid_sdf), udf_iso)   # ← 부호 버림
    else:                   marching_cubes(grid_sdf, 0)                 # ← 정상 SDF
infer_shape.py 는 앞 경로(udf_iso=0.375)를 쓰므로, |f|=iso 의 해가 안/밖 두 개라
**표면 양쪽에 껍질**이 생긴다(= 오프셋 셸이 이미 내장). 그 위에 우리 sign-fix +
shell_delta 를 또 씌우면 이중 팽창이 되어 과생성·정밀도 하락으로 이어진다.

여기서는 `vae.model.query(queries, latents)` 로 부호 있는 필드를 그대로 그리드에 떠서
world 좌표계 npz 로 저장한다 → sdf_distill_depth.py 의 `--prior_field` 로 주입.
결과적으로 sign-fix / shell_delta / unseen_open 이 모두 불필요해진다.

  python shaper_field.py --input_pkl data/refinegs_obj1.pkl --config balance \
      --grid 256 --out ~/prior/obj1_shaper_field.npz

출력 npz:
  field   (G,G,G) float32  근사 metric SDF (음수=내부). eikonal 정규화 적용
  center  (3,)  world 오브젝트 중심,  R_align (3,3),  scale  (정규화 배율)
  raw_g   float  eikonal 정규화에 쓴 |∇f| 중앙값 (진단용)
"""
import argparse
import os
import pickle

import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from dataset.shaper_dataset import InferenceDataset
from model.download import setup_checkpoints
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper

import infer_shape_pinhole  # noqa: F401  (핀홀 rectify 우회 패치 적용)

preset_configs = {"quality": (16, 4, 50), "speed": (4, 2, 10), "balance": (16, 4, 25)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkl", required=True, help="로컬 pkl 경로(그대로 사용)")
    ap.add_argument("--config", default="balance", choices=list(preset_configs))
    ap.add_argument("--grid", type=int, default=256, help="필드 그리드 해상도(축당 점 수)")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_mesh", default="", help="선택: 부호 필드의 zero-level 메쉬(검증용)")
    args = ap.parse_args()

    num_images, token_multiplier, num_steps = preset_configs[args.config]
    setup_checkpoints()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load("checkpoints/019-0-bfloat16.ckpt", map_location=device,
                            weights_only=False)
    config = omegaconf.OmegaConf.load("checkpoints/config.yaml")
    print("Loading model...")
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)
    vae = MichelangeloLikeAutoencoderWrapper("checkpoints/vae-088-0-bfloat16.ckpt", device)
    tfe = TextFeatureExtractor(device=device).to(torch.bfloat16)
    model = torch.compile(model, fullgraph=True).eval()

    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales); scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    token_shape = (1, token_count, vae.get_embed_dim())
    use_shifted = getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"

    ds = InferenceDataset(config, paths=[args.input_pkl], override_num_views=num_images)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=0, collate_fn=ds.custom_collate)

    with torch.no_grad():
        batch = next(iter(loader))
        batch = InferenceDataset.move_batch_to_device(batch, device, dtype=torch.bfloat16)
        kl = model.infer_latents(batch, token_shape=token_shape,
                                 text_feature_extractor=tfe, num_steps=num_steps,
                                 use_shifted_sampling=use_shifted)
        core = getattr(vae.model, "_orig_mod", vae.model)   # torch.compile 래퍼 우회
        latents = core.decode(kl)

        # ---- 부호 있는 필드를 [-1,1]^3 그리드에 평가 (extract_mesh 와 동일 규약) ----
        G = args.grid
        lin = np.linspace(-1.0, 1.0, G, dtype=np.float32)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        pts = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        vals = np.empty(len(pts), np.float32)
        for s in tqdm(range(0, len(pts), args.chunk), desc="field"):
            q = torch.from_numpy(pts[s:s + args.chunk]).to(device=device,
                                                           dtype=latents.dtype)[None]
            vals[s:s + args.chunk] = core.query(q, latents)[0].float().cpu().numpy()
        F = vals.reshape(G, G, G)

    # ---- pkl 에서 world 변환 복원 ----
    smp = pickle.load(open(args.input_pkl, "rb"))
    bounds = smp["bounds"].numpy()
    scale = float(0.9 / np.max(bounds))                    # dataset 과 동일 규약
    Tmw = smp["T_model_world"].numpy()                     # world → model
    R_align = Tmw[:3, :3]
    center = -R_align.T @ Tmw[:3, 3]
    vox_world = (2.0 / (G - 1)) / scale                    # 그리드 1칸의 world 길이(m)

    # ---- eikonal 정규화: 필드 단위 → 근사 미터 ----
    # 디코더 출력은 metric SDF 가 아니므로(학습 목표에 따라 스케일이 다름)
    # 영교차 근방의 |∇f| 중앙값으로 나눠 거리 스케일을 맞춘다.
    gxg, gyg, gzg = np.gradient(F, vox_world)
    gmag = np.sqrt(gxg ** 2 + gyg ** 2 + gzg ** 2)
    sgn = np.sign(F)
    near = np.zeros_like(F, bool)
    near[:-1] |= sgn[:-1] != sgn[1:]
    near[:, :-1] |= sgn[:, :-1] != sgn[:, 1:]
    near[:, :, :-1] |= sgn[:, :, :-1] != sgn[:, :, 1:]
    g = float(np.median(gmag[near])) if near.any() else 1.0
    assert g > 1e-8, "영교차 없음 — 생성 실패 또는 부호 규약 확인 필요"
    Fm = (F / g).astype(np.float32)
    print(f"[field] G={G} voxel={vox_world*1000:.2f}mm  |∇f| 중앙값={g:.4f} "
          f"→ metric 환산\n        내부 복셀 {(Fm < 0).mean()*100:.2f}%  "
          f"범위 [{Fm.min():.3f}, {Fm.max():.3f}]m")
    if (Fm < 0).mean() < 1e-4:
        print("  ⚠ 내부 복셀이 거의 없음 — 부호가 반대일 수 있습니다(--flip 로 확인)")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, field=Fm, center=center.astype(np.float64),
                        R_align=R_align.astype(np.float64), scale=np.float64(scale),
                        vox_world=np.float64(vox_world), raw_g=np.float64(g))
    print(f"→ {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

    if args.save_mesh:                                     # 부호 규약 육안 검증용
        from skimage import measure
        import trimesh
        v, f_, _, _ = measure.marching_cubes(Fm, 0.0, method="lewiner",
                                             gradient_direction="ascent")
        v = v * vox_world + (center - (G - 1) / 2 * vox_world)   # R_align=I 가정
        trimesh.Trimesh(v, f_[:, [2, 1, 0]]).export(os.path.expanduser(args.save_mesh))
        print(f"→ zero-level 메쉬(검증): {args.save_mesh}")


if __name__ == "__main__":
    main()
