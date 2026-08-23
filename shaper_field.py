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
    ap.add_argument("--seed", type=int, default=0, help="flow matching 초기 노이즈 시드")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="K개 시드로 필드를 뽑아 mean/std 저장. >50%% 미관측에서는 정답이 "
                         "하나가 아니므로, 합의도(σ)를 융합 가중치로 넘긴다")
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

    pkls = [p.strip() for p in args.input_pkl.split(",") if p.strip()]
    G = args.grid
    lin = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    core = getattr(vae.model, "_orig_mod", vae.model)       # torch.compile 래퍼 우회

    # 다양성 원천은 두 가지: (a) 시드 --ensemble, (b) 서로 다른 입력 pkl(포인트 서브샘플·
    # 뷰 구성이 다른 것)을 콤마로 여러 개. 모델이 조건 대비 결정적이면 (a)는 σ≈0 이 되므로
    # (b) 가 실질적인 epistemic 다양성을 준다.
    fields = []
    K = max(1, args.ensemble)
    with torch.no_grad():
        for pi, pk in enumerate(pkls):
            ds = InferenceDataset(config, paths=[pk], override_num_views=num_images)
            loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                                 num_workers=0,
                                                 collate_fn=ds.custom_collate)
            batch = next(iter(loader))
            batch = InferenceDataset.move_batch_to_device(batch, device,
                                                          dtype=torch.bfloat16)
            for k in range(K):
                torch.manual_seed(args.seed + k)           # flow matching 초기 노이즈
                np.random.seed(args.seed + k)
                kl = model.infer_latents(batch, token_shape=token_shape,
                                         text_feature_extractor=tfe, num_steps=num_steps,
                                         use_shifted_sampling=use_shifted)
                latents = core.decode(kl)
                # 부호 있는 필드를 [-1,1]^3 그리드에 평가 (extract_mesh 와 동일 규약)
                vals = np.empty(len(pts), np.float32)
                tag = f"{pi * K + k + 1}/{len(pkls) * K}"
                for s in tqdm(range(0, len(pts), args.chunk), desc=f"field[{tag}]"):
                    q = torch.from_numpy(pts[s:s + args.chunk]).to(
                        device=device, dtype=latents.dtype)[None]
                    vals[s:s + args.chunk] = core.query(q, latents)[0].float().cpu().numpy()
                fields.append(vals.reshape(G, G, G))
    Fstack = np.stack(fields)
    F = Fstack.mean(0)
    Fstd = Fstack.std(0) if len(fields) > 1 else None
    if Fstd is not None:
        # ※ 전체 복셀 σ 중앙값은 포화 영역(멀리) 때문에 항상 0 에 가깝다 —
        #   의미 있는 값은 '영교차 근방' σ. 여기가 0 이면 샘플이 사실상 동일하다는 뜻.
        nz = np.abs(F) < 0.05 * np.abs(F).max()
        smed = float(np.median(Fstd[nz])) if nz.any() else float("nan")
        agree = (np.sign(Fstack) == np.sign(F)[None]).all(0)
        print(f"[ensemble] K={len(fields)}  부호 합의 {agree.mean()*100:.1f}%  "
              f"표면근방 σ 중앙값 {smed:.5f} / 최대 {Fstd.max():.5f} (raw 단위)")
        if smed < 1e-5:
            print("  ⚠ 샘플이 사실상 동일 — 시드가 초기 노이즈에 영향을 못 줍니다.\n"
                  "     서로 다른 pkl(포인트 서브샘플 다름)을 --input_pkl 에 콤마로 넘기세요.")

    # ---- pkl 에서 world 변환 복원 ----
    smp = pickle.load(open(pkls[0], "rb"))     # world 변환은 첫 pkl 기준
    if len(pkls) > 1:                          # 여러 pkl 이면 프레임이 같아야 평균이 유효
        for pk in pkls[1:]:
            o = pickle.load(open(pk, "rb"))
            assert np.allclose(o["T_model_world"].numpy(), smp["T_model_world"].numpy(),
                               atol=1e-6) and np.allclose(o["bounds"].numpy(),
                                                          smp["bounds"].numpy(), atol=1e-6), \
                (f"pkl 간 오브젝트 프레임 불일치: {pk}\n"
                 "  → make_shaper_input 을 --n_points 만 바꿔 만들면 bounds 가 달라질 수 있습니다.\n"
                 "     동일 프레임을 쓰려면 같은 --bounds_margin 과 같은 recon 을 쓰고,\n"
                 "     bounds/center 를 첫 pkl 값으로 맞추세요.")
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
    save = dict(field=Fm, center=center.astype(np.float64),
                R_align=R_align.astype(np.float64), scale=np.float64(scale),
                vox_world=np.float64(vox_world), raw_g=np.float64(g))
    if Fstd is not None:
        save["field_std"] = (Fstd / g).astype(np.float32)   # 동일 스케일로 환산
    np.savez_compressed(out, **save)
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
