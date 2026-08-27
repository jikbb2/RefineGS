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


def infer_latents_guided(model, batch, token_shape, tfe, num_steps, cfg_value,
                         use_shifted, core, obs_n, guide_w, guide_every, guide_t0,
                         free_n=None, guide_free_w=0.0):
    """관측 구속을 넣은 flow matching 샘플링 (infer_latents 의 커스텀 대체).

    왜: ShapeR 는 포인트를 조건으로만 받을 뿐, 생성 결과가 관측점을 실제로 지나간다는
    보장이 없다. rectified flow 에서 현재 상태 x_t 로부터 데이터 추정치는
        x̂₁ = x_t + (1-t)·v
    이므로, 매 스텝 x̂₁ 을 디코딩해 **관측점에서 필드가 0** 이 되도록 그래디언트를 주면
    생성 과정 자체에 관측이 hard constraint 로 들어간다(재학습 불필요).
    """
    from model.flow_matching.helpers.scheduler import FluxTimeSampler
    from model.flow_matching.shaper_denoiser import WrappedModel

    # ※ 반드시 컴파일 전 원본 모듈을 쓴다.
    #   torch.compile(..., fullgraph=True) 된 래퍼를 직접 호출하면 torchsparse 의
    #   `if np.prod(size) % 2 == 1:` 에서 dynamo 가 data-dependent branching 으로 에러.
    #   원래 코드가 멀쩡한 이유는 model.infer_latents(...) 가 메서드 위임이라
    #   내부 self 가 이미 원본 모듈이기 때문이다.
    model = getattr(model, "_orig_mod", model)
    dev = batch["semi_dense_points"].feats.device
    if use_shifted:
        T = FluxTimeSampler(mode="inference")(num_steps, min(token_shape[0], 2048 * 2),
                                              device=dev)
    else:
        T = torch.linspace(0, 1, num_steps, device=dev)
    vm = WrappedModel(model, batch, tfe, None, cfg_value=cfg_value)
    x = model.get_x0_from_input(batch, token_shape=token_shape)
    q = (torch.from_numpy(np.asarray(obs_n, np.float32)).to(dev)
         if obs_n is not None and len(obs_n) else None)
    qf = (torch.from_numpy(np.asarray(free_n, np.float32)).to(dev)
          if free_n is not None and len(free_n) else None)
    on = (guide_w > 0 and q is not None) or (guide_free_w > 0 and qf is not None)

    n_g = 0
    for i in tqdm(range(len(T) - 1), desc="guided sampling"):
        # 시간은 float32 로 유지(dt 정밀도) 하고, 모델에 넣을 때만 모델 dtype 으로 캐스팅.
        # ODESolver 가 해주던 일 — 안 하면 timestep_embedding 이 Float 를 내고
        # bfloat16 Linear 와 곱해져 dtype 불일치 에러가 난다.
        t, tn = T[i].float(), T[i + 1].float()
        v = vm(x=x, t=t.to(x.dtype))
        if on and float(t) >= guide_t0 and (i % max(1, guide_every) == 0):
            with torch.enable_grad():
                x1 = (x + (1.0 - t).to(x.dtype) * v).detach().requires_grad_(True)
                lat = core.decode(x1)
                loss = 0.0
                if guide_w > 0 and q is not None:      # 관측점: 표면 위 → f = 0
                    loss = loss + guide_w * (
                        core.query(q[None].to(lat.dtype), lat).float() ** 2).mean()
                if guide_free_w > 0 and qf is not None:  # 빈 공간: 밖 → f ≥ 0
                    loss = loss + guide_free_w * torch.relu(
                        -core.query(qf[None].to(lat.dtype), lat).float()).mean()
                g, = torch.autograd.grad(loss, x1)
            v = v - g.to(v.dtype)
            n_g += 1
        x = x + (tn - t).to(x.dtype) * v
    if on:
        print(f"  [guide] 구속 적용 {n_g}/{len(T)-1} 스텝  "
              f"obs(w={guide_w}, {0 if q is None else len(q)}점) / "
              f"free(w={guide_free_w}, {0 if qf is None else len(qf)}점), t≥{guide_t0}")
    return x


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
    ap.add_argument("--combine", default="best",
                    choices=["mean", "median", "majority", "best"],
                    help="K개 샘플 결합 방식. mean=필드 평균(형상이 뭉개짐 — 권장 안 함) / "
                         "median=원소별 중앙값(이상치에 강함) / majority=점유 다수결 후 EDT "
                         "(선명함 유지) / best=관측 정합도 최고 샘플 선택(권장)")
    ap.add_argument("--cfg", type=float, default=-1.0,
                    help="classifier-free guidance 배율. ShapeR 에 구현돼 있으나 "
                         "infer_shape.py 는 안 넘겨 기본 비활성(-1). 2~5 로 올리면 조건"
                         "(포인트·이미지·텍스트)을 강하게 따라 mode-averaging(사다리/그물) 완화")
    ap.add_argument("--guide_w", type=float, default=0.0,
                    help="관측 구속 강도(0=off). 매 스텝 x̂₁ 을 디코딩해 관측점에서 "
                         "필드가 0 이 되도록 그래디언트 주입. 0.5~5 부터 시도")
    ap.add_argument("--guide_free_w", type=float, default=0.0,
                    help="free-space 구속 강도(0=off). pkl 의 free_points_model 에서 "
                         "필드가 음수(내부)가 되지 않도록 벌점 → 다리 밑 등 '관측된 빈 공간'의 "
                         "할루시네이션을 생성 단계에서 차단")
    ap.add_argument("--guide_every", type=int, default=1, help="관측 구속 적용 간격(스텝)")
    ap.add_argument("--guide_t0", type=float, default=0.3,
                    help="이 시각 이후에만 구속 적용(초반 저노이즈 구간은 형상이 없어 무의미)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_comp_frac", type=float, default=0.02,
                    help="[부유물] 음수 연결성분 중 최대 성분 부피의 이 비율 미만은 제거. "
                         "0=off. 얇은 다리가 본체와 끊겨 있다면 값을 낮추세요")
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
            # 관측 구속용 점(정규화 좌표) — dataset 과 동일 규약으로 pkl 에서 가져온다
            obs_n = free_n = None
            if args.guide_w > 0 or args.guide_free_w > 0:
                _s = pickle.load(open(pk, "rb"))
                _sc = float(0.9 / np.max(_s["bounds"].numpy()))
                _rng = np.random.default_rng(0)

                def _norm(key, cap=8192):
                    if key not in _s:
                        return None
                    p = _s[key].numpy()[:, :3] * _sc
                    p = p[np.all(np.abs(p) <= 1.0, axis=-1)]
                    if len(p) > cap:
                        p = p[_rng.choice(len(p), cap, replace=False)]
                    return p if len(p) else None

                obs_n = _norm("points_model")
                free_n = _norm("free_points_model")
                if args.guide_free_w > 0 and free_n is None:
                    print("  ⚠ pkl 에 free_points_model 없음 — make_shaper_input 을 "
                          "--depth_dir 와 --free_points 로 다시 생성하세요")
            for k in range(K):
                torch.manual_seed(args.seed + k)           # flow matching 초기 노이즈
                np.random.seed(args.seed + k)
                if args.guide_w > 0 or args.guide_free_w > 0:
                    kl = infer_latents_guided(
                        model, batch, token_shape, tfe, num_steps, args.cfg,
                        use_shifted, core, obs_n, args.guide_w, args.guide_every,
                        args.guide_t0, free_n=free_n, guide_free_w=args.guide_free_w)
                else:
                    kl = model.infer_latents(batch, token_shape=token_shape,
                                             text_feature_extractor=tfe,
                                             num_steps=num_steps,
                                             cfg_value=args.cfg,
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
    Fstd = Fstack.std(0) if len(fields) > 1 else None

    # ---- 관측 정합도로 샘플 채점 (GT 불필요 — 조건으로 준 관측점에서 |f|≈0 이어야) ----
    smp0 = pickle.load(open(pkls[0], "rb"))
    _b0 = smp0["bounds"].numpy(); _sc0 = float(0.9 / np.max(_b0))
    pm = smp0["points_model"].numpy()[:, :3] * _sc0          # dataset 과 동일 정규화
    pm = pm[np.all(np.abs(pm) <= 1.0, axis=-1)]
    pi_ = np.clip((pm + 1.0) * (G - 1) / 2.0, 0, G - 1.001)
    i0 = np.floor(pi_).astype(np.int64); wgt = pi_ - i0; i1 = i0 + 1

    def _probe(vol):                                          # 관측점에서 필드 삼선형 보간
        v = np.zeros(len(pi_), np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w_ = ((wgt[:, 0] if dx else 1 - wgt[:, 0])
                          * (wgt[:, 1] if dy else 1 - wgt[:, 1])
                          * (wgt[:, 2] if dz else 1 - wgt[:, 2]))
                    v += w_ * vol[(i1 if dx else i0)[:, 0],
                                  (i1 if dy else i0)[:, 1],
                                  (i1 if dz else i0)[:, 2]]
        return v

    scores = [float(np.median(np.abs(_probe(f)))) for f in fields]
    print("[score] 관측점 |f| 중앙값(작을수록 관측과 일치): "
          + ", ".join(f"#{i}:{s:.4f}" for i, s in enumerate(scores)))

    if len(fields) == 1 or args.combine == "mean":
        F = Fstack.mean(0)
    elif args.combine == "median":
        F = np.median(Fstack, 0)
    elif args.combine == "best":
        bi = int(np.argmin(scores))
        F = Fstack[bi]
        print(f"[combine] best 샘플 #{bi} 선택 (관측 정합도 최고)")
    else:                                                     # majority: 점유 다수결 → EDT
        from scipy.ndimage import distance_transform_edt as _edt
        occ = (Fstack < 0).mean(0) >= 0.5
        if not occ.any():
            print("  ⚠ 다수결 내부 복셀 없음 — mean 으로 폴백"); F = Fstack.mean(0)
        else:
            vox_n = 2.0 / (G - 1)                             # 정규화 좌표 복셀
            F = ((_edt(~occ) - _edt(occ)) * vox_n).astype(np.float32)
            print(f"[combine] majority 점유 {occ.mean()*100:.2f}% → EDT SDF "
                  "(형상 선명도 유지)")
    if Fstd is not None:
        # 진단 주의: '평균이 0 근처'인 복셀은 (a) 진짜 표면 (b) 부호가 갈려 상쇄된 곳
        # 두 가지가 섞인다. (b)만 보면 σ 가 필드 전 범위에 육박해 과대평가되므로,
        # 표면 근방은 '샘플 각각의 |f|'로 정의하고, 불일치는 물체 부피 대비로 본다.
        near_any = (np.abs(Fstack) < 0.05 * np.abs(Fstack).max()).any(0)
        smed = float(np.median(Fstd[near_any])) if near_any.any() else float("nan")
        disagree = (np.sign(Fstack) != np.sign(F)[None]).any(0)
        obj = (Fstack < 0).any(0)                       # 어느 샘플이든 내부라고 본 복셀
        rel = disagree.sum() / max(obj.sum(), 1) * 100
        print(f"[ensemble] K={len(fields)}  부호 불일치 {disagree.mean()*100:.2f}% "
              f"(물체 부피 대비 {rel:.0f}%)  표면근방 σ 중앙값 {smed:.5f} raw")
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
    if Fstd is not None:                                   # σ 를 미터로 환산해 해석 가능하게
        sm_mm = float(np.median((Fstd / g)[near_any])) * 1000 if near_any.any() else float("nan")
        print(f"        표면근방 σ 중앙값 {sm_mm:.1f}mm "
              f"(sdf_distill 의 --prior_sigma_ref 기본 50mm 와 비교)")
    if (Fm < 0).mean() < 1e-4:
        print("  ⚠ 내부 복셀이 거의 없음 — 부호가 반대일 수 있습니다(--flip 로 확인)")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # [부유물 제거] 본체에서 떨어져 나온 작은 음수 성분을 필드 단계에서 걷어낸다.
    # 융합의 keep_connected 는 '관측과의 연결'을 보므로, 물체 근처에 떠 있는 조각은
    # 통과할 수 있다. 여기서는 크기 기준(최대 성분 대비 비율)이라 그런 조각도 잡힌다.
    if args.min_comp_frac > 0:
        from scipy.ndimage import label as _label
        neg = Fm < 0
        lab, ncomp = _label(neg)
        if ncomp > 1:
            sizes = np.bincount(lab.ravel())[1:]                # 배경(0) 제외
            keep_ids = np.where(sizes >= args.min_comp_frac * sizes.max())[0] + 1
            drop = neg & ~np.isin(lab, keep_ids)
            if drop.any():
                Fm[drop] = float(np.abs(Fm).max())              # 밖(양수)으로 채움
            vox_l = (vox_world ** 3) * 1000                     # 복셀 부피(리터)
            print(f"[floater] 음수 성분 {ncomp}개 → {len(keep_ids)}개 유지, "
                  f"{int(drop.sum())}복셀({drop.sum()*vox_l:.2f}L, "
                  f"내부의 {drop.sum()/max(neg.sum(),1)*100:.2f}%) 제거 "
                  f"(임계 = 최대성분의 {args.min_comp_frac*100:.0f}%)")
        else:
            print(f"[floater] 음수 성분 1개 — 제거할 부유물 없음")

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