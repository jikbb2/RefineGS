#!/usr/bin/env python3
"""ShapeR -> phi_prior (a SIGNED SDF grid).

Lives in this repository, runs against a ShapeR checkout given by --shaper_root:

  PYTHONPATH="<repo>/shaper:$SHAPER_DIR" python <repo>/shaper/shaper_field.py \
      --shaper_root "$SHAPER_DIR" \
      --input_pkl output/<scene>/prior/<pipeline>/pkl/obj1.pkl --config balance \
      --grid 256 --out output/<scene>/prior/<pipeline>/obj1_field.npz

sys.path[0] is this file's directory, which is how `import infer_shape_pinhole` below finds
the copy in this repository; $SHAPER_DIR on PYTHONPATH supplies dataset.* and model.*. Keep
only this file and infer_shape_pinhole.py in that directory: a dataset.py or model.py beside
them would shadow ShapeR's own packages.

ShapeR resolves checkpoints/019-0-bfloat16.ckpt, checkpoints/config.yaml and
checkpoints/vae-088-0-bfloat16.ckpt relative to the working directory, and its
setup_checkpoints() does `Path("checkpoints").mkdir()` before downloading, so main() changes
into --shaper_root first. Every path this script was given is made absolute BEFORE that, or a
relative --out would land inside somebody's ShapeR checkout.

Why a field and not a mesh
--------------------------
`AutoEncoder.extract_mesh` shows that the decoder output is SIGNED to begin with:
    if use_udf_extraction:  marching_cubes(np.abs(grid_sdf), udf_iso)   # <- sign discarded
    else:                   marching_cubes(grid_sdf, 0)                 # <- plain SDF
infer_shape.py takes the first path (udf_iso=0.375), where |f|=iso has two solutions, one
inside the surface and one outside, so it produces a SHELL ON EITHER SIDE -- an offset shell
is already baked in. Adding our own sign fix and shell_delta on top of that inflates it
twice, which over-generates and costs accuracy.

Here `vae.model.query(queries, latents)` reads the signed field straight onto a grid and
saves it as an npz in world coordinates, to be injected through sdf_distill_depth.py's
--prior_field. sign-fix, shell_delta and unseen_open all become unnecessary as a result.

Output npz:
  field   (G,G,G) float32  approximately metric SDF (negative = inside), eikonal-normalised
  center  (3,)  object centre in world,  R_align (3,3),  scale  (normalisation factor)
  raw_g   float  the median |grad f| used for the eikonal normalisation (diagnostic)
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

import infer_shape_pinhole  # noqa: F401  (applies the pinhole rectify bypass)

preset_configs = {"quality": (16, 4, 50), "speed": (4, 2, 10), "balance": (16, 4, 25)}


def infer_latents_guided(model, batch, token_shape, tfe, num_steps, cfg_value,
                         use_shifted, core, obs_n, guide_w, guide_every, guide_t0,
                         free_n=None, guide_free_w=0.0):
    """Flow-matching sampling with an observation constraint (a custom replacement for
    infer_latents).

    Why: ShapeR takes the points as a condition only -- nothing guarantees the generated
    surface actually passes through them. In a rectified flow, the data estimate from the
    current state x_t is
        x_hat_1 = x_t + (1-t)*v
    so decoding x_hat_1 at every step and pushing the field towards zero AT the observed
    points turns the observation into a hard constraint on the sampling process itself, with
    no retraining.
    """
    from model.flow_matching.helpers.scheduler import FluxTimeSampler
    from model.flow_matching.shaper_denoiser import WrappedModel

    # Must use the pre-compile module. Calling the torch.compile(..., fullgraph=True) wrapper
    # directly makes dynamo fail on torchsparse's `if np.prod(size) % 2 == 1:` as
    # data-dependent branching. The upstream code is fine because model.infer_latents(...)
    # delegates to a method whose self is already the original module.
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
        # Keep time in float32 (dt precision) and cast to the model dtype only on the way in.
        # ODESolver used to do this; without it timestep_embedding returns a Float that then
        # multiplies a bfloat16 Linear and raises a dtype mismatch.
        t, tn = T[i].float(), T[i + 1].float()
        v = vm(x=x, t=t.to(x.dtype))
        if on and float(t) >= guide_t0 and (i % max(1, guide_every) == 0):
            with torch.enable_grad():
                x1 = (x + (1.0 - t).to(x.dtype) * v).detach().requires_grad_(True)
                lat = core.decode(x1)
                loss = 0.0
                if guide_w > 0 and q is not None:      # observed points: on the surface -> f = 0
                    loss = loss + guide_w * (
                        core.query(q[None].to(lat.dtype), lat).float() ** 2).mean()
                if guide_free_w > 0 and qf is not None:  # empty space: outside -> f >= 0
                    loss = loss + guide_free_w * torch.relu(
                        -core.query(qf[None].to(lat.dtype), lat).float()).mean()
                g, = torch.autograd.grad(loss, x1)
            v = v - g.to(v.dtype)
            n_g += 1
        x = x + (tn - t).to(x.dtype) * v
    if on:
        print(f"  [guide] constraint applied on {n_g}/{len(T)-1} steps  "
              f"obs(w={guide_w}, {0 if q is None else len(q)} pts) / "
              f"free(w={guide_free_w}, {0 if qf is None else len(qf)} pts), t>={guide_t0}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pkl", required=True, help="local pkl path, used as given")
    ap.add_argument("--config", default="balance", choices=list(preset_configs))
    ap.add_argument("--grid", type=int, default=256, help="field grid resolution (points per axis)")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0, help="flow-matching initial noise seed")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="draw the field with K seeds and save mean/std. Above 50%% unobserved "
                         "there is no single right answer, so the agreement (sigma) is passed "
                         "on as a fusion weight")
    ap.add_argument("--combine", default="best",
                    choices=["mean", "median", "majority", "best"],
                    help="how to combine the K samples. mean=average the fields (smears the "
                         "shape -- not recommended) / median=elementwise median (robust to "
                         "outliers) / majority=vote on occupancy then EDT (stays sharp) / "
                         "best=pick the sample that fits the observation best (recommended)")
    ap.add_argument("--cfg", type=float, default=-1.0,
                    help="classifier-free guidance scale. ShapeR implements it but "
                         "infer_shape.py never passes it, so it is off by default (-1). "
                         "Raising it to 2-5 follows the conditions (points, images, text) more "
                         "closely and relieves the mode-averaging (ladders, webbing)")
    ap.add_argument("--guide_w", type=float, default=0.0,
                    help="observation constraint strength (0=off). Decodes x_hat_1 at every "
                         "step and injects a gradient driving the field to zero at the "
                         "observed points. Start between 0.5 and 5")
    ap.add_argument("--guide_free_w", type=float, default=0.0,
                    help="free-space constraint strength (0=off). Penalises a negative "
                         "(inside) field at the pkl's free_points_model, which blocks the "
                         "hallucination under a table and the like at generation time")
    ap.add_argument("--guide_every", type=int, default=1, help="apply the constraint every N steps")
    ap.add_argument("--guide_t0", type=float, default=0.3,
                    help="apply the constraint only after this time (early, low-noise steps "
                         "have no shape to constrain)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_comp_frac", type=float, default=0.02,
                    help="[floaters] drop any negative connected component below this "
                         "fraction of the largest one's volume. 0=off. Lower it if a thin leg "
                         "is disconnected from the body")
    ap.add_argument("--save_mesh", default="", help="optional: zero-level mesh of the signed field (for checking)")
    # The ShapeR checkout. Its checkpoints/ holds the weights, and ShapeR reads them relative
    # to the working directory, so main() chdir's here. $SHAPER_DIR is the default so the
    # driver does not have to pass it twice.
    ap.add_argument("--shaper_root", default=os.environ.get("SHAPER_DIR", ""),
                    help="ShapeR checkout holding checkpoints/ (default: $SHAPER_DIR). "
                         "ShapeR resolves its weights relative to the working directory, so "
                         "this script changes into it")
    args = ap.parse_args()

    # Absolute BEFORE the chdir. A relative --out would otherwise be written inside the
    # ShapeR checkout, where nothing downstream looks for it and nobody would notice.
    args.out = os.path.abspath(os.path.expanduser(args.out))
    args.input_pkl = ",".join(os.path.abspath(os.path.expanduser(p))
                              for p in args.input_pkl.split(",") if p.strip())
    if args.save_mesh:
        args.save_mesh = os.path.abspath(os.path.expanduser(args.save_mesh))
    if args.shaper_root:
        _sr = os.path.abspath(os.path.expanduser(args.shaper_root))
        if not os.path.isdir(_sr):
            raise SystemExit(f"[abort] --shaper_root is not a directory: {_sr}")
        if not os.path.isdir(os.path.join(_sr, "checkpoints")):
            print(f"[warn] no checkpoints/ under {_sr} -- setup_checkpoints() will download "
                  "about 3.9 GB from HuggingFace into it")
        os.chdir(_sr)
        print(f"[shaper] working directory -> {_sr}")

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
    core = getattr(vae.model, "_orig_mod", vae.model)       # bypass the torch.compile wrapper

    # Diversity has two sources: (a) the seed, via --ensemble, and (b) several different input
    # pkls given comma-separated (different point subsamples, different view sets). If the
    # model is deterministic given its conditions, (a) gives sigma ~ 0 and (b) is what
    # provides real epistemic diversity.
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
            # Points for the constraint, in normalised coordinates -- taken from the pkl under
            # the same convention the dataset uses.
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
                    print("  ! no free_points_model in the pkl -- rebuild it with "
                          "make_shaper_input --depth_dir and --free_points")
            for k in range(K):
                torch.manual_seed(args.seed + k)           # flow-matching initial noise
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
                # Evaluate the signed field on the [-1,1]^3 grid (the same convention
                # extract_mesh uses).
                vals = np.empty(len(pts), np.float32)
                tag = f"{pi * K + k + 1}/{len(pkls) * K}"
                for s in tqdm(range(0, len(pts), args.chunk), desc=f"field[{tag}]"):
                    q = torch.from_numpy(pts[s:s + args.chunk]).to(
                        device=device, dtype=latents.dtype)[None]
                    vals[s:s + args.chunk] = core.query(q, latents)[0].float().cpu().numpy()
                fields.append(vals.reshape(G, G, G))
    Fstack = np.stack(fields)
    Fstd = Fstack.std(0) if len(fields) > 1 else None

    # ---- score the samples by their fit to the observation (no GT needed: |f| should be
    #      about 0 at the points that were given as the condition) ----
    smp0 = pickle.load(open(pkls[0], "rb"))
    _b0 = smp0["bounds"].numpy(); _sc0 = float(0.9 / np.max(_b0))
    pm = smp0["points_model"].numpy()[:, :3] * _sc0          # the dataset's normalisation
    pm = pm[np.all(np.abs(pm) <= 1.0, axis=-1)]
    pi_ = np.clip((pm + 1.0) * (G - 1) / 2.0, 0, G - 1.001)
    i0 = np.floor(pi_).astype(np.int64); wgt = pi_ - i0; i1 = i0 + 1

    def _probe(vol):                                          # trilinear field value at the points
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
    print("[score] median |f| at the observed points (lower = closer to the observation): "
          + ", ".join(f"#{i}:{s:.4f}" for i, s in enumerate(scores)))

    if len(fields) == 1 or args.combine == "mean":
        F = Fstack.mean(0)
    elif args.combine == "median":
        F = np.median(Fstack, 0)
    elif args.combine == "best":
        bi = int(np.argmin(scores))
        F = Fstack[bi]
        print(f"[combine] best: sample #{bi} (closest fit to the observation)")
    else:                                                     # majority: occupancy vote -> EDT
        from scipy.ndimage import distance_transform_edt as _edt
        occ = (Fstack < 0).mean(0) >= 0.5
        if not occ.any():
            print("  ! no voxel is inside by majority -- falling back to mean"); F = Fstack.mean(0)
        else:
            vox_n = 2.0 / (G - 1)                             # voxel in normalised coordinates
            F = ((_edt(~occ) - _edt(occ)) * vox_n).astype(np.float32)
            print(f"[combine] majority occupancy {occ.mean()*100:.2f}% -> EDT SDF "
                  "(keeps the shape sharp)")
    if Fstd is not None:
        # A caveat on this diagnostic: a voxel whose mean is near zero is either (a) really on
        # the surface or (b) a place where the signs cancelled. Reading only (b) makes sigma
        # approach the field's whole range, so "near the surface" is defined per sample, by
        # its own |f|, and disagreement is reported against the object's volume.
        near_any = (np.abs(Fstack) < 0.05 * np.abs(Fstack).max()).any(0)
        smed = float(np.median(Fstd[near_any])) if near_any.any() else float("nan")
        disagree = (np.sign(Fstack) != np.sign(F)[None]).any(0)
        obj = (Fstack < 0).any(0)                       # voxels any sample called inside
        rel = disagree.sum() / max(obj.sum(), 1) * 100
        print(f"[ensemble] K={len(fields)}  sign disagreement {disagree.mean()*100:.2f}% "
              f"({rel:.0f}% of the object volume)  median sigma near the surface {smed:.5f} raw")
        if smed < 1e-5:
            print("  ! the samples are effectively identical -- the seed is not reaching the\n"
                  "     initial noise. Pass several pkls (different point subsamples) to "
                  "--input_pkl, comma-separated.")

    # ---- recover the world transform from the pkl ----
    smp = pickle.load(open(pkls[0], "rb"))     # the world transform comes from the first pkl
    if len(pkls) > 1:                          # with several pkls the frames must agree, or the average is meaningless
        for pk in pkls[1:]:
            o = pickle.load(open(pk, "rb"))
            assert np.allclose(o["T_model_world"].numpy(), smp["T_model_world"].numpy(),
                               atol=1e-6) and np.allclose(o["bounds"].numpy(),
                                                          smp["bounds"].numpy(), atol=1e-6), \
                (f"object frames disagree between pkls: {pk}\n"
                 "  -> building them with make_shaper_input and only changing --n_points can\n"
                 "     change bounds. For one frame, use the same --bounds_margin and the same\n"
                 "     recon, and align bounds/center to the first pkl's values.")
    bounds = smp["bounds"].numpy()
    scale = float(0.9 / np.max(bounds))                    # the dataset's convention
    Tmw = smp["T_model_world"].numpy()                     # world -> model
    R_align = Tmw[:3, :3]
    center = -R_align.T @ Tmw[:3, 3]
    vox_world = (2.0 / (G - 1)) / scale                    # world length of one grid cell (m)

    # ---- eikonal normalisation: field units -> approximately metres ----
    # The decoder output is not a metric SDF (its scale depends on the training objective), so
    # divide by the median |grad f| near the zero crossing to fix the distance scale.
    gxg, gyg, gzg = np.gradient(F, vox_world)
    gmag = np.sqrt(gxg ** 2 + gyg ** 2 + gzg ** 2)
    sgn = np.sign(F)
    near = np.zeros_like(F, bool)
    near[:-1] |= sgn[:-1] != sgn[1:]
    near[:, :-1] |= sgn[:, :-1] != sgn[:, 1:]
    near[:, :, :-1] |= sgn[:, :, :-1] != sgn[:, :, 1:]
    g = float(np.median(gmag[near])) if near.any() else 1.0
    assert g > 1e-8, "no zero crossing -- generation failed, or check the sign convention"
    Fm = (F / g).astype(np.float32)
    print(f"[field] G={G} voxel={vox_world*1000:.2f}mm  median |grad f|={g:.4f} "
          f"-> metric\n        inside voxels {(Fm < 0).mean()*100:.2f}%  "
          f"range [{Fm.min():.3f}, {Fm.max():.3f}]m")
    if Fstd is not None:                                   # sigma in metres, so it can be read
        sm_mm = float(np.median((Fstd / g)[near_any])) * 1000 if near_any.any() else float("nan")
        print(f"        median sigma near the surface {sm_mm:.1f}mm "
              f"(compare sdf_distill's --prior_sigma_ref, 50mm by default)")
    if (Fm < 0).mean() < 1e-4:
        print("  ! almost no voxel is inside -- the sign may be inverted (check with --flip)")

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # [floater removal] Strip the small negative components that broke off the body, at the
    # field stage. The fusion's keep_connected looks at connectivity TO THE OBSERVATION, so a
    # fragment floating near the object can pass it. The criterion here is size (a fraction of
    # the largest component), which catches those too.
    if args.min_comp_frac > 0:
        from scipy.ndimage import label as _label
        neg = Fm < 0
        lab, ncomp = _label(neg)
        if ncomp > 1:
            sizes = np.bincount(lab.ravel())[1:]                # excluding the background (0)
            keep_ids = np.where(sizes >= args.min_comp_frac * sizes.max())[0] + 1
            drop = neg & ~np.isin(lab, keep_ids)
            if drop.any():
                Fm[drop] = float(np.abs(Fm).max())              # fill as outside (positive)
            vox_l = (vox_world ** 3) * 1000                     # voxel volume (litres)
            print(f"[floater] {ncomp} negative components -> {len(keep_ids)} kept, "
                  f"{int(drop.sum())} voxels ({drop.sum()*vox_l:.2f}L, "
                  f"{drop.sum()/max(neg.sum(),1)*100:.2f}% of the inside) removed "
                  f"(threshold = {args.min_comp_frac*100:.0f}% of the largest)")
        else:
            print(f"[floater] one negative component -- nothing to remove")

    save = dict(field=Fm, center=center.astype(np.float64),
                R_align=R_align.astype(np.float64), scale=np.float64(scale),
                vox_world=np.float64(vox_world), raw_g=np.float64(g))
    if Fstd is not None:
        save["field_std"] = (Fstd / g).astype(np.float32)   # converted to the same scale
    np.savez_compressed(out, **save)
    print(f"-> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")

    if args.save_mesh:                                     # to eyeball the sign convention
        from skimage import measure
        import trimesh
        v, f_, _, _ = measure.marching_cubes(Fm, 0.0, method="lewiner",
                                             gradient_direction="ascent")
        v = v * vox_world + (center - (G - 1) / 2 * vox_world)   # assumes R_align=I
        trimesh.Trimesh(v, f_[:, [2, 1, 0]]).export(args.save_mesh)
        print(f"-> zero-level mesh (for checking): {args.save_mesh}")


if __name__ == "__main__":
    main()
