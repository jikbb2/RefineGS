# RefineGS

Robust **per-object 3D reconstruction**: SAM3 video re-labeling → part-aware amodal masking → observation-consistent per-object 2DGS refinement → scene assembly.

The pipeline addresses three verified weaknesses of Split&Splat-style per-object reconstruction: (1) re-labeling failure in instance-dense scenes, (2) part/whole granularity inconsistency, and (3) per-instance quality variance. The guiding principle is **input-quality-first** (segmentation gates reconstruction) and **observation-consistency** (geometry must not contradict the images; the unverifiable part is quantified, not hallucinated).

## Repository layout

```
refinegs/
  relabel/    sam3_relabel_video.py        # Axis 1: SAM3 video predictor re-labeling
  amodal/     amodal_mask.py               # Axis 2: fill on-object occlusion holes
              amodal_complete_general.py   # Axis 3: observation-consistent surface completion
  refine/     register_generated_to_recon.py  # Sim(3) + Chamfer-rotation + scaled ICP
              fuse_carve.py                # free-space + visual-hull veto fusion
              obs_consistency_report.py    # certificate (violating/observed/unverified %)
  assemble/   scene_assemble.py            # concat per-object outputs → one scene
scripts/      run_full_pipeline.sh         # 2-stage driver (relabel | recon)
tools/        eval_object_mesh.py          # visibility-split metrics, auto-match to GT
docs/         design notes (problem redefinition, novelty landscape, axis designs)
archive/      superseded experiments (kept for provenance)
envs/         conda environment exports
configs/      hyperparameters (move hardcoded values here)
third_party/  patches/ for external deps (NOT the deps themselves)
data/, output/   (git-ignored)
```

## Environments

Two conda environments (kept separate — the SAM3 stack and the 2DGS stack have conflicting CUDA/cuDNN):

```bash
conda env create -f envs/sam3.yml              # SAM3 re-labeling
conda env create -f envs/split_and_splat.yml   # 2DGS train/render + fusion/eval
```

Regenerate the exports anytime with:

```bash
conda activate sam3            && conda env export --no-builds > envs/sam3.yml
conda activate split_and_splat && conda env export --no-builds > envs/split_and_splat.yml
```

## External dependencies (sam3, Amodal3R) — not vendored

`sam3` and `Amodal3R` are **separate upstream repositories** and are intentionally **not** copied into this repo (size, licensing, and we apply local patches). Treat them as external dependencies:

1. **Clone next to this repo and pin the commit you used:**
   ```bash
   git clone <SAM3_URL>     ~/sam3      && git -C ~/sam3      checkout <SAM3_COMMIT>
   git clone <AMODAL3R_URL> ~/Amodal3R  && git -C ~/Amodal3R checkout <AMODAL3R_COMMIT>
   ```
   Record the exact commit hashes in this section so results are reproducible.

2. **Save our local patches** (do not commit their whole trees — commit only the diffs):
   ```bash
   git -C ~/sam3     diff > third_party/patches/sam3.patch
   git -C ~/Amodal3R diff > third_party/patches/amodal3r.patch
   ```
   Known local changes to document/patch:
   - SAM3 BPE path passed explicitly: `--bpe <sam3>/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
   - cuDNN conflict in the sam3 env: prefix the call with `LD_LIBRARY_PATH=` (clears split_and_splat's libs).

3. **Reference them by environment variable, not a hardcoded `/home/...` path.** Set once:
   ```bash
   export SAM3_ROOT=$HOME/sam3
   export AMODAL3R_ROOT=$HOME/Amodal3R
   ```
   (Recommended cleanup: replace the hardcoded `VOCAB`/`BPE` paths in `scripts/run_full_pipeline.sh`
   with `${SAM3_ROOT}/...` so the repo is portable.)

> Alternative: if you maintain a **fork** of either repo with your patches, add it as a `git submodule`
> under `third_party/` and pin the commit. Use submodules only for forks you own — for upstream-plus-patches,
> the commit-pin + patch-file approach above is cleaner.

## End-to-end run (2 stages)

```bash
# Stage 1 — re-labeling (sam3 env)
conda activate sam3
SCENE=replica_room0_v2 STRIDE=2 NPF=10 \
  SCENE_COLMAP=$HOME/RefineGS/data/replica_room0/sparse/0 \
  bash scripts/run_full_pipeline.sh relabel

# Stage 2 — amodal → per-object 2DGS → completion → eval (split_and_splat env)
conda activate split_and_splat
SCENE=replica_room0_v2 bash scripts/run_full_pipeline.sh recon
#   MAX=30  → train only the 30 best-observed objects (partial run)
```

Downstream steps are **standalone** and run on any finished subset (useful while a full run is still training,
or on a copied output dir to avoid interfering):

```bash
OUT=output/replica_room0_v2/refinegs_full
GT=$HOME/room_0/habitat/mesh_semantic.ply

# aggregate eval (recon vs observation-consistent completed)
python tools/eval_object_mesh.py batch --gt_mesh $GT \
  --recon_glob "${OUT}/*/train/ours_*/fuse_post.ply"      --label_from_path -4 --auto_match --out ${OUT}/_eval_recon.csv
python tools/eval_object_mesh.py batch --gt_mesh $GT \
  --recon_glob "${OUT}/*/train/ours_*/fuse_completed.ply" --label_from_path -4 --auto_match --out ${OUT}/_eval_completed.csv

# scene assembly (gaussian splat scene + geometry mesh)
python refinegs/assemble/scene_assemble.py --root ${OUT} \
  --out_gauss output/replica_room0_v2/scene_gaussians.ply \
  --out_mesh  output/replica_room0_v2/scene_mesh.ply
```

## Key results (Replica room0)

| Metric | Baseline | Ours |
|---|---|---|
| Instance discovery recall | 0.756 (SAM2+graph) | **0.919** (SAM3) |
| On-plane occlusion-hole surface vs GT | 33.8 mm | **4.3 mm** |
| Observation-consistency: violating surface | 26.0% (naive fusion) | **16.6%** (carve) |
| Observation-consistency: verified surface | 73% | **82%** |

Full-scene aggregate (recon vs completed over all objects) and instance-dense (Waldo Kitchen / ScanNet)
validation are produced by the commands above.

## Notes

- Per-object models share the **scene COLMAP** (`sparse/0`), so they live in one world frame and
  `scene_assemble.py` simply concatenates them — no extra alignment needed.
- The driver is **resumable**: already-trained objects are skipped.
- Design rationale and the experiment-by-experiment history live in `docs/`.
