# RefineGS

Object-level surface completion for indoor scenes. A scene is reconstructed with 2D Gaussian
Splatting, decomposed into objects by SAM 3 video re-labeling and multi-view voting, and each
object's unobserved surface is completed from a generative shape prior (ShapeR). The prior is
taken as a **signed field** rather than a mesh and fused with the observed TSDF per voxel, so
the completion fills what the cameras never saw without moving the surface they did see.

Everything downstream of the scene model runs per object, and the evaluation reports *seen*,
*unseen* and *free* space separately — the point of the method is what happens in the unseen
part, and a single averaged number hides it.

---

## Repository layout

```
run_refinegs.sh              the driver. One scene name -> numbers. Every stage is here.
run_field_fusion_batch.sh    per-object prior generation, grid fusion and evaluation
sdf_distill_depth.py         the fusion itself (observation x prior, per voxel)
eval_seen_unseen.py          the seen / unseen / free evaluation protocol
sam3_relabel_video.py        SAM 3 video re-labeling with cross-window re-identification
shaper/                      our ShapeR driver scripts (see "ShapeR" below)
submodules/                  CUDA extensions, committed as ordinary files
tools/                       dataset conversion and diagnostics
environment/                 conda environment specifications
legacy/                      inherited and superseded code, not used. See legacy/README.md.
```

---

## Requirements

A CUDA GPU. The published runs used PyTorch 2.5.1 with CUDA 12.4 and Python 3.12, as pinned in
`environment/refinegs.yml`. Three conda environments are needed, because SAM 3 and this
project cannot share a process (conflicting cuDNN), and ShapeR pins its own stack:

| environment | built from | used by |
|---|---|---|
| `refinegs` | `environment/refinegs.yml` | everything except the two stages below |
| `sam3` | `environment/sam3.yml` + [SAM 3](https://github.com/facebookresearch/sam3) | the `relabel` stage |
| `shaper` | `environment/shaper.yml` + [ShapeR](https://github.com/facebookresearch/ShapeR) | the `field` stage |

The driver enters the right environment per stage; you do not switch by hand.

---

## Install

```bash
git clone <this repository> RefineGS
cd RefineGS

conda env create -f environment/refinegs.yml
conda activate refinegs

pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```

`submodules/` is committed as ordinary files, not as git submodules, so a plain `git clone`
is enough — there is no `git submodule update` step and running one does nothing.

Two things that will bite otherwise:

- **Do not run `conda activate` under `set -u`.** This environment carries the conda
  cross-compiler and CUDA toolchain packages, whose activation hooks reference variables
  without defaults; under `set -u` every one of them errors and the compiler variables end up
  unset, which breaks the extension builds above. If you see a wall of
  `CONDA_BACKUP_...: unbound variable`, that is this.
- `fused-ssim` is optional. `train.py` imports it in a `try/except` and falls back to the
  reference SSIM, so it is a speed optimisation, not a dependency.

Then set up the two external repositories and their environments as their own instructions
say, and point this project at them:

```bash
conda env create -f environment/sam3.yml      # then install SAM 3 into it
conda env create -f environment/shaper.yml    # then install ShapeR into it
export SHAPER_DIR=$HOME/ShapeR                # the ShapeR checkout, with its checkpoints/
```

### ShapeR

`shaper/shaper_field.py` and `shaper/infer_shape_pinhole.py` belong to **this** repository.
Cloning ShapeR does not produce them, and they do not need to be copied into the ShapeR tree.
They run against your ShapeR checkout: `SHAPER_DIR` is put on `PYTHONPATH` for ShapeR's own
packages, and `shaper_field.py` changes into it so ShapeR finds `checkpoints/` where it
expects. Nothing is written into the ShapeR tree.

If you keep them somewhere else, point `SHAPER_PY` at `shaper_field.py`. Keep only those two
files in that directory — a `dataset.py` or `model.py` beside them would shadow ShapeR's own
packages.

---

## Data

The published experiments use Replica, from two separate downloads:

```bash
export REPLICA_ROOT=$HOME/nice-slam/Datasets/Replica   # frames, depth, traj.txt
export REPLICA_SEMANTIC=$HOME/replica_dl               # parent of <scene>/habitat/
```

- **`REPLICA_ROOT`** is the NICE-SLAM Replica dump: per-scene `results/` (RGB `frame*.jpg`
  and depth `depth*.png`) and `traj.txt`.
- **`REPLICA_SEMANTIC`** is the parent of the original Replica release's `habitat/`
  directories. Only two files from each are read: `mesh_semantic.ply` and
  `info_semantic.json`. The scene naming differs between the two downloads (`room1` versus
  `room_1`); the driver tries both.

`REPLICA_SEMANTIC` is **optional**. Without it the reconstruction runs and the evaluation is
unavailable, which is the correct behaviour for a dataset that has no semantic ground truth —
see "Other datasets" below.

---

## Running

```bash
export REPLICA_ROOT=$HOME/nice-slam/Datasets/Replica
export REPLICA_SEMANTIC=$HOME/replica_dl

bash run_refinegs.sh room0 --dry     # resolve and print every path, run nothing
bash run_refinegs.sh room0           # the whole pipeline
```

Start with `--dry`. It resolves every path, prints the run manifest, checks that the inputs
each stage in the span actually reads are present, and exits without creating anything.

Everything after the scene name is `VAR=value` or `--dry`:

```bash
bash run_refinegs.sh room0 FROM=mesh TO=fuse          # run a span of stages
bash run_refinegs.sh room0 ONLY="6 14" FROM=cond      # only these object ids
RUN=myrun bash run_refinegs.sh room0 FROM=fuse        # tag the outputs
bash run_refinegs.sh                                  # no scene name: all paths from the environment
```

A scene name is authoritative: the GT paths are recomputed from it and a value left over in
your shell is ignored and reported, because a stale `GT_MESH` evaluates one room against
another room's ground truth and the numbers still look plausible. An explicit `VAR=value`
**on the command line** is a deliberate override and still wins.

### Stages

`FROM` and `TO` select a span of this list.

| stage | what it does | environment |
|---|---|---|
| `stage0` | Replica dump → `data/<scene>`, frame links | refinegs |
| `colmap` | a pose for every frame | refinegs |
| `relabel` | SAM 3 video instances → per-object masks | **sam3** |
| `masks` | amodal masks, per-object folders | refinegs |
| `labels` | per-view label maps, `id_map.json` | refinegs |
| `train` | the scene 2DGS model | refinegs |
| `carve` | rendered scene depth, the free-space reference | refinegs |
| `objects` | vote labels onto gaussians and slice out objects | refinegs |
| `name` | a caption per object → `names.tsv` | refinegs |
| `mesh` | per-object TSDF → `fuse_post.ply` (**side A**, the observed surface) | refinegs |
| `cond` | the conditioning surface → `tsdf_clean.ply` | refinegs |
| `pkl` | ShapeR input per object | refinegs |
| `field` | ShapeR signed SDF grid | **shaper** |
| `fuse` | grid fusion (**side B**) and evaluation | refinegs |
| `eval` | re-evaluate the fused meshes and nothing else | refinegs |

`eval` is separate from `fuse` on purpose: the field and the fusion are the expensive parts
and neither depends on how a reconstruction is matched to its GT instance, so a change to the
matching costs minutes rather than hours.

### Where things go

```
data/<scene>/      images, poses, masks, per-view labels
output/<scene>/
  scene/           the scene 2DGS model
  carve_depth/     rendered scene depth
  objects_voted/   per-object gaussians, meshes and names.tsv
  prior/<pipeline>/  ShapeR pkl inputs and field npz files
  runs/<RUN>/      manifest.txt, results.csv, per-object logs
```

`<scene>` is `replica_<name>` for a Replica scene name. Every run writes a `manifest.txt`
recording the resolved paths, the stage span, the git revision and the settings that do not
otherwise appear in any file — including the training flags, which a trained model does not
record about itself.

---

## Options worth knowing

| variable | default | what it decides |
|---|---|---|
| `PIPELINE` | `scene` | `scene` slices one trained model; `perobj` trains each object |
| `DEPTH_SUPERVISION` | `none` | GT depth loss in scene training. The published runs used `none` |
| `OBJ_DEPTH_SUPERVISION` | `gt` | the same choice for per-object models |
| `VOTE_REF` | `gt` | first-surface reference for the vote: `gt` depth or `carve` (rendered) |
| `ENSEMBLE` / `COMBINE` | `3` / `best` | how many prior samples per object, and how they are combined |
| `GRID` | `256` | the prior field's grid resolution |
| `CLEAN` | `0` | retire this stage's outputs and everything after it, then rebuild |
| `--dry` | — | resolve and check everything, create nothing |

A result is reused only when it is newer than both its generator and its inputs, and stages
whose behaviour lives in a flag rather than a file also compare a stored copy of that flag. So
changing an argument invalidates the stage that used it, rather than silently reusing a file
built to answer a different question.

### The scene training flags

The `train` stage stops rather than guessing:

```
[STOP] set TRAIN_SCENE_ARGS (label-embedding and resolution flags).
       Guessing them would train for hours and produce the wrong model.
```

The scene models behind the published results were trained with a label directory and GT
depth supervision. Reproduce that with:

```bash
bash run_refinegs.sh office0 \
  TRAIN_SCENE_ARGS="--label_dir $PWD/data/replica_office0/labels_scene" \
  DEPTH_SUPERVISION=gt
```

`--label_dir` gives `train.py` the per-view label maps the `labels` stage wrote; the `objects`
stage later needs that embedding to vote instance labels onto the gaussians, so a scene model
trained without it cannot be decomposed. **Point it at this scene's `labels_scene`** — the
path is not derived for you, and a stale one trains for hours against another scene's labels.

`DEPTH_SUPERVISION=gt` appends `--gt_depth_dir <GTD> --lambda_gtdepth 0.5`, which is what the
published models were trained with. Leave it at the default `none` to train without the depth
loss, so that the reconstruction claim does not rest on ground truth; a model trained that way
is what `output/<scene>/scene_nodepth` holds in our runs.

Either way the flags are recorded in that run's `manifest.txt`, because a trained model stores
only its `ModelParams` and carries no record of the loss configuration it was trained under.
That manifest line is the only place the two variants above are distinguishable after the
fact.

---

## Other datasets

The driver no longer assumes Replica: dataset roots are inputs, each ground-truth path is
demanded only by the stages that actually read it, and the places that used to fail silently
now stop. What it does **not** yet have is a replacement for the two things Replica supplies.

**Per-frame depth is required by the `relabel` stage.** The 3-D signature that unifies SAM 3's
instance tracks is a back-projection of the mask pixels through a depth map. Without depth
every track falls below `--min_sig` and is discarded; the driver now checks for it up front
and aborts rather than reporting zero objects and exiting successfully. Rendered scene depth
cannot stand in, because `relabel` runs before `train`. A dataset with sensor depth (ScanNet,
for instance) satisfies this directly:

```bash
DEPTH_DIR=/path/to/depth DEPTH_FROM=frame DEPTH_TO=depth DEPTH_EXT=.png DEPTH_SCALE=1000 \
  bash run_refinegs.sh ...
```

A capture with no depth at all needs a depth source — MVS or monocular — produced before
`relabel`. That step is not part of this repository.

**Evaluation requires the Replica semantic mesh.** `eval_seen_unseen.py` extracts the target
object by per-face `object_id` and uses the same mesh as its visibility oracle. Without a
semantic mesh the reconstruction and the fusion still run, and the seen/unseen metrics are
skipped with a note.

**Captions.** Object captions condition the generation; generating every object from "a 3D
object in a room" measurably degrades the result, so a run without captions stops unless you
ask for it with `ALLOW_GENERIC_CAPTIONS=1`. With a Replica semantic mesh they come from the GT
classes; otherwise they come from the SAM 3 concept each object was segmented from, written by
`sam3_relabel_video.py` as `concepts.json` in the relabel output.

Still Replica-specific, and not addressed here: the `traj.txt` pose format, the fixed camera
intrinsics, and the `6553.5` depth scale assumed by several tools outside the `relabel` stage.

---

## License and attribution

This repository is a derivative work of 2D Gaussian Splatting, which derives from
gaussian-splatting. The **Gaussian-Splatting License** in `LICENSE.md` governs all of it:
research and evaluation use only, with commercial use requiring prior written consent from
Inria.

`NOTICE` lists what comes from where — the upstream files modified here, the CUDA extensions
redistributed under `submodules/`, the inherited code kept under `legacy/`, and the external
models and datasets the pipeline calls but does not redistribute. Each of those carries its
own license; obtain and comply with them separately.
