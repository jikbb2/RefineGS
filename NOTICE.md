RefineGS
Copyright (c) 2026 Lee Byung Jik, Konkuk University

This product includes software developed by third parties, listed below.
See LICENSE.md for the terms that govern this repository as a whole.

--------------------------------------------------------------------------------
1. Gaussian Splatting  (Inria and Max Planck Institut fuer Informatik)
--------------------------------------------------------------------------------
This repository is derived from the 2D Gaussian Splatting codebase, which is
itself derived from the original gaussian-splatting implementation. The
Gaussian-Splatting License in LICENSE.md therefore governs this repository in
its entirety, including every file listed in sections 1 through 4 below.

    Research and evaluation use only. Commercial use requires prior written
    consent from Inria (stip-sophia.transfert@inria.fr).

  Upstream : https://github.com/hbb1/2d-gaussian-splatting
             https://github.com/graphdeco-inria/gaussian-splatting
  Papers   : B. Kerbl, G. Kopanas, T. Leimkuehler, G. Drettakis,
             "3D Gaussian Splatting for Real-Time Radiance Field Rendering,"
             ACM Trans. Graph. 42(4), 2023.
             B. Huang, Z. Yu, A. Chen, A. Geiger, S. Gao,
             "2D Gaussian Splatting for Geometrically Accurate Radiance Fields,"
             Proc. ACM SIGGRAPH, 2024.

Files inherited from upstream and MODIFIED in this repository (Section 4.2(b) of
the Gaussian-Splatting License):

    arguments/__init__.py
    gaussian_renderer/__init__.py
    scene/__init__.py
    scene/cameras.py
    scene/colmap_loader.py
    scene/dataset_readers.py
    scene/gaussian_model.py
    train.py
    utils/camera_utils.py
    utils/feature_extractor.py
    utils/loss_utils.py
    utils/mcube_utils.py
    utils/mesh_utils.py

Files inherited from upstream and used UNMODIFIED are not listed individually;
they remain under the Gaussian-Splatting License. One of them sits outside the
usual directories and is easy to miss: utils_mask/mask_filters.py, inherited from
Split&Splat and imported by train.py to drop empty views. Everything else that
came from Split&Splat and is not used by this pipeline is kept under legacy/ and
is described in section 4.

--------------------------------------------------------------------------------
2. CUDA extensions redistributed in this repository
--------------------------------------------------------------------------------
These are committed as ordinary files rather than as git submodules, so a plain
`git clone` is enough to build them; `git submodule update` is neither needed nor
useful here. Each is built with `pip install <path>`.

    submodules/diff-surfel-rasterization
        The 2D Gaussian Splatting rasteriser.
        Upstream: https://github.com/hbb1/diff-surfel-rasterization
        Its own LICENSE.md is the Gaussian-Splatting License (Inria and MPII),
        the same terms as this repository.

    submodules/simple-knn
        Upstream: https://gitlab.inria.fr/bkerbl/simple-knn
        Its own LICENSE.md is the Gaussian-Splatting License (Inria and MPII).

    submodules/diff-surfel-rasterization/third_party/glm
        OpenGL Mathematics (GLM), bundled inside the rasteriser and required to
        compile it. Its copying.txt states that GLM is licensed under The Happy
        Bunny License (a modified MIT licence) or the MIT License.
        Upstream: https://github.com/g-truc/glm

--------------------------------------------------------------------------------
3. Files original to this repository
--------------------------------------------------------------------------------
The following were written for this work. They are distributed under the same
Gaussian-Splatting License as the rest of the repository, because they form part
of a derivative Work under Section 4.2 of that license.

    run_refinegs.sh              pipeline driver (all stages)
    run_field_fusion_batch.sh    per-object prior generation, fusion and evaluation
    sdf_distill_depth.py         observation-prior grid fusion
    mesh_tsdf_views.py           per-object TSDF meshing
    make_shaper_input.py         conditioning-point extraction for the shape prior
    vote_labels.py               multi-view instance voting onto gaussians
    sam3_relabel_video.py        SAM 3 video relabeling with cross-window re-id
    amodal_mask.py               amodal mask generation
    prior_mesh.py                meshing of the generated signed field
    dump_scene_depth.py          scene-model depth dump for free-space carving
    eval_seen_unseen.py          seen / unseen / free evaluation protocol
    eval_instance_seg.py         3-D instance segmentation metrics
    agg_ab.py, render_ab.py      aggregation and qualitative rendering
    shaper/shaper_field.py       signed SDF grid extracted from ShapeR's decoder
    shaper/infer_shape_pinhole.py  pinhole-camera bypass for ShapeR's rectify step
    tools/                       dataset conversion and diagnostics
    audit_*.py, probe_*.py       dataset and vocabulary audits

The two files under shaper/ are worth calling out: they run against a ShapeR
checkout but are NOT part of ShapeR, and cloning ShapeR does not produce them.
They import ShapeR's packages through PYTHONPATH and read its checkpoints, and
nothing is written into the ShapeR tree.

--------------------------------------------------------------------------------
4. legacy/ -- inherited code retained but unused
--------------------------------------------------------------------------------
Nothing under legacy/ is called by this pipeline. It is kept so that the code
this work inherits stays visible alongside the attribution above, rather than
being deleted from the record. legacy/README.md says what is in it.

Three of its directories came from elsewhere and carry no license file in this
tree. Their terms are those of their own upstream projects, recorded here by the
commit this project installed them from; obtain the terms from those projects.

    legacy/sam2/
        A copy of Meta's SAM 2 source. This work uses SAM 3 (section 5), not
        SAM 2, and no file outside legacy/ imports it.
        Upstream: https://github.com/facebookresearch/sam2
        Recorded commit: 2b90b9f5ceec907a1c18123530e92e794ad901a4

    legacy/point_projection/
    legacy/refine_gs/
    legacy/utils_mask/ (all but mask_filters.py, which stays in section 1)
        Inherited from Split&Splat.
        Upstream: https://github.com/LTTM/Split_and_Splat
        Recorded commit: 9b7ba906511f75b87c438b4cf30ae3d1d5302f61

The remainder of legacy/ is this project's own superseded drivers, experiments,
probes and audits, under the same license as section 3.

--------------------------------------------------------------------------------
5. External models, code and datasets (not redistributed here)
--------------------------------------------------------------------------------
The pipeline calls these at run time. Obtain each from its own source and comply
with its own license; no weights or data from any of them are included in this
repository.

    SAM 3      N. Carion et al., "SAM 3: Segment Anything with Concepts,"
               arXiv:2511.16719, 2025.
               https://github.com/facebookresearch/sam3

    ShapeR     Y. Siddiqui et al., "ShapeR: Robust Conditional 3D Shape
               Generation from Casual Captures," arXiv:2601.11514, 2026.
               https://github.com/facebookresearch/ShapeR

    COLMAP     J. L. Schoenberger, J.-M. Frahm, "Structure-from-Motion
               Revisited," Proc. IEEE CVPR, 2016.

    Replica    J. Straub et al., "The Replica Dataset: A Digital Replica of
               Indoor Spaces," arXiv:1906.05797, 2019.
               Camera trajectories follow NICE-SLAM:
               Z. Zhu et al., Proc. IEEE/CVF CVPR, 2022.

    fused-ssim OPTIONAL. train.py imports it inside a try/except and falls back
               to the reference SSIM when it is absent, so it is a speed
               optimisation rather than a dependency. Obtained from Split&Splat
               (commit above), which vendors it.
