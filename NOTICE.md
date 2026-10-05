RefineGS
Copyright (c) 2026 Lee Byung Jik, Konkuk University

This product includes software developed by third parties, listed below.
See LICENSE.md for the terms that govern this repository as a whole.

--------------------------------------------------------------------------------
1. Gaussian Splatting  (Inria and Max Planck Institut fuer Informatik)
--------------------------------------------------------------------------------
This repository is derived from the 2D Gaussian Splatting codebase, which is
itself derived from the original gaussian-splatting implementation. The
Gaussian-Splatting License in LICENSE.md therefore governs this repository,
including every file listed under "Modified files" below.

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

Files inherited from upstream and MODIFIED in this repository:

    train.py              <describe the change in one line>
    convert.py            <describe the change in one line>
    scene/                <describe the change in one line>
    arguments/            <describe the change in one line>
    ...                   (list every upstream file you edited)

Files inherited from upstream and used UNMODIFIED are not listed individually;
they remain under the Gaussian-Splatting License.

--------------------------------------------------------------------------------
2. Files original to this repository
--------------------------------------------------------------------------------
The following were written for this work. They are distributed under the same
Gaussian-Splatting License as the rest of the repository, because they are part
of a derivative Work under Section 4.2 of that license.

    run_refinegs.sh            pipeline driver
    run_field_fusion_batch.sh  per-object fusion and evaluation batch
    sdf_distill_depth.py       observation-prior grid fusion
    mesh_tsdf_views.py         per-object TSDF meshing
    vote_labels.py             multi-view instance voting onto gaussians
    eval_seen_unseen.py        seen / unseen / free evaluation protocol
    eval_instance_seg.py       3-D instance segmentation metrics
    agg_ab.py, render_ab.py    aggregation and qualitative rendering
    sam3_relabel_video.py      SAM 3 video relabeling with cross-window re-id
    amodal_mask.py             amodal mask generation
    tools/, audit_*, probe_*   dataset conversion and diagnostics

--------------------------------------------------------------------------------
3. External models and datasets (not redistributed here)
--------------------------------------------------------------------------------
The pipeline calls these at run time. Obtain each from its own source and comply
with its own license; none of their weights or data are included in this
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