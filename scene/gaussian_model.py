#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        # self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)

        # pure color (RGB, 0~1)
        self._albedo = torch.empty(0)

        # Shading MLP
        self.shading_mlp = ShadingMLP(hidden_dim=64).cuda()

        # ── Stage 2: Semantic feature per Gaussian ──────────────
        self._semantic_feature = torch.empty(0)
        self.feature_dim = 32          # must match feature_extractor --feature_dim
        # ────────────────────────────────────────────────────────

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_albedo(self):
        return torch.sigmoid(self._albedo)

    # ── Stage 2: semantic feature property ──────────────────────
    @property
    def get_semantic_feature(self):
        """Raw feature vector per Gaussian  [N, feature_dim]."""
        return self._semantic_feature
    # ────────────────────────────────────────────────────────────

    @property
    def get_normal(self):
        rot = build_rotation(self._rotation)  # [N, 3, 3]
        return F.normalize(rot[:, :, 2], dim=-1)  # [N, 3]

    def get_shading(self, normals, positions, view_dirs):
        """_Calculate shading factor of each gaussian_
        Args:
            normals: [N, 3]
            positions: [N, 3]
            view_dirs: [N, 3]
        Returns:
            shading: [N, 1]
        """
        return self.shading_mlp(normals, positions, view_dirs)

    def compute_color(self, camera):
        positions = self.get_xyz           # [N, 3]
        normals = self.get_normal          # [N, 3]
        albedo = self.get_albedo           # [N, 3]
        cam_pos = camera.camera_center     # [3]
        view_dirs = F.normalize(
            positions - cam_pos[None, :],
            dim=-1
        )  # [N, 3]
        shading = self.get_shading(normals, positions, view_dirs)  # [N, 1]
        self._cached_shading = shading.detach()
        color = albedo * shading  # [N, 3]
        return color

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            # self._features_dc,
            # self._features_rest,
            self._albedo,
            self._semantic_feature,        # ← Stage 2 추가
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.shading_mlp.state_dict()
        )

    def restore(self, model_args, training_args):
        # old checkpoints (Stage 1) don't have _semantic_feature → handle gracefully
        if len(model_args) == 13:
            # Stage 2 checkpoint (with semantic_feature)
            (self.active_sh_degree,
             self._xyz,
             self._albedo,
             self._semantic_feature,       # ← Stage 2 추가
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             mlp_state_dict) = model_args
        else:
            # Stage 1 checkpoint (13 items, no semantic_feature) → init to zeros
            (self.active_sh_degree,
             self._xyz,
             self._albedo,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             mlp_state_dict) = model_args
            n = self._xyz.shape[0]
            self._semantic_feature = nn.Parameter(
                torch.zeros(n, self.feature_dim, device="cuda").requires_grad_(True)
            )

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shading_mlp.load_state_dict(mlp_state_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32).cuda()
        colors = torch.clamp(colors, 1e-5, 1.0 - 1e-5)
        albedo_init = torch.log(colors / (1.0 - colors))
        self._albedo = nn.Parameter(albedo_init.requires_grad_(True))

        # ── Stage 2: init semantic features to zero ──────────────
        n_points = fused_point_cloud.shape[0]
        self._semantic_feature = nn.Parameter(
            torch.zeros(n_points, self.feature_dim, dtype=torch.float32,
                        device="cuda").requires_grad_(True)
        )
        # ─────────────────────────────────────────────────────────

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._albedo], 'lr': 0.005, 'name': 'albedo'},
            {'params': self.shading_mlp.parameters(), 'lr': 0.001, 'name': 'shading_mlp'},
            # ── Stage 2 ──────────────────────────────────────────
            {'params': [self._semantic_feature], 'lr': 0.001, 'name': 'semantic_feature'},
            # ─────────────────────────────────────────────────────
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append(f"albedo_{i}")
        # ── Stage 2: semantic feature columns ────────────────────
        for i in range(self.feature_dim):
            l.append(f"feat_{i}")
        # ─────────────────────────────────────────────────────────
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz      = self._xyz.detach().cpu().numpy()
        normals  = np.zeros_like(xyz)
        albedo   = self._albedo.detach().cpu().numpy()
        # ── Stage 2 ──────────────────────────────────────────────
        sem_feat = self._semantic_feature.detach().cpu().numpy()   # [N, 32]
        # ─────────────────────────────────────────────────────────
        opacities = self._opacity.detach().cpu().numpy()
        scale     = self._scaling.detach().cpu().numpy()
        rotation  = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements   = np.empty(xyz.shape[0], dtype=dtype_full)
        # order must match construct_list_of_attributes
        attributes = np.concatenate(
            (xyz, normals, albedo, sem_feat, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        albedo_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        assert len(albedo_names) == 3
        albedo = np.zeros((xyz.shape[0], 3))
        for idx, attr_name in enumerate(albedo_names):
            albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # ── Stage 2: load semantic features (backward-compatible) ─
        feat_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("feat_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        if feat_names:
            sem_feat = np.zeros((xyz.shape[0], len(feat_names)))
            for idx, attr_name in enumerate(feat_names):
                sem_feat[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            # old PLY (Stage 1) → init to zeros
            sem_feat = np.zeros((xyz.shape[0], self.feature_dim))
        # ─────────────────────────────────────────────────────────

        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._albedo = nn.Parameter(
            torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        # ── Stage 2 ──────────────────────────────────────────────
        self._semantic_feature = nn.Parameter(
            torch.tensor(sem_feat, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        # ─────────────────────────────────────────────────────────
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"]    = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "shading_mlp":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"]    = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz             = optimizable_tensors["xyz"]
        self._albedo          = optimizable_tensors["albedo"]
        self._opacity         = optimizable_tensors["opacity"]
        self._scaling         = optimizable_tensors["scaling"]
        self._rotation        = optimizable_tensors["rotation"]
        # ── Stage 2 ──────────────────────────────────────────────
        self._semantic_feature = optimizable_tensors["semantic_feature"]
        # ─────────────────────────────────────────────────────────
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom              = self.denom[valid_points_mask]
        self.max_radii2D        = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "shading_mlp":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self, new_xyz, new_albedo, new_opacities, new_scaling, new_rotation,
        new_semantic_feature,       # ← Stage 2 추가
    ):
        d = {
            "xyz":              new_xyz,
            "albedo":           new_albedo,
            "opacity":          new_opacities,
            "scaling":          new_scaling,
            "rotation":         new_rotation,
            # ── Stage 2 ──────────────────────────────────────────
            "semantic_feature": new_semantic_feature,
            # ─────────────────────────────────────────────────────
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz              = optimizable_tensors["xyz"]
        self._albedo           = optimizable_tensors["albedo"]
        self._opacity          = optimizable_tensors["opacity"]
        self._scaling          = optimizable_tensors["scaling"]
        self._rotation         = optimizable_tensors["rotation"]
        # ── Stage 2 ──────────────────────────────────────────────
        self._semantic_feature = optimizable_tensors["semantic_feature"]
        # ─────────────────────────────────────────────────────────
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom              = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D        = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )
        stds    = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds    = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means   = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots    = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz      = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling  = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_albedo   = self._albedo[selected_pts_mask].repeat(N, 1)
        new_opacity  = self._opacity[selected_pts_mask].repeat(N, 1)
        # ── Stage 2: inherit parent feature ──────────────────────
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N, 1)
        # ─────────────────────────────────────────────────────────
        self.densification_postfix(
            new_xyz, new_albedo, new_opacity, new_scaling, new_rotation,
            new_semantic_feature,
        )
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        new_xyz              = self._xyz[selected_pts_mask]
        new_albedo           = self._albedo[selected_pts_mask]
        new_opacities        = self._opacity[selected_pts_mask]
        new_scaling          = self._scaling[selected_pts_mask]
        new_rotation         = self._rotation[selected_pts_mask]
        # ── Stage 2: inherit parent feature ──────────────────────
        new_semantic_feature = self._semantic_feature[selected_pts_mask]
        # ─────────────────────────────────────────────────────────
        self.densification_postfix(
            new_xyz, new_albedo, new_opacities, new_scaling, new_rotation,
            new_semantic_feature,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1


class ShadingMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, normal, position, view_dir):
        """
        Args:
            normal:   [N, 3]
            position: [N, 3]
            view_dir: [N, 3]
        Returns:
            shading:  [N, 1]
        """
        x = torch.cat([normal, position, view_dir], dim=-1)  # [N, 9]
        return self.net(x)
