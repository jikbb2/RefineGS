#
# RefineGS - scene/gaussian_model.py
# ---------------------------------------------------------------------------
# BASE:  2D Gaussian Splatting (hbb1/2d-gaussian-splatting) — Inria GRAPHDECO
# GRAFT: Split&Splat (LTTM/Split_and_Splat) instance layer
#
# 머지 원칙
#   - 2DGS 고유 geometry는 전부 보존:
#       * 2D scaling: create_from_pcd 의 scales = ...repeat(1, 2)
#       * surfel covariance: build_covariance_from_scaling_rotation(center, ...)
#       * densify_and_split 의 stds 3D 패딩
#       * add_densification_stats 의 전체 grad norm (3DGS 의 [...,:2] 아님)
#   - Split&Splat 인스턴스 레이어만 graft:
#       * _id (N,3), _desc_test (N,384)
#       * get_id / get_desc / get_id_color / get_black / get_no_opacity
#       * filter_by_id / filter_points
#       * save_large_ply (청크 바이너리, desc 384차원 대응)
#       * create_from_pcd(color_id=...)
#       * capture/restore/prune/densify 에 id/desc 엮기
#
# 제거된 3DGS-mip 기능 (2DGS 베이스와 정합 위해)
#   - exposure 서브시스템 (_exposure, exposure_optimizer, get_exposure_from_name)
#   - SparseGaussianAdam (2DGS 는 plain Adam)
#
# 변경 지점은 모두 "# [S&S]" 주석으로 표시.
# ---------------------------------------------------------------------------

import copy                                   # [S&S] filter_by_id / filter_points 의 deepcopy
import os
import struct                                 # [S&S] save_large_ply
import numpy as np
import torch
from torch import nn

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


# ---------------------------------------------------------------------------
# [S&S] 청크 바이너리 PLY 라이터
#   desc(384차원) 때문에 plyfile.PlyElement 일괄 변환이 무거워 S&S 는
#   직접 바이너리로 스트리밍한다. RefineGS 도 동일하게 사용.
#   concat 순서는 construct_list_of_attributes() 의 순서와 반드시 일치해야 함:
#     xyz, normals, f_dc, f_rest, ids, desc, opacities, scale, rotation
# ---------------------------------------------------------------------------
def save_large_ply(path, xyz, normals, f_dc, f_rest, ids, desc,
                   opacities, scale, rotation, attribute_names, chunk=150000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = xyz.shape[0]

    floats_per_vertex = (
        xyz.shape[1] + normals.shape[1] +
        f_dc.shape[1] + f_rest.shape[1] +
        ids.shape[1] + desc.shape[1] +
        opacities.shape[1] + scale.shape[1] + rotation.shape[1]
    )
    vertex_struct = struct.Struct("<" + "f" * floats_per_vertex)

    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for attr in attribute_names:
            f.write(f"property float {attr}\n".encode())
        f.write(b"end_header\n")

    with open(path, "ab") as f:
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            combined = np.concatenate((
                xyz[i:j], normals[i:j], f_dc[i:j], f_rest[i:j],
                ids[i:j], desc[i:j], opacities[i:j], scale[i:j], rotation[i:j],
            ), axis=1)
            for row in combined:
                f.write(vertex_struct.pack(*row.astype(np.float32)))
    return


class GaussianModel:

    def setup_functions(self):
        # [2DGS] surfel covariance: scaling(2D) 에 1 을 패딩해 4x4 변환 행렬 구성
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(
                torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                rotation,
            ).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    # [S&S] active_sh_degree / optimizer_type 인자 추가 (composition 시 sh_degree=3 등)
    def __init__(self, sh_degree: int, active_sh_degree: int = 0, optimizer_type: str = "default"):
        self.active_sh_degree = active_sh_degree
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._id = torch.empty(0)            # [S&S] 인스턴스 id (N,3)
        self._desc_test = torch.empty(0)     # [S&S] CLIP 디스크립터 (N,384)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._id,            # [S&S]
            self._desc_test,     # [S&S]
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._id,             # [S&S]
         self._desc_test,      # [S&S]
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # --------------------------- properties ---------------------------
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # [2DGS] 2D scaling

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_no_opacity(self):
        # [S&S] full-opacity 렌더(α=1)용 — mask reprojection(3.2.2)
        return self.opacity_activation(torch.zeros_like(self._opacity) + 1.0)

    # [S&S] 인스턴스 id / descriptor 접근자
    @property
    def get_id(self):
        return self._id

    @property
    def get_desc(self):
        return self._desc_test

    @property
    def get_id_color(self):
        # id 를 색(DC)으로 인코딩한 feature → 2-pass 마스크 렌더링에 사용
        features_dc = self._id.unsqueeze(1)                 # (N,1,3)
        features_rest = torch.zeros_like(self._features_rest)  # (N,K,3)
        return torch.cat((features_dc, features_rest), dim=1)

    def get_black(self, black_th=-1.75):
        # [S&S] occlusion(검은) 가우시안 개수
        is_black = (self._features_dc < black_th).all(dim=-1).squeeze(-1)
        return is_black.sum().item()

    def set_desc(self, idx, desc):
        self._desc_test[idx] = desc

    def get_covariance(self, scaling_modifier=1):
        # [2DGS] center 전달
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # --------------------------- init ---------------------------
    # [S&S] color_id 인자 추가 (per-object 재구성 시 인스턴스 색 지정)
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, color_id=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)  # [2DGS] 2D scaling
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")  # [2DGS] random init

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # [S&S] 인스턴스 id / descriptor 초기화
        if color_id is None:
            color_id = [0.0, 0.0, 0.0]
        self._id = torch.tensor(color_id, dtype=torch.float, device="cuda").repeat(self._xyz.shape[0], 1)
        self._desc_test = torch.full((self._xyz.shape[0], 384), float('nan'), device="cuda")
        print("ID: R: %f G: %f B: %f" % (self._id[0][0].item(), self._id[0][1].item(), self._id[0][2].item()))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        # [2DGS] plain Adam (SparseGaussianAdam 제거)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._id.shape[1]):              # [S&S]
            l.append('id_{}'.format(i))
        for i in range(self._desc_test.shape[1]):       # [S&S]
            l.append('desc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):         # [2DGS] = 2 (scale_0, scale_1)
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # [2DGS] normal 은 rotation 에서 렌더 시 유도되므로 PLY 에는 zeros 저장(공식 2DGS 동일)
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ids = self._id.detach().cpu().numpy()             # [S&S]
        descs = self._desc_test.detach().cpu().numpy()    # [S&S]
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attrs = self.construct_list_of_attributes()
        # [S&S] 청크 바이너리 라이터 (desc 384차원 대응)
        save_large_ply(path, xyz, normals, f_dc, f_rest, ids, descs,
                       opacities, scale, rotation, attrs, chunk=150000)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # [S&S] id / descriptor 파싱
        ids = np.stack((np.asarray(plydata.elements[0]["id_0"]),
                        np.asarray(plydata.elements[0]["id_1"]),
                        np.asarray(plydata.elements[0]["id_2"])), axis=1)
        desc_keys = [k for k in plydata.elements[0].data.dtype.names if k.startswith("desc_")]
        desc_keys = sorted(desc_keys, key=lambda x: int(x.split("_")[-1]))
        desc_test = np.stack([np.asarray(plydata.elements[0][k]) for k in desc_keys], axis=1)

        # [S&S] f_rest 개수로 SH degree 자동 판정 (per-object 는 degree 0 일 수 있음)
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        if len(extra_f_names) == 0:
            self.max_sh_degree = 0
        elif len(extra_f_names) == 9:
            self.max_sh_degree = 1
        elif len(extra_f_names) == 24:
            self.max_sh_degree = 2
        elif len(extra_f_names) == 45:
            self.max_sh_degree = 3
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))  # [2DGS] 2개여야 정상
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._id = torch.tensor(ids, dtype=torch.float, device="cuda")            # [S&S]
        self._desc_test = torch.tensor(desc_test, dtype=torch.float, device="cuda")  # [S&S]

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # --------------------------- optimizer helpers ---------------------------
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._id = self._id[valid_points_mask]                  # [S&S]
        self._desc_test = self._desc_test[valid_points_mask]    # [S&S]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # [S&S] new_id / new_desc 인자 추가
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_id, new_desc, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._id = torch.cat((self._id, new_id))                  # [S&S]
        self._desc_test = torch.cat((self._desc_test, new_desc))  # [S&S]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        # [2DGS] 2D scaling → 3D 샘플링 위해 세 번째 축에 0 패딩
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_id = self._id[selected_pts_mask].repeat(N, 1)             # [S&S]
        new_desc = self._desc_test[selected_pts_mask].repeat(N, 1)    # [S&S]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_id, new_desc, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_id = self._id[selected_pts_mask]             # [S&S]
        new_desc = self._desc_test[selected_pts_mask]    # [S&S]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_id, new_desc, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # [2DGS] grad 전체 norm (3DGS 의 [...,:2] 아님)
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # --------------------------- [S&S] instance filters ---------------------------
    def filter_by_id(self, obj_id, keep_occlusions=False):
        """인스턴스 id 로 가우시안 선택. keep_occlusions=False 면 비대상은 사실상 제거(α↓, 색 어둡게)."""
        if not torch.is_tensor(obj_id):
            obj_id = torch.tensor(obj_id, dtype=self._id.dtype, device=self._id.device)
        if obj_id.ndim == 1:
            obj_id = obj_id.unsqueeze(0)

        mask = (self._id == obj_id).all(dim=1).to(self._id.device)
        masked_gs = copy.deepcopy(self)
        with torch.no_grad():
            if keep_occlusions:
                masked_gs._id[~mask] = torch.zeros_like(self._id[~mask]) - 5
            else:
                masked_gs._opacity[~mask] = torch.zeros_like(self._opacity[~mask]) - 20
                masked_gs._features_dc[~mask, :3] = -10
                masked_gs._id[~mask] = torch.zeros_like(self._id[~mask]) - 20
        return masked_gs

    def filter_points(self, black_th=-1.75, alpha_th=4.5):
        """검은/투명 floater 제거된 GaussianModel 복사본 반환."""
        f_dc = self._features_dc.detach().cpu().transpose(1, 2).squeeze().numpy()  # (N,3)
        opacity = self._opacity.detach().cpu().squeeze().numpy()                   # (N,)
        is_black = np.all(f_dc < black_th, axis=1)
        keep_mask = ~is_black
        f_dc = f_dc[keep_mask]
        opacity = opacity[keep_mask]
        is_transparent = opacity > alpha_th
        keep_mask2 = ~is_transparent
        final_mask = np.zeros(self._xyz.shape[0], dtype=bool)
        idx = np.where(~is_black)[0]
        final_mask[idx[keep_mask2]] = True

        filtered_gs = copy.deepcopy(self)
        with torch.no_grad():
            keep = torch.from_numpy(final_mask).to(self._xyz.device)
            filtered_gs._xyz = filtered_gs._xyz[keep]
            filtered_gs._features_dc = filtered_gs._features_dc[keep]
            filtered_gs._features_rest = filtered_gs._features_rest[keep]
            filtered_gs._scaling = filtered_gs._scaling[keep]
            filtered_gs._rotation = filtered_gs._rotation[keep]
            filtered_gs._opacity = filtered_gs._opacity[keep]
            filtered_gs._id = filtered_gs._id[keep]
            filtered_gs._desc_test = filtered_gs._desc_test[keep]
            if filtered_gs.max_radii2D.shape[0] == self._xyz.shape[0]:
                filtered_gs.max_radii2D = filtered_gs.max_radii2D[keep]
            if hasattr(filtered_gs, 'xyz_gradient_accum') and filtered_gs.xyz_gradient_accum.shape[0] == self._xyz.shape[0]:
                filtered_gs.xyz_gradient_accum = filtered_gs.xyz_gradient_accum[keep]
            if hasattr(filtered_gs, 'denom') and filtered_gs.denom.shape[0] == self._xyz.shape[0]:
                filtered_gs.denom = filtered_gs.denom[keep]
        return filtered_gs
