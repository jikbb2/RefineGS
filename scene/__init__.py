#
# RefineGS - scene/__init__.py
# ---------------------------------------------------------------------------
# BASE: Split&Splat scene/__init__.py (composition/mask/instance 로직 포함)
# 적용된 RefineGS 수정 (merged gaussian_model 와 정합):
#   [fix1] create_from_pcd 호출에서 cam_infos 인자 제거 → (pcd, spatial_lr_scale, color_id=)
#   [fix2] load_ply 호출 3곳에서 두 번째 인자 제거 → load_ply(path)
#   [fix3] save() 의 exposure 블록 제거 (exposure 서브시스템 미사용)
#
# ※ 2DGS 버전(3-인자 Colmap 호출)을 이 파일로 교체할 것.
# ---------------------------------------------------------------------------

from scene.cameras import Camera
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import time
import numpy as np
from utils.sh_utils import RGB2SH


class Scene:

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None,
                 shuffle=True, resolution_scales=[1.0]):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.composition = args.composition
        is_blender = False

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.black_cameras = {}

        print(os.path.join(args.source_path, "sparse"))
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.depths, args.eval,
                args.train_test_exp, args.is_instance)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.depths, args.eval)
            is_blender = True
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, \
                 open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        mask_color = None
        if not self.composition:
            np.random.seed(int(time.time()))
            mask_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale], self.black_cameras[resolution_scale] = \
                cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args,
                                         scene_info.is_nerf_synthetic, False, mask_color=mask_color)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale], _ = \
                cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args,
                                         scene_info.is_nerf_synthetic, True, mask_color=mask_color)

        if self.loaded_iter:
            # [fix2] load_ply(path) — cam_infos/train_test_exp 인자 제거
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud",
                                                 "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            if self.composition:
                if is_blender:
                    self.gaussians.load_ply(os.path.join(args.source_path, "points3d.ply"))          # [fix2]
                else:
                    print("Initialize composition")
                    self.gaussians.load_ply(os.path.join(args.source_path, "sparse", "0", "points3D.ply"))  # [fix2]
            else:
                color_id = RGB2SH(mask_color / 255)
                # [fix1] create_from_pcd(pcd, spatial_lr_scale, color_id=) — cam_infos 제거
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, color_id=color_id)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud")
        save_folder = os.path.join(point_cloud_path, f"iteration_{iteration}")
        os.makedirs(save_folder, exist_ok=True)
        self.gaussians.save_ply(os.path.join(save_folder, "point_cloud.ply"))
        # [fix3] exposure 블록 제거 (exposure 서브시스템 미사용)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getBlackCameras(self, scale=1.0):
        return self.black_cameras[scale]

    def filter_gaussian(self):
        self.gaussians = self.gaussians.filter_points()
        return self

    def get_id(self):
        return self.gaussians.get_id_color
