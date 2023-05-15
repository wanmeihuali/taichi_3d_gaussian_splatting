# %%
from GaussianPointCloudScene import GaussianPointCloudScene
from ImagePoseDataset import ImagePoseDataset
from Camera import CameraInfo
from GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from GaussianPointAdaptiveController import GaussianPointAdaptiveController
from LossFunction import LossFunction
import torch
import argparse
import yaml
from dataclasses import dataclass
import itertools


class GaussianPointCloudTrainer:
    @dataclass
    class TrainConfig:
        train_dataset_json_path: str
        val_dataset_json_path: str
        pointcloud_parquet_path: str
        num_iterations: int = 300000
        val_interval: int = 1000
        increase_color_max_sh_band_interval: int = 1000.
        rasterisation_config: GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig()
        adaptive_controller_config: GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig()
        gaussian_point_cloud_scene_config: GaussianPointCloudScene.PointCloudSceneConfig = GaussianPointCloudScene.PointCloudSceneConfig()
        loss_function_config: LossFunction.LossFunctionConfig = LossFunction.LossFunctionConfig()

    def __init__(self, config: TrainConfig):
        self.config = config

        self.train_dataset = ImagePoseDataset(
            dataset_json_path=self.config.train_dataset_json_path)
        self.val_dataset = ImagePoseDataset(
            dataset_json_path=self.config.val_dataset_json_path)
        self.scene = GaussianPointCloudScene.from_parquet(
            self.config.pointcloud_parquet_path, config=self.config.gaussian_point_cloud_scene_config)
        self.adaptive_controller = GaussianPointAdaptiveController(
            config=self.config.adaptive_controller_config,
            maintained_parameters=GaussianPointAdaptiveController.GaussianPointAdaptiveControllerMaintainedParameters(
                point_cloud=self.scene.point_cloud,
                point_cloud_features=self.scene.point_cloud_features,
                point_invalid_mask=self.scene.point_invalid_mask,
            )
        )
        self.rasterisation = GaussianPointCloudRasterisation(
            config=self.config.rasterisation_config,
            backward_valid_point_hook=self.adaptive_controller.update,
        )
        self.loss_function = LossFunction(
            config=self.config.loss_function_config)

        # move scene to GPU
        self.scene = self.scene.cuda()

    def train(self):
        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=2)
        val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=2)
        train_data_loader_iter = itertools.cycle(train_data_loader)
        for iteration in range(self.config.num_iterations):
            image_gt, T_pointcloud_camera, camera_info = next(
                train_data_loader_iter)
            image_gt = image_gt.cuda()
            T_pointcloud_camera = T_pointcloud_camera.cuda()
            camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
            gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=self.scene.point_cloud,
                point_cloud_features=self.scene.point_cloud_features,
                point_invalid_mask=self.scene.point_invalid_mask,
                camera_info=camera_info,
                T_pointcloud_camera=T_pointcloud_camera,
                color_max_sh_band=iteration // self.config.increase_color_max_sh_band_interval,
            )
            image_pred = self.rasterisation(
                gaussian_point_cloud_rasterisation_input)
            # clip image_pred to [0, 1]
            image_pred = torch.clamp(image_pred, 0.0, 1.0)
            loss = self.loss_function(image_pred, image_gt)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a Gaussian Point Cloud Scene")
    parser.add_argument("--train_config", type=str, required=True)
