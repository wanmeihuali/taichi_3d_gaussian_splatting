# %%
from GaussianPointCloudScene import GaussianPointCloudScene
from ImagePoseDataset import ImagePoseDataset
from Camera import CameraInfo
from GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from GaussianPointAdaptiveController import GaussianPointAdaptiveController
from LossFunction import LossFunction
import torch
import argparse
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass
import itertools
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim
from tqdm import tqdm
import taichi as ti
import os


class GaussianPointCloudTrainer:
    @dataclass
    class TrainConfig(YAMLWizard):
        train_dataset_json_path: str = ""
        val_dataset_json_path: str = ""
        pointcloud_parquet_path: str = ""
        num_iterations: int = 300000
        val_interval: int = 1000
        increase_color_max_sh_band_interval: int = 1000.
        log_loss_interval: int = 10
        log_metrics_interval: int = 100
        log_image_interval: int = 1000
        summary_writer_log_dir: str = "logs"
        rasterisation_config: GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig()
        adaptive_controller_config: GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig()
        gaussian_point_cloud_scene_config: GaussianPointCloudScene.PointCloudSceneConfig = GaussianPointCloudScene.PointCloudSceneConfig()
        loss_function_config: LossFunction.LossFunctionConfig = LossFunction.LossFunctionConfig()

    def __init__(self, config: TrainConfig):
        self.config = config
        self.writer = SummaryWriter(
            log_dir=self.config.summary_writer_log_dir)

        self.train_dataset = ImagePoseDataset(
            dataset_json_path=self.config.train_dataset_json_path)
        self.val_dataset = ImagePoseDataset(
            dataset_json_path=self.config.val_dataset_json_path)
        self.scene = GaussianPointCloudScene.from_parquet(
            self.config.pointcloud_parquet_path, config=self.config.gaussian_point_cloud_scene_config)
        self.scene = self.scene.cuda()
        self.adaptive_controller = GaussianPointAdaptiveController(
            config=self.config.adaptive_controller_config,
            maintained_parameters=GaussianPointAdaptiveController.GaussianPointAdaptiveControllerMaintainedParameters(
                pointcloud=self.scene.point_cloud,
                pointcloud_features=self.scene.point_cloud_features,
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

    def train(self):
        ti.init(arch=ti.cuda)
        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=2)
        val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=2)
        train_data_loader_iter = itertools.cycle(train_data_loader)
        optimizer = torch.optim.Adam(
            self.scene.parameters(), lr=1e-4, betas=(0.9, 0.999))
        for iteration in tqdm(range(self.config.num_iterations)):
            image_gt, T_pointcloud_camera, camera_info = next(
                train_data_loader_iter)
            image_gt = image_gt.cuda()
            T_pointcloud_camera = T_pointcloud_camera.cuda()
            camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
            camera_info.camera_width = int(camera_info.camera_width)
            camera_info.camera_height = int(camera_info.camera_height)
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
            image_pred = torch.sigmoid(image_pred)
            # hxwx3->3xhxw
            image_pred = image_pred.permute(2, 0, 1)
            loss, l1_loss, ssim_loss = self.loss_function(image_pred, image_gt)
            loss.backward()
            optimizer.step()

            if self.adaptive_controller.input_data is not None and iteration % self.config.log_image_interval == 0:
                self._plot_grad_histogram(self.adaptive_controller.input_data, writer=self.writer, iteration=iteration)
                self._plot_value_histogram(self.scene, writer=self.writer, iteration=iteration)
            self.adaptive_controller.refinement()
            if iteration % self.config.log_loss_interval == 0:
                self.writer.add_scalar(
                    "train/loss", loss.item(), iteration)
                self.writer.add_scalar(
                    "train/l1 loss", l1_loss.item(), iteration)
                self.writer.add_scalar(
                    "train/ssim loss", ssim_loss.item(), iteration)
            if iteration % self.config.log_metrics_interval == 0:
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                    image_pred=image_pred, image_gt=image_gt)
                self.writer.add_scalar(
                    "train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar(
                    "train/ssim", ssim_score.item(), iteration)
            if iteration % self.config.log_image_interval == 0:
                self.writer.add_image(
                    "train/image_pred", image_pred, iteration)
                self.writer.add_image(
                    "train/image_gt", image_gt, iteration)
            del image_gt, T_pointcloud_camera, camera_info, gaussian_point_cloud_rasterisation_input, image_pred, loss, l1_loss, ssim_loss
            if iteration % self.config.val_interval == 0 and iteration != 0:
                self.validation(val_data_loader, iteration)
    
    @staticmethod
    def _compute_pnsr_and_ssim(image_pred, image_gt):
        with torch.no_grad():
            psnr_score = 10 * torch.log10(1.0 / torch.mean((image_pred - image_gt) ** 2))
            ssim_score = ssim(image_pred.unsqueeze(0), image_gt.unsqueeze(0), data_range=1.0, size_average=True)
            return psnr_score, ssim_score
    
    @staticmethod
    def _plot_grad_histogram(grad_input: GaussianPointCloudRasterisation.BackwardValidPointHookInput, writer, iteration):
        with torch.no_grad():
            xyz_grad = grad_input.grad_point_in_camera
            uv_grad = grad_input.grad_viewspace
            feature_grad = grad_input.grad_pointfeatures_in_camera
            q_grad = feature_grad[:, :4]
            s_grad = feature_grad[:, 4:7]
            alpha_grad = feature_grad[:, 7]
            r_grad = feature_grad[:, 8:24]
            g_grad = feature_grad[:, 24:40]
            b_grad = feature_grad[:, 40:56]
            writer.add_histogram("grad/xyz_grad", xyz_grad, iteration)
            writer.add_histogram("grad/uv_grad", uv_grad, iteration)
            writer.add_histogram("grad/q_grad", q_grad, iteration)
            writer.add_histogram("grad/s_grad", s_grad, iteration)
            writer.add_histogram("grad/alpha_grad", alpha_grad, iteration)
            writer.add_histogram("grad/r_grad", r_grad, iteration)
            writer.add_histogram("grad/g_grad", g_grad, iteration)
            writer.add_histogram("grad/b_grad", b_grad, iteration)
    
    @staticmethod
    def _plot_value_histogram(scene: GaussianPointCloudScene, writer, iteration):
        with torch.no_grad():
            valid_point_cloud = scene.point_cloud[scene.point_invalid_mask == 0]
            valid_point_cloud_features = scene.point_cloud_features[scene.point_invalid_mask == 0]
            q = valid_point_cloud_features[:, :4]
            s = valid_point_cloud_features[:, 4:7]
            alpha = valid_point_cloud_features[:, 7]
            r = valid_point_cloud_features[:, 8:24]
            g = valid_point_cloud_features[:, 24:40]
            b = valid_point_cloud_features[:, 40:56]
            writer.add_histogram("value/q", q, iteration)
            writer.add_histogram("value/s", s, iteration)
            writer.add_histogram("value/alpha", alpha, iteration)
            writer.add_histogram("value/r", r, iteration)
            writer.add_histogram("value/g", g, iteration)
            writer.add_histogram("value/b", b, iteration)
        
    
    def validation(self, val_data_loader, iteration):
        with torch.no_grad():
            total_loss = 0.0
            total_psnr_score = 0.0
            total_ssim_score = 0.0
            for idx, val_data in enumerate(tqdm(val_data_loader)):
                image_gt, T_pointcloud_camera, camera_info = val_data
                image_gt = image_gt.cuda()
                T_pointcloud_camera = T_pointcloud_camera.cuda()
                camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                # make taichi happy.
                camera_info.camera_width = int(camera_info.camera_width)
                camera_info.camera_height = int(camera_info.camera_height)
                gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info,
                    T_pointcloud_camera=T_pointcloud_camera,
                    color_max_sh_band=3
                )
                image_pred = self.rasterisation(
                    gaussian_point_cloud_rasterisation_input)
                # apply sigmoid
                image_pred = torch.sigmoid(image_pred)
                image_pred = image_pred.permute(2, 0, 1)
                loss, _, _ = self.loss_function(image_pred, image_gt)
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                    image_pred=image_pred, image_gt=image_gt)
                total_loss += loss.item()
                total_psnr_score += psnr_score.item()
                total_ssim_score += ssim_score.item()
                self.writer.add_image(
                    f"val/image pred {idx}", image_pred, iteration)
                self.writer.add_image(
                    f"val/image gt {idx}", image_gt, iteration)
                
            mean_loss = total_loss / len(val_data_loader)
            mean_psnr_score = total_psnr_score / len(val_data_loader)
            mean_ssim_score = total_ssim_score / len(val_data_loader)
            self.writer.add_scalar(
                "val/loss", mean_loss, iteration)
            self.writer.add_scalar(
                "val/psnr", mean_psnr_score, iteration)
            self.writer.add_scalar(
                "val/ssim", mean_ssim_score, iteration)
            self.scene.to_parquet(
                os.path.join(self.config.summary_writer_log_dir, f"scene_{iteration}.parquet"))

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a Gaussian Point Cloud Scene")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--gen_template_only", action="store_true", default=False)
    args = parser.parse_args()
    if args.gen_template_only:
        config = GaussianPointCloudTrainer.TrainConfig()
        # convert config to yaml
        config.to_yaml_file(args.train_config)
        exit(0)
    config = GaussianPointCloudTrainer.TrainConfig.from_yaml_file(args.train_config)
    trainer = GaussianPointCloudTrainer(config)
    trainer.train()
    

