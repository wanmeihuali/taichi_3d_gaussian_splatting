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
from torchvision.utils import make_grid
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
        feature_learning_rate: float = 1e-3
        position_learning_rate: float = 1e-5
        increase_color_max_sh_band_interval: int = 1000.
        log_loss_interval: int = 10
        log_metrics_interval: int = 100
        log_image_interval: int = 1000
        enable_taichi_kernel_profiler: bool = False
        log_taichi_kernel_profile_interval: int = 1000
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
        ti.init(arch=ti.cuda, device_memory_GB=0.1, kernel_profiler=self.config.enable_taichi_kernel_profiler) # we don't use taichi fields, so we don't need to allocate memory, but taichi requires the memory to be allocated > 0
        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=2)
        val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=2)
        train_data_loader_iter = itertools.cycle(train_data_loader)
        
        optimizer = torch.optim.AdamW(
            [self.scene.point_cloud_features], lr=self.config.feature_learning_rate, betas=(0.9, 0.999))
        position_optimizer = torch.optim.AdamW(
            [self.scene.point_cloud], lr=self.config.position_learning_rate, betas=(0.9, 0.999))
            
        for iteration in tqdm(range(self.config.num_iterations)):
            optimizer.zero_grad()
            position_optimizer.zero_grad()
            
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
            image_pred, image_depth, pixel_valid_point_count = self.rasterisation(
                gaussian_point_cloud_rasterisation_input)
            # hxwx3->3xhxw
            image_pred = image_pred.permute(2, 0, 1)
            loss, l1_loss, ssim_loss = self.loss_function(
                image_pred, 
                image_gt, 
                point_invalid_mask=self.scene.point_invalid_mask,
                pointcloud_features=self.scene.point_cloud_features)
            loss.backward()
            optimizer.step()
            position_optimizer.step()
            magnitude_grad_viewspace_on_image = None
            if self.adaptive_controller.input_data is not None:
                magnitude_grad_viewspace_on_image = self.adaptive_controller.input_data.magnitude_grad_viewspace_on_image
                self._plot_grad_histogram(
                    self.adaptive_controller.input_data, writer=self.writer, iteration=iteration)
                self._plot_value_histogram(
                    self.scene, writer=self.writer, iteration=iteration)
                self.writer.add_histogram(
                    "train/pixel_valid_point_count", pixel_valid_point_count, iteration)
            self.adaptive_controller.refinement()
            if iteration % self.config.log_loss_interval == 0:
                self.writer.add_scalar(
                    "train/loss", loss.item(), iteration)
                self.writer.add_scalar(
                    "train/l1 loss", l1_loss.item(), iteration)
                self.writer.add_scalar(
                    "train/ssim loss", ssim_loss.item(), iteration)
            if self.config.enable_taichi_kernel_profiler and iteration % self.config.log_taichi_kernel_profile_interval == 0 and iteration > 0:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()
            if iteration % self.config.log_metrics_interval == 0:
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                    image_pred=image_pred, image_gt=image_gt)
                self.writer.add_scalar(
                    "train/psnr", psnr_score.item(), iteration)
                self.writer.add_scalar(
                    "train/ssim", ssim_score.item(), iteration)
            if iteration % self.config.log_image_interval == 0:
                # make image_depth to be 3 channels
                image_depth = image_depth.unsqueeze(0).repeat(3, 1, 1) / \
                    image_depth.max()
                pixel_valid_point_count = pixel_valid_point_count.float().unsqueeze(0).repeat(3, 1, 1) / \
                    pixel_valid_point_count.max()
                image_list = [image_pred, image_gt, image_depth, pixel_valid_point_count]
                if magnitude_grad_viewspace_on_image is not None:
                    magnitude_grad_viewspace_on_image = magnitude_grad_viewspace_on_image.permute(2, 0, 1)
                    magnitude_grad_u_viewspace_on_image = magnitude_grad_viewspace_on_image[0]
                    magnitude_grad_v_viewspace_on_image = magnitude_grad_viewspace_on_image[1]
                    magnitude_grad_u_viewspace_on_image /= magnitude_grad_u_viewspace_on_image.max()
                    magnitude_grad_v_viewspace_on_image /= magnitude_grad_v_viewspace_on_image.max()
                    image_list.append(magnitude_grad_u_viewspace_on_image.unsqueeze(0).repeat(3, 1, 1))
                    image_list.append(magnitude_grad_v_viewspace_on_image.unsqueeze(0).repeat(3, 1, 1))
                grid = make_grid(image_list, nrow=2)
                
                self.writer.add_image(
                    "train/image", grid, iteration)
            del image_gt, T_pointcloud_camera, camera_info, gaussian_point_cloud_rasterisation_input, image_pred, loss, l1_loss, ssim_loss
            if iteration % self.config.val_interval == 0 and iteration != 0:
                self.validation(val_data_loader, iteration)

    @staticmethod
    def _compute_pnsr_and_ssim(image_pred, image_gt):
        with torch.no_grad():
            psnr_score = 10 * \
                torch.log10(1.0 / torch.mean((image_pred - image_gt) ** 2))
            ssim_score = ssim(image_pred.unsqueeze(0), image_gt.unsqueeze(
                0), data_range=1.0, size_average=True)
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
            num_overlap_tiles = grad_input.num_overlap_tiles
            num_affected_pixels = grad_input.num_affected_pixels
            magnitude_grad_color = grad_input.magnitude_grad_color
            mean_magnitude_grad_color = magnitude_grad_color / num_affected_pixels
            # fill nan with 0
            mean_magnitude_grad_color[mean_magnitude_grad_color != mean_magnitude_grad_color] = 0
            writer.add_histogram("grad/xyz_grad", xyz_grad, iteration)
            writer.add_histogram("grad/uv_grad", uv_grad, iteration)
            writer.add_histogram("grad/q_grad", q_grad, iteration)
            writer.add_histogram("grad/s_grad", s_grad, iteration)
            writer.add_histogram("grad/alpha_grad", alpha_grad, iteration)
            writer.add_histogram("grad/r_grad", r_grad, iteration)
            writer.add_histogram("grad/g_grad", g_grad, iteration)
            writer.add_histogram("grad/b_grad", b_grad, iteration)
            writer.add_histogram("value/num_overlap_tiles", num_overlap_tiles, iteration)
            writer.add_histogram("value/num_affected_pixels", num_affected_pixels, iteration)
            writer.add_histogram("value/magnitude_grad_color", magnitude_grad_color, iteration)
            writer.add_histogram("value/mean_magnitude_grad_color", mean_magnitude_grad_color, iteration)

    @staticmethod
    def _plot_value_histogram(scene: GaussianPointCloudScene, writer, iteration):
        with torch.no_grad():
            valid_point_cloud = scene.point_cloud[scene.point_invalid_mask == 0]
            valid_point_cloud_features = scene.point_cloud_features[scene.point_invalid_mask == 0]
            num_valid_points = valid_point_cloud.shape[0]
            q = valid_point_cloud_features[:, :4]
            s = valid_point_cloud_features[:, 4:7]
            alpha = valid_point_cloud_features[:, 7]
            r = valid_point_cloud_features[:, 8:24]
            g = valid_point_cloud_features[:, 24:40]
            b = valid_point_cloud_features[:, 40:56]
            writer.add_scalar("value/num_valid_points", num_valid_points, iteration)
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
                image_pred, image_depth, pixel_valid_point_count = self.rasterisation(
                    gaussian_point_cloud_rasterisation_input)
                image_pred = image_pred.permute(2, 0, 1)
                image_depth = image_depth.unsqueeze(0).repeat(3, 1, 1) / image_depth.max()
                pixel_valid_point_count = pixel_valid_point_count.float().unsqueeze(0).repeat(3, 1, 1) / pixel_valid_point_count.max()
                loss, _, _ = self.loss_function(image_pred, image_gt)
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                    image_pred=image_pred, image_gt=image_gt)
                total_loss += loss.item()
                total_psnr_score += psnr_score.item()
                total_ssim_score += ssim_score.item()
                grid = make_grid([image_pred, image_gt, image_depth, pixel_valid_point_count], nrow=2)
                self.writer.add_image(
                    f"val/image {idx}", grid, iteration)

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
    parser.add_argument("--gen_template_only",
                        action="store_true", default=False)
    args = parser.parse_args()
    if args.gen_template_only:
        config = GaussianPointCloudTrainer.TrainConfig()
        # convert config to yaml
        config.to_yaml_file(args.train_config)
        exit(0)
    config = GaussianPointCloudTrainer.TrainConfig.from_yaml_file(
        args.train_config)
    trainer = GaussianPointCloudTrainer(config)
    trainer.train()
