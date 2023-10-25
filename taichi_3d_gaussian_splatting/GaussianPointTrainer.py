# %%
from .GaussianPointCloudScene import GaussianPointCloudScene
from .ImagePoseDataset import ImagePoseDataset
from .Camera import CameraInfo
from .GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from .GaussianPointAdaptiveController import GaussianPointAdaptiveController
from .LossFunction import LossFunction
import torch
import argparse
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from pytorch_msssim import ssim
from tqdm import tqdm
import taichi as ti
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import deque
import numpy as np
from typing import Optional


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

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
        position_learning_rate_decay_rate: float = 0.97
        position_learning_rate_decay_interval: int = 100
        increase_color_max_sh_band_interval: int = 1000.
        log_loss_interval: int = 10
        log_metrics_interval: int = 100
        print_metrics_to_console: bool = False
        log_image_interval: int = 1000
        enable_taichi_kernel_profiler: bool = False
        log_taichi_kernel_profile_interval: int = 1000
        log_validation_image: bool = True
        initial_downsample_factor: int = 4
        half_downsample_factor_interval: int = 250
        summary_writer_log_dir: str = "logs"
        output_model_dir: Optional[str] = None
        rasterisation_config: GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig()
        adaptive_controller_config: GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig()
        gaussian_point_cloud_scene_config: GaussianPointCloudScene.PointCloudSceneConfig = GaussianPointCloudScene.PointCloudSceneConfig()
        loss_function_config: LossFunction.LossFunctionConfig = LossFunction.LossFunctionConfig()

    def __init__(self, config: TrainConfig):
        self.config = config
        # create the log directory if it doesn't exist
        os.makedirs(self.config.summary_writer_log_dir, exist_ok=True)
        if self.config.output_model_dir is None:
            self.config.output_model_dir = self.config.summary_writer_log_dir
            os.makedirs(self.config.output_model_dir, exist_ok=True)
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
                point_object_id=self.scene.point_object_id,
            ))
        self.rasterisation = GaussianPointCloudRasterisation(
            config=self.config.rasterisation_config,
            backward_valid_point_hook=self.adaptive_controller.update,
        )
        
        self.loss_function = LossFunction(
            config=self.config.loss_function_config)
        
        self.best_psnr_score = 0.

        # move scene to GPU

    @staticmethod
    def _downsample_image_and_camera_info(image: torch.Tensor, camera_info: CameraInfo, downsample_factor: int):
        camera_height = camera_info.camera_height // downsample_factor
        camera_width = camera_info.camera_width // downsample_factor
        image = transforms.functional.resize(image, size=(camera_height, camera_width), antialias=True)
        camera_width = camera_width - camera_width % 16
        camera_height = camera_height - camera_height % 16
        image = image[:3, :camera_height, :camera_width].contiguous()
        camera_intrinsics = camera_info.camera_intrinsics
        camera_intrinsics = camera_intrinsics.clone()
        camera_intrinsics[0, 0] /= downsample_factor
        camera_intrinsics[1, 1] /= downsample_factor
        camera_intrinsics[0, 2] /= downsample_factor
        camera_intrinsics[1, 2] /= downsample_factor
        resized_camera_info = CameraInfo(
            camera_intrinsics=camera_intrinsics,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_id=camera_info.camera_id)
        return image, resized_camera_info

    def train(self):
        ti.init(arch=ti.cuda, device_memory_GB=0.1, kernel_profiler=self.config.enable_taichi_kernel_profiler) # we don't use taichi fields, so we don't need to allocate memory, but taichi requires the memory to be allocated > 0
        train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=4)
        val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=4)
        train_data_loader_iter = cycle(train_data_loader)
        
        optimizer = torch.optim.Adam(
            [self.scene.point_cloud_features], lr=self.config.feature_learning_rate, betas=(0.9, 0.999))
        position_optimizer = torch.optim.Adam(
            [self.scene.point_cloud], lr=self.config.position_learning_rate, betas=(0.9, 0.999))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=position_optimizer, gamma=self.config.position_learning_rate_decay_rate)
        downsample_factor = self.config.initial_downsample_factor

        recent_losses = deque(maxlen=100)
            
        previous_problematic_iteration = -1000
        for iteration in tqdm(range(self.config.num_iterations)):
            if iteration % self.config.half_downsample_factor_interval == 0 and iteration > 0 and downsample_factor > 1:
                downsample_factor = downsample_factor // 2
            optimizer.zero_grad()
            position_optimizer.zero_grad()
            
            image_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info = next(
                train_data_loader_iter)
            if downsample_factor > 1:
                image_gt, camera_info = GaussianPointCloudTrainer._downsample_image_and_camera_info(
                    image_gt, camera_info, downsample_factor=downsample_factor)
            image_gt = image_gt.cuda()
            q_pointcloud_camera = q_pointcloud_camera.cuda()
            t_pointcloud_camera = t_pointcloud_camera.cuda()
            camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
            camera_info.camera_width = int(camera_info.camera_width)
            camera_info.camera_height = int(camera_info.camera_height)
            gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=self.scene.point_cloud,
                point_cloud_features=self.scene.point_cloud_features,
                point_object_id=self.scene.point_object_id,
                point_invalid_mask=self.scene.point_invalid_mask,
                camera_info=camera_info,
                q_pointcloud_camera=q_pointcloud_camera,
                t_pointcloud_camera=t_pointcloud_camera,
                color_max_sh_band=iteration // self.config.increase_color_max_sh_band_interval,
            )
            image_pred, image_depth, pixel_valid_point_count = self.rasterisation(
                gaussian_point_cloud_rasterisation_input)
            # clip to [0, 1]
            image_pred = torch.clamp(image_pred, min=0, max=1)
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

            recent_losses.append(loss.item())
            

            if iteration % self.config.position_learning_rate_decay_interval == 0:
                scheduler.step()
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
            if self.adaptive_controller.has_plot:
                fig, ax = self.adaptive_controller.figure, self.adaptive_controller.ax
                # plot image_pred in ax
                ax.imshow(image_pred.detach().cpu().numpy().transpose(
                    1, 2, 0), zorder=1, vmin=0, vmax=1)

                self.writer.add_figure(
                    "train/densify_points", fig, iteration)
                self.adaptive_controller.figure, self.adaptive_controller.ax = plt.subplots()
                self.adaptive_controller.has_plot = False
            if iteration % self.config.log_loss_interval == 0:
                self.writer.add_scalar(
                    "train/loss", loss.item(), iteration)
                self.writer.add_scalar(
                    "train/l1 loss", l1_loss.item(), iteration)
                self.writer.add_scalar(
                    "train/ssim loss", ssim_loss.item(), iteration)
                if self.config.print_metrics_to_console:
                    print(f"train_iteration={iteration};")
                    print(f"train_loss={loss.item()};")
                    print(f"train_l1_loss={l1_loss.item()};")
                    print(f"train_ssim_loss={ssim_loss.item()};")
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
                if self.config.print_metrics_to_console:
                    print(f"train_psnr={psnr_score.item()};")
                    print(f"train_psnr_{iteration}={psnr_score.item()};")
                    print(f"train_ssim={ssim_score.item()};")
                    print(f"train_ssim_{iteration}={ssim_score.item()};")

            is_problematic = False
            if len(recent_losses) == recent_losses.maxlen and iteration - previous_problematic_iteration > recent_losses.maxlen:
                avg_loss = sum(recent_losses) / len(recent_losses)
                if loss.item() > avg_loss * 1.5:
                    is_problematic = True
                    previous_problematic_iteration = iteration

            if iteration % self.config.log_image_interval == 0 or is_problematic:
                # make image_depth to be 3 channels
                image_depth = self._easy_cmap(image_depth)
                pixel_valid_point_count = pixel_valid_point_count.float().unsqueeze(0).repeat(3, 1, 1) / \
                    pixel_valid_point_count.max()
                image_list = [image_pred, image_gt, image_depth, pixel_valid_point_count]
                if magnitude_grad_viewspace_on_image is not None:
                    magnitude_grad_viewspace_on_image = magnitude_grad_viewspace_on_image.permute(2, 0, 1)
                    magnitude_grad_u_viewspace_on_image = magnitude_grad_viewspace_on_image[0]
                    magnitude_grad_v_viewspace_on_image = magnitude_grad_viewspace_on_image[1]
                    magnitude_grad_u_viewspace_on_image /= magnitude_grad_u_viewspace_on_image.max()
                    magnitude_grad_v_viewspace_on_image /= magnitude_grad_v_viewspace_on_image.max()
                    image_diff = torch.abs(image_pred - image_gt)
                    image_list.append(magnitude_grad_u_viewspace_on_image.unsqueeze(0).repeat(3, 1, 1))
                    image_list.append(magnitude_grad_v_viewspace_on_image.unsqueeze(0).repeat(3, 1, 1))
                    image_list.append(image_diff)
                grid = make_grid(image_list, nrow=2)
                
                if is_problematic:
                    self.writer.add_image(
                        "train/image_problematic", grid, iteration)
                else:
                    self.writer.add_image(
                        "train/image", grid, iteration)

            del image_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info, gaussian_point_cloud_rasterisation_input, image_pred, loss, l1_loss, ssim_loss
            if (iteration % self.config.val_interval == 0 and iteration != 0) or iteration == 7000 or iteration == 5000: # they use 7000 in paper, it's hard to set a interval so hard code it here
                self.validation(val_data_loader, iteration)
    
    @staticmethod
    def _easy_cmap(x: torch.Tensor):
        x_rgb = torch.zeros((3, x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
        x_rgb[0] = torch.clamp(x, 0, 10) / 10.
        x_rgb[1] = torch.clamp(x - 10, 0, 50) / 50.
        x_rgb[2] = torch.clamp(x - 60, 0, 200) / 200.
        return 1. - x_rgb
        

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
            # print(f"num_valid_points={num_valid_points};")
            writer.add_histogram("value/q", q, iteration)
            writer.add_histogram("value/s", s, iteration)
            writer.add_histogram("value/alpha", alpha, iteration)
            writer.add_histogram("value/sigmoid_alpha", torch.sigmoid(alpha), iteration)
            writer.add_histogram("value/r", r, iteration)
            writer.add_histogram("value/g", g, iteration)
            writer.add_histogram("value/b", b, iteration)

    def validation(self, val_data_loader, iteration):
        with torch.no_grad():
            total_loss = 0.0
            total_psnr_score = 0.0
            total_ssim_score = 0.0
            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()
            total_inference_time = 0.0
            for idx, val_data in enumerate(tqdm(val_data_loader)):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                image_gt, q_pointcloud_camera, t_pointcloud_camera, camera_info = val_data
                image_gt = image_gt.cuda()
                q_pointcloud_camera = q_pointcloud_camera.cuda()
                t_pointcloud_camera = t_pointcloud_camera.cuda()
                camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()
                # make taichi happy.
                camera_info.camera_width = int(camera_info.camera_width)
                camera_info.camera_height = int(camera_info.camera_height)
                gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                    point_cloud=self.scene.point_cloud,
                    point_cloud_features=self.scene.point_cloud_features,
                    point_object_id=self.scene.point_object_id,
                    point_invalid_mask=self.scene.point_invalid_mask,
                    camera_info=camera_info,
                    q_pointcloud_camera=q_pointcloud_camera,
                    t_pointcloud_camera=t_pointcloud_camera,
                    color_max_sh_band=3
                )
                start_event.record()
                image_pred, image_depth, pixel_valid_point_count = self.rasterisation(
                    gaussian_point_cloud_rasterisation_input)
                end_event.record()
                torch.cuda.synchronize()
                time_taken = start_event.elapsed_time(end_event)
                total_inference_time += time_taken
                image_pred = torch.clamp(image_pred, 0, 1)
                image_pred = image_pred.permute(2, 0, 1)
                image_depth = self._easy_cmap(image_depth)
                pixel_valid_point_count = pixel_valid_point_count.float().unsqueeze(0).repeat(3, 1, 1) / pixel_valid_point_count.max()
                loss, _, _ = self.loss_function(image_pred, image_gt)
                psnr_score, ssim_score = self._compute_pnsr_and_ssim(
                    image_pred=image_pred, image_gt=image_gt)
                image_diff = torch.abs(image_pred - image_gt)
                total_loss += loss.item()
                total_psnr_score += psnr_score.item()
                total_ssim_score += ssim_score.item()
                grid = make_grid([image_pred, image_gt, image_depth, pixel_valid_point_count, image_diff], nrow=2)
                if self.config.log_validation_image:
                    self.writer.add_image(
                        f"val/image {idx}", grid, iteration)

            if self.config.enable_taichi_kernel_profiler:
                ti.profiler.print_kernel_profiler_info("count")
                ti.profiler.clear_kernel_profiler_info()
            average_inference_time = total_inference_time / len(val_data_loader)

            mean_loss = total_loss / len(val_data_loader)
            mean_psnr_score = total_psnr_score / len(val_data_loader)
            mean_ssim_score = total_ssim_score / len(val_data_loader)
            self.writer.add_scalar(
                "val/loss", mean_loss, iteration)
            self.writer.add_scalar(
                "val/psnr", mean_psnr_score, iteration)
            self.writer.add_scalar(
                "val/ssim", mean_ssim_score, iteration)
            self.writer.add_scalar(
                "val/inference_time", average_inference_time, iteration)
            if self.config.print_metrics_to_console:
                print(f"val_loss={mean_loss};")
                print(f"val_psnr={mean_psnr_score};")
                print(f"val_psnr_{iteration}={mean_psnr_score};")
                print(f"val_ssim={mean_ssim_score};")
                print(f"val_ssim_{iteration}={mean_ssim_score};")
                print(f"val_inference_time={average_inference_time};")
            self.scene.to_parquet(
                os.path.join(self.config.output_model_dir, f"scene_{iteration}.parquet"))
            if mean_psnr_score > self.best_psnr_score:
                self.best_psnr_score = mean_psnr_score
                self.scene.to_parquet(
                    os.path.join(self.config.output_model_dir, f"best_scene.parquet"))
