import numpy as np
import torch
from dataclasses import dataclass
from GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from dataclass_wizard import YAMLWizard
from typing import Optional


class GaussianPointAdaptiveController:
    """
    For simplicity, I set the size of point cloud to be fixed during training. an extra mask is used to indicate whether a point is invalid or not.
    When initialising, the input point cloud is concatenated with extra points filled with zero. The mask is concatenated with extra True.
    When densifying and splitting points, new points are assigned to locations of invalid points.
    When removing points, we just set the mask to True.
    """
    @dataclass
    class GaussianPointAdaptiveControllerConfig(YAMLWizard):
        num_iterations_warm_up: int = 500
        num_iterations_densify: int = 100
        # from paper: densify every 100 iterations and remove any Gaussians that are essentially transparent, i.e., with 𝛼 less than a threshold 𝜖𝛼.
        transparent_alpha_threshold: float = -0.5
        # from paper: densify Gaussians with an average magnitude of view-space position gradients above a threshold 𝜏pos, which we set to 0.0002 in our tests.
        # I have no idea why their threshold is so low, may be their view space is normalized to [0, 1]?
        # TODO: find out a proper threshold
        densification_view_space_position_gradients_threshold: float = 0.005
        # from paper:  large Gaussians in regions with high variance need to be split into smaller Gaussians. We replace such Gaussians by two new ones, and divide their scale by a factor of 𝜙 = 1.6
        gaussian_split_factor_phi: float = 1.6
        # in paper section 5.2, they describe a method to moderate the increase in the number of Gaussians is to set the 𝛼 value close to zero every
        # 3000 iterations. I have no idea how it is implemented. I just assume that it is a reset of 𝛼 to fixed value.
        num_iterations_reset_alpha: int = 3000
        reset_alpha_value: float = 0.1
        # the paper doesn't mention this value, but we need a value and method to determine whether a point is under-reconstructed or over-reconstructed
        # for now, the method is to threshold norm of exp(s)
        # TODO: find out a proper threshold
        under_reconstructed_s_threshold: float = 0.001
        floater_threshold: float = 0.5

    @dataclass
    class GaussianPointAdaptiveControllerMaintainedParameters:
        pointcloud: torch.Tensor  # shape: [num_points, 3]
        # shape: [num_points, num_features], num_features is 56
        pointcloud_features: torch.Tensor
        # shape: [num_points], dtype: int8 because taichi doesn't support bool type
        point_invalid_mask: torch.Tensor

    @dataclass
    class GaussianPointAdaptiveControllerDensifyPointInfo:
        floater_point_id: torch.Tensor # shape: [num_floater_points]
        transparent_point_id: torch.Tensor # shape: [num_transparent_points]
        under_reconstructed_point_id: torch.Tensor # shape: [num_points_under_reconstructed]
        over_reconstructed_point_id: torch.Tensor # shape: [num_points_over_reconstructed]
        under_reconstructed_point_position_before_optimization: torch.Tensor # shape: [num_points_under_reconstructed, 3]
        over_reconstructed_point_position_before_optimization: torch.Tensor # shape: [num_points_over_reconstructed, 3]
        

    def __init__(self,
                 config: GaussianPointAdaptiveControllerConfig,
                 maintained_parameters: GaussianPointAdaptiveControllerMaintainedParameters):
        self.iteration_counter = -1
        self.config = config
        self.maintained_parameters = maintained_parameters
        self.input_data = None
        self.densify_point_info: Optional[GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo] = None


    def update(self, input_data: GaussianPointCloudRasterisation.BackwardValidPointHookInput):
        self.iteration_counter += 1
        with torch.no_grad():
            if self.iteration_counter < self.config.num_iterations_warm_up:
                pass
            elif self.iteration_counter % self.config.num_iterations_densify == 0:
                self._find_densify_points(input_data)
                self.input_data = input_data

    def refinement(self):
        with torch.no_grad():
            if self.iteration_counter < self.config.num_iterations_warm_up:
                return
            if self.iteration_counter % self.config.num_iterations_densify == 0:
                self._add_densify_points()
            if self.iteration_counter % self.config.num_iterations_reset_alpha == 0:
                self.reset_alpha()
            self.input_data = None

    def _find_densify_points(self, input_data: GaussianPointCloudRasterisation.BackwardValidPointHookInput):
        """ find points to densify, it should happened in backward pass before optimiser step.
        so that the original point values are recorded, and when a point is cloned/split, the
        two points are not the same.

        Args:
            input_data (GaussianPointCloudRasterisation.BackwardValidPointHookInput): input
        """
        pointcloud = self.maintained_parameters.pointcloud
        pointcloud_features = self.maintained_parameters.pointcloud_features
        point_id_list = torch.arange(pointcloud.shape[0], device=pointcloud.device)
        
        # Note that both floater points and transparent points are apply on all valid points
        # while densification only apply on points in camera in the current frame
        floater_mask = (pointcloud_features[:, 4:7].exp().norm(dim=1) > self.config.floater_threshold) & \
            (self.maintained_parameters.point_invalid_mask == 0)
        
        floater_point_id = point_id_list[floater_mask]
        point_alpha = pointcloud_features[:, 7]  # alpha before sigmoid
        nan_mask = torch.isnan(pointcloud_features).any(dim=1)
        transparent_point_mask = ((point_alpha < self.config.transparent_alpha_threshold) | nan_mask) & \
            (self.maintained_parameters.point_invalid_mask == 0) & \
                (~floater_mask) # ensure floater points and transparent points don't overlap
        transparent_point_id = point_id_list[transparent_point_mask]
        
        

        # find points that are under-reconstructed or over-reconstructed
        point_id_in_camera_list: torch.Tensor = input_data.point_id_in_camera_list
        point_features_in_camera = pointcloud_features[point_id_in_camera_list]
        will_be_remove_mask = floater_mask[point_id_in_camera_list] | transparent_point_mask[point_id_in_camera_list]
        # shape: [num_points_in_camera, 2]
        grad_viewspace = input_data.grad_viewspace
        # shape: [num_points_in_camera, num_features]
        # all these three masks are on num_points_in_camera, not num_points
        to_densify_mask = (grad_viewspace.norm(
            dim=1) > self.config.densification_view_space_position_gradients_threshold) & \
                (~will_be_remove_mask)
        # shape: [num_points_in_camera, 3]
        point_s_in_camera = point_features_in_camera[:, 4:7]
        under_reconstructed_mask = to_densify_mask & (point_s_in_camera.exp().norm(
            dim=1) < self.config.under_reconstructed_s_threshold)
        over_reconstructed_mask = to_densify_mask & (~under_reconstructed_mask)

        under_reconstructed_point_id = point_id_in_camera_list[under_reconstructed_mask]
        over_reconstructed_point_id = point_id_in_camera_list[over_reconstructed_mask]

        under_reconstructed_point_position_before_optimization = pointcloud[under_reconstructed_point_id]
        over_reconstructed_point_position_before_optimization = pointcloud[over_reconstructed_point_id]

        self.densify_point_info = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo(
            floater_point_id=floater_point_id,
            transparent_point_id=transparent_point_id,
            under_reconstructed_point_id=under_reconstructed_point_id,
            over_reconstructed_point_id=over_reconstructed_point_id,
            under_reconstructed_point_position_before_optimization=under_reconstructed_point_position_before_optimization,
            over_reconstructed_point_position_before_optimization=over_reconstructed_point_position_before_optimization,
        )

    def _add_densify_points(self):
        assert self.densify_point_info is not None
        total_valid_points_before_densify = self.maintained_parameters.point_invalid_mask.shape[0] - \
            self.maintained_parameters.point_invalid_mask.sum()
        num_transparent_points = self.densify_point_info.transparent_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.transparent_point_id] = 1
        num_floaters_points = self.densify_point_info.floater_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.floater_point_id] = 1
        num_under_reconstructed_points = self.densify_point_info.under_reconstructed_point_id.shape[0]
        num_over_reconstructed_points = self.densify_point_info.over_reconstructed_point_id.shape[0]
        num_of_densify_points = num_under_reconstructed_points + num_over_reconstructed_points
        invalid_point_id_to_fill = torch.where(self.maintained_parameters.point_invalid_mask == 1)[0][:num_of_densify_points]

        under_reconstructed_point_id_to_fill = invalid_point_id_to_fill[:self.densify_point_info.under_reconstructed_point_id.shape[0]]
        over_reconstructed_point_id_to_fill = invalid_point_id_to_fill[self.densify_point_info.under_reconstructed_point_id.shape[0]:]

        # for position, we use the position before optimization for new points, so that original points and new points have different positions
        num_fillable_under_reconstructed_points = 0
        if num_under_reconstructed_points > 0:
            num_fillable_under_reconstructed_points = under_reconstructed_point_id_to_fill.shape[0]
            self.maintained_parameters.pointcloud[under_reconstructed_point_id_to_fill] = \
                self.densify_point_info.under_reconstructed_point_position_before_optimization[:num_fillable_under_reconstructed_points]
            self.maintained_parameters.pointcloud_features[under_reconstructed_point_id_to_fill] = \
                self.maintained_parameters.pointcloud_features[self.densify_point_info.under_reconstructed_point_id[:num_fillable_under_reconstructed_points]]
            self.maintained_parameters.point_invalid_mask[under_reconstructed_point_id_to_fill] = 0

        num_fillable_over_reconstructed_points = 0
        if num_over_reconstructed_points > 0:
            num_fillable_over_reconstructed_points = over_reconstructed_point_id_to_fill.shape[0]
            self.maintained_parameters.pointcloud[over_reconstructed_point_id_to_fill] = \
                self.densify_point_info.over_reconstructed_point_position_before_optimization[:num_fillable_over_reconstructed_points]
            self.maintained_parameters.pointcloud_features[over_reconstructed_point_id_to_fill] = \
                self.maintained_parameters.pointcloud_features[self.densify_point_info.over_reconstructed_point_id[:num_fillable_over_reconstructed_points]]
            self.maintained_parameters.pointcloud_features[over_reconstructed_point_id_to_fill, 4:7] -= np.log(self.config.gaussian_split_factor_phi)
            self.maintained_parameters.pointcloud_features[self.densify_point_info.over_reconstructed_point_id, 4:7] -= np.log(self.config.gaussian_split_factor_phi)
            self.maintained_parameters.point_invalid_mask[over_reconstructed_point_id_to_fill] = 0
        total_valid_points_after_densify = self.maintained_parameters.point_invalid_mask.shape[0] - \
            self.maintained_parameters.point_invalid_mask.sum()
        assert total_valid_points_after_densify == total_valid_points_before_densify - num_transparent_points - num_floaters_points + num_fillable_under_reconstructed_points + num_fillable_over_reconstructed_points
        print(f"total valid points: {total_valid_points_before_densify} -> {total_valid_points_after_densify}, under reconstructed points: {num_under_reconstructed_points}, over reconstructed points: {num_over_reconstructed_points}, transparent points: {num_transparent_points}, floaters points: {num_floaters_points}, fillable under reconstructed points: {num_fillable_under_reconstructed_points}, fillable over reconstructed points: {num_fillable_over_reconstructed_points}")
        self.densify_point_info = None # clear densify point info

    def reset_alpha(self):
        pointcloud_features = self.maintained_parameters.pointcloud_features
        pointcloud_features[:, 7] = self.config.reset_alpha_value
