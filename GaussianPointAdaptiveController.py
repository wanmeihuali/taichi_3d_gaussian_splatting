import numpy as np
import torch
from dataclasses import dataclass
from GaussianPointCloudRasterisation import GaussianPointCloudRasterisation, load_point_cloud_row_into_gaussian_point_3d
from dataclass_wizard import YAMLWizard
from typing import Optional
import taichi as ti

@ti.kernel
def compute_ellipsoid_offset(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, 56)
    ellipsoid_offset: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
):
    for idx in range(pointcloud.shape[0]):
        point = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=idx,
        )
        foci_vector = point.get_ellipsoid_foci_vector()
        ellipsoid_offset[idx, 0] = foci_vector[0]
        ellipsoid_offset[idx, 1] = foci_vector[1]
        ellipsoid_offset[idx, 2] = foci_vector[2]

@ti.kernel
def sample_from_point(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, 56)
    sample_result: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
):
    for idx in range(pointcloud.shape[0]):
        point = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=idx,
        )
        foci_vector = point.sample()
        sample_result[idx, 0] = foci_vector[0]
        sample_result[idx, 1] = foci_vector[1]
        sample_result[idx, 2] = foci_vector[2]

       

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
        floater_num_pixels_threshold: int = 10000
        floater_near_camrea_num_pixels_threshold: int = 10000
        floater_depth_threshold: float = 100
        iteration_start_remove_floater: int = 2000
        under_reconstructed_num_pixels_threshold: int = 512
        enable_ellipsoid_offset: bool = False
        enable_sample_from_point: bool = True

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
        densify_point_id: torch.Tensor # shape: [num_points_to_densify]
        densify_point_position_before_optimization: torch.Tensor # shape: [num_points_to_densify, 3]
        densify_size_reduction_factor: torch.Tensor # shape: [num_points_to_densify]
        

    def __init__(self,
                 config: GaussianPointAdaptiveControllerConfig,
                 maintained_parameters: GaussianPointAdaptiveControllerMaintainedParameters):
        self.iteration_counter = -1
        self.config = config
        self.maintained_parameters = maintained_parameters
        self.input_data = None
        self.densify_point_info: Optional[GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo] = None
        self.accumulated_num_pixels = torch.zeros_like(
            self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)
        self.accumulated_num_in_camera = torch.zeros_like(
            self.maintained_parameters.pointcloud[:, 0], dtype=torch.int32)


    def update(self, input_data: GaussianPointCloudRasterisation.BackwardValidPointHookInput):
        self.iteration_counter += 1
        with torch.no_grad():
            self.accumulated_num_in_camera[input_data.point_id_in_camera_list] += 1
            self.accumulated_num_pixels[input_data.point_id_in_camera_list] += input_data.num_affected_pixels
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
        point_id_in_camera_list: torch.Tensor = input_data.point_id_in_camera_list
        num_affected_pixels: torch.Tensor = input_data.num_affected_pixels
        point_depth_in_camera: torch.Tensor = input_data.point_depth
        average_num_affect_pixels = self.accumulated_num_pixels / self.accumulated_num_in_camera
        average_num_affect_pixels[torch.isnan(average_num_affect_pixels)] = 0
        
        # Note that transparent points are apply on all valid points
        # while floater and densification only apply on points in camera in the current frame
        floater_mask = torch.zeros_like(point_id_list, dtype=torch.bool)
        floater_mask_in_camera = torch.zeros_like(point_id_in_camera_list, dtype=torch.bool)
        floater_point_id = torch.empty(0, dtype=torch.int32, device=pointcloud.device)
        if self.iteration_counter > self.config.iteration_start_remove_floater:
            floater_mask_in_camera = ((num_affected_pixels > self.config.floater_near_camrea_num_pixels_threshold) & \
                    (point_depth_in_camera < self.config.floater_depth_threshold))

            # floater_mask_in_camera = (num_affected_pixels > self.config.floater_num_pixels_threshold)
            floater_point_id = point_id_in_camera_list[floater_mask_in_camera]
            # floater_mask = average_num_affect_pixels > self.config.floater_num_pixels_threshold
            floater_mask[floater_point_id] = True
            floater_mask = floater_mask & (self.maintained_parameters.point_invalid_mask == 0)
        
        point_alpha = pointcloud_features[:, 7]  # alpha before sigmoid
        nan_mask = torch.isnan(pointcloud_features).any(dim=1)
        transparent_point_mask = ((point_alpha < self.config.transparent_alpha_threshold) | nan_mask) & \
            (self.maintained_parameters.point_invalid_mask == 0) & \
                (~floater_mask) # ensure floater points and transparent points don't overlap
        transparent_point_id = point_id_list[transparent_point_mask]
        

        # find points that are under-reconstructed or over-reconstructed
        # point_features_in_camera = pointcloud_features[point_id_in_camera_list]
        will_be_remove_mask = floater_mask_in_camera | transparent_point_mask[point_id_in_camera_list]
        # shape: [num_points_in_camera, 2]
        grad_viewspace = input_data.grad_viewspace
        # shape: [num_points_in_camera, num_features]
        # all these three masks are on num_points_in_camera, not num_points
        to_densify_mask = (grad_viewspace.norm(
            dim=1) > self.config.densification_view_space_position_gradients_threshold) 
        to_densify_mask &= (~will_be_remove_mask) # don't densify floater or transparent points
        
        densify_point_id = point_id_in_camera_list[to_densify_mask]
        densify_point_position_before_optimization = pointcloud[densify_point_id]
        densify_size_reduction_factor = torch.zeros_like(densify_point_id, dtype=torch.float32, device=pointcloud.device)
        over_reconstructed_mask = (num_affected_pixels[to_densify_mask] > self.config.under_reconstructed_num_pixels_threshold)
        densify_size_reduction_factor[over_reconstructed_mask] = \
            np.log(self.config.gaussian_split_factor_phi)
        densify_size_reduction_factor = densify_size_reduction_factor.unsqueeze(-1)
        self.densify_point_info = GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo(
            floater_point_id=floater_point_id,
            transparent_point_id=transparent_point_id,
            densify_point_id=densify_point_id,
            densify_point_position_before_optimization=densify_point_position_before_optimization,
            densify_size_reduction_factor=densify_size_reduction_factor,
        )

    def _add_densify_points(self):
        assert self.densify_point_info is not None
        total_valid_points_before_densify = self.maintained_parameters.point_invalid_mask.shape[0] - \
            self.maintained_parameters.point_invalid_mask.sum()
        num_transparent_points = self.densify_point_info.transparent_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.transparent_point_id] = 1
        num_floaters_points = self.densify_point_info.floater_point_id.shape[0]
        self.maintained_parameters.point_invalid_mask[self.densify_point_info.floater_point_id] = 1
        num_of_densify_points = self.densify_point_info.densify_point_id.shape[0]
        invalid_point_id_to_fill = torch.where(self.maintained_parameters.point_invalid_mask == 1)[0][:num_of_densify_points]


        # for position, we use the position before optimization for new points, so that original points and new points have different positions
        num_fillable_densify_points = 0
        if num_of_densify_points > 0:
            # num_fillable_over_reconstructed_points = over_reconstructed_point_id_to_fill.shape[0]
            num_fillable_densify_points = min(num_of_densify_points, invalid_point_id_to_fill.shape[0])
            self.maintained_parameters.pointcloud[invalid_point_id_to_fill] = \
                self.densify_point_info.densify_point_position_before_optimization[:num_fillable_densify_points]
            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill] = \
                self.maintained_parameters.pointcloud_features[self.densify_point_info.densify_point_id[:num_fillable_densify_points]]
            self.maintained_parameters.pointcloud_features[invalid_point_id_to_fill, 4:7] -= \
                self.densify_point_info.densify_size_reduction_factor[:num_fillable_densify_points]
            over_reconstructed_mask = (self.densify_point_info.densify_size_reduction_factor[:num_fillable_densify_points] > 1e-6).reshape(-1)
            num_over_reconstructed = over_reconstructed_mask.sum().item()
            num_under_reconstructed = num_fillable_densify_points - num_over_reconstructed
            print(f"num_over_reconstructed: {num_over_reconstructed}, num_under_reconstructed: {num_under_reconstructed}")
            densify_point_id = self.densify_point_info.densify_point_id[:num_fillable_densify_points]
            self.maintained_parameters.pointcloud_features[densify_point_id, 4:7] -= \
                    self.densify_point_info.densify_size_reduction_factor[:num_fillable_densify_points]
            if self.config.enable_ellipsoid_offset:
                point_offset = self._generate_point_offset(
                    point_to_split=self.maintained_parameters.pointcloud[densify_point_id],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[densify_point_id])
                self.maintained_parameters.pointcloud[invalid_point_id_to_fill] += point_offset
                self.maintained_parameters.pointcloud[densify_point_id] -= point_offset
            if self.config.enable_sample_from_point:
                over_reconstructed_point_id_to_fill = invalid_point_id_to_fill[over_reconstructed_mask]
                point_position = self._sample_from_point(
                    point_to_split=self.maintained_parameters.pointcloud[over_reconstructed_point_id_to_fill],
                    point_feature_to_split=self.maintained_parameters.pointcloud_features[over_reconstructed_point_id_to_fill])
                self.maintained_parameters.pointcloud[over_reconstructed_point_id_to_fill] = point_position
                
            self.maintained_parameters.point_invalid_mask[invalid_point_id_to_fill] = 0
        total_valid_points_after_densify = self.maintained_parameters.point_invalid_mask.shape[0] - \
            self.maintained_parameters.point_invalid_mask.sum()
        assert total_valid_points_after_densify == total_valid_points_before_densify - num_transparent_points - num_floaters_points + num_fillable_densify_points
        print(f"total valid points: {total_valid_points_before_densify} -> {total_valid_points_after_densify}, num_densify_points: {num_of_densify_points}, num_fillable_densify_points: {num_fillable_densify_points}")
        print(f"num_transparent_points: {num_transparent_points}, num_floaters_points: {num_floaters_points}")
        self.densify_point_info = None # clear densify point info

    def reset_alpha(self):
        pointcloud_features = self.maintained_parameters.pointcloud_features
        pointcloud_features[:, 7] = self.config.reset_alpha_value

    def _generate_point_offset(self, 
                               point_to_split: torch.Tensor, # (N, 3)
                               point_feature_to_split: torch.Tensor, # (N, 56)
    ):
        # generate extra offset for the point to split. The point is modeled as ellipsoid, with center at point_to_split, 
        # and axis length specified by s, and rotation specified by q.
        # For this solution, we want to put the two new points on the foci of the ellipsoid, so the offset
        # is the vector from the center to the foci.
        select_points = point_to_split.contiguous()
        select_point_features = point_feature_to_split.contiguous()
        point_offset = torch.zeros_like(select_points)
        compute_ellipsoid_offset(
            pointcloud=select_points,
            pointcloud_features=select_point_features,
            ellipsoid_offset=point_offset,
        )
        return point_offset
    
    def _sample_from_point(self,
                           point_to_split: torch.Tensor, # (N, 3)
                            point_feature_to_split: torch.Tensor, # (N, 56)
    ):
        select_points = point_to_split.contiguous()
        select_point_features = point_feature_to_split.contiguous()
        point_sampled = torch.zeros_like(select_points)
        sample_from_point(
            pointcloud=select_points,
            pointcloud_features=select_point_features,
            sample_result=point_sampled)
        return point_sampled
        
        
        
        
