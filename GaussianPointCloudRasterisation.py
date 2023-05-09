import torch
import taichi as ti
from dataclasses import dataclass
from Camera import CameraInfo, CameraView
from torch.cuda.amp import custom_bwd, custom_fwd
from utils import torch_type, data_type, ti2torch, torch2ti, ti2torch_grad, torch2ti_grad
from GaussianPoint3D import GaussianPoint3D, project_point_to_camera


@ti.kernel
def filter_point_in_camera(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
    point_in_camera_mask: ti.types.ndarray(ti.i8, ndim=1),  # (N)
    near_plane: ti.f32,
    far_plane: ti.f32,
    camera_width: ti.i32,
    camera_height: ti.i32,
):
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in range(4)] for row in range(4)])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in range(3)] for row in range(3)])

    # filter points in camera
    for point_id in range(pointcloud.shape[0]):
        point_xyz = ti.Vector(
            [pointcloud[point_id, 0], pointcloud[point_id, 1], pointcloud[point_id, 2]])
        pixel_uv, point_in_camera = project_point_to_camera(
            translation=point_xyz,
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
        )
        pixel_u = pixel_uv[0]
        pixel_v = pixel_uv[1]
        depth_in_camera = point_in_camera[2]
        if depth_in_camera > near_plane and \
            depth_in_camera < far_plane and \
            pixel_u >= 0 and pixel_u < camera_width and \
                pixel_v >= 0 and pixel_v < camera_height:
            point_in_camera_mask[point_id] = ti.cast(1, ti.i8)
        else:
            point_in_camera_mask[point_id] = ti.cast(0, ti.i8)


@ti.kernel
def generate_point_sort_key(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
    point_in_camera_id: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    point_in_camera_sort_key: ti.types.ndarray(ti.i64, ndim=1),  # (M)
    camera_width: ti.i32,  # required to be multiple of 16
    camera_height: ti.i32,
    depth_to_sort_key_scale: ti.f32,
):
    # we do not save the point_uv and point_in_camera here to save GPU memory. Re-compute should be fast enough.
    # if we save them, we will need to permute them according to the sort_key.
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in range(4)] for row in range(4)])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in range(3)] for row in range(3)])
    for idx in range(point_in_camera_id.shape[0]):
        point_id = point_in_camera_id[idx]
        point_xyz = ti.Vector(
            [pointcloud[point_id, 0], pointcloud[point_id, 1], pointcloud[point_id, 2]])
        pixel_uv, point_in_camera = project_point_to_camera(
            translation=point_xyz,
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
        )
        point_depth = point_in_camera.z
        # as the paper said:  the lower 32 bits encode its projected depth and the higher bits encode the index of the overlapped tile.
        encoded_projected_depth = ti.cast(
            point_depth * depth_to_sort_key_scale, ti.i32)
        tile_u = ti.cast(pixel_uv[0] / 16, ti.i32)
        tile_v = ti.cast(pixel_uv[1] / 16, ti.i32)
        encoded_tile_id = ti.cast(
            tile_u + tile_v * (camera_width // 16), ti.i32)
        sort_key = ti.cast(encoded_projected_depth, ti.i64) + \
            (ti.cast(encoded_tile_id, ti.i64) << 32)
        point_in_camera_sort_key[idx] = sort_key


@ti.kernel
def find_tile_start_and_end(
    point_in_camera_sort_key: ti.types.ndarray(ti.i64, ndim=1),  # (M)
    # (tiles_per_row * tiles_per_col), for output
    tile_points_start: ti.types.ndarray(ti.i32, ndim=1),
    # (tiles_per_row * tiles_per_col), for output
    tile_points_end: ti.types.ndarray(ti.i32, ndim=1),
):
    for idx in range(point_in_camera_sort_key.shape[0] - 1):
        sort_key = point_in_camera_sort_key[idx]
        tile_id = ti.cast(sort_key >> 32, ti.i32)
        next_sort_key = point_in_camera_sort_key[idx + 1]
        next_tile_id = ti.cast(next_sort_key >> 32, ti.i32)
        if tile_id != next_tile_id:
            tile_points_start[next_tile_id] = idx + 1
            tile_points_end[tile_id] = idx + 1
    last_sort_key = point_in_camera_sort_key[point_in_camera_sort_key.shape[0] - 1]
    last_tile_id = ti.cast(last_sort_key >> 32, ti.i32)
    tile_points_end[last_tile_id] = point_in_camera_sort_key.shape[0]


class GaussianPointCloudRasterisation(torch.nn.Module):
    @dataclass
    class GaussianPointCloudRasterisationConfig:
        near_plane: float = 0.8
        far_plane: float = 1000.
        depth_to_sort_key_scale: float = 100.

    @dataclass
    class GaussianPointCloudRasterisationInput:
        point_cloud: torch.Tensor  # Nx3
        point_cloud_features: torch.Tensor  # NxM
        camera_info: CameraInfo
        T_pointcloud_camera: torch.Tensor  # 4x4

    def __init__(
        self,
        config: GaussianPointCloudRasterisationConfig,
    ):
        super().__init__()
        self.config = config

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, pointcloud, pointcloud_features, T_pointcloud_camera, camera_info):
                point_in_camera_mask = torch.zeros(
                    size=(pointcloud.shape[0],), dtype=torch.bool, device=pointcloud.device)
                point_id = torch.arange(
                    pointcloud.shape[0], dtype=torch.int32, device=pointcloud.device)
                T_camera_pointcloud = torch.inverse(T_pointcloud_camera)
                filter_point_in_camera(
                    pointcloud=pointcloud,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    T_camera_pointcloud=T_camera_pointcloud,
                    point_in_camera_mask=point_in_camera_mask,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    camera_height=camera_info.camera_height,
                    camera_width=camera_info.camera_width,
                )
                point_in_camera_id = point_id[point_in_camera_mask].contiguous(
                )
                point_in_camera_sort_key = torch.zeros(
                    size=(point_in_camera_id.shape[0],), dtype=torch.int64, device=pointcloud.device)
                generate_point_sort_key(
                    pointcloud=pointcloud,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    T_camera_pointcloud=T_camera_pointcloud,
                    point_in_camera_id=point_in_camera_id,
                    point_in_camera_sort_key=point_in_camera_sort_key,
                    camera_height=camera_info.camera_height,
                    camera_width=camera_info.camera_width,
                    depth_to_sort_key_scale=self.config.depth_to_sort_key_scale,
                )
                permutation = point_in_camera_sort_key.argsort()
                point_in_camera_id = point_in_camera_id[permutation].contiguous(
                )  # now the point_in_camera_id is sorted by the sort_key
                del permutation
                tiles_per_row = camera_info.camera_width // 16
                tiles_per_col = camera_info.camera_height // 16
                tile_points_start = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                tile_points_end = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)

            @staticmethod
            @custom_bwd
            def backward(ctx, doutput):

                self.zero_grad()

                self.torch2ti_grad(self.output_fields, doutput.contiguous())
                self._hash_encode_kernel.grad(
                    self.input_fields,
                    self.parameter_fields,
                    self.output_fields,
                    self.hash_map_indicator,
                    self.hash_map_sizes_field,
                    self.offsets,
                    doutput.shape[0],
                    self.per_level_scale,
                )
                self.ti2torch_grad(self.parameter_fields,
                                   self.hash_grad.contiguous())
                return None, self.hash_grad

        self._module_function = _module_function

    def zero_grad(self):
        self.parameter_fields.grad.fill(0.)

    def forward(self, positions):
        return

    def forward(self, input_data: GaussianPointCloudRasterisationInput):
        pointcloud = input_data.point_cloud
        pointcloud_features = input_data.point_cloud_features
        T_pointcloud_camera = input_data.T_pointcloud_camera
        camera_info = input_data.camera_info
        assert camera_info.camera_width % 16 == 0
        assert camera_info.camera_height % 16 == 0
        return self._module_function.apply(pointcloud, pointcloud_features, T_pointcloud_camera, camera_info)
