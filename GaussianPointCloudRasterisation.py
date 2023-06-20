import numpy as np
import torch
import taichi as ti
from dataclasses import dataclass
from Camera import CameraInfo, CameraView
from torch.cuda.amp import custom_bwd, custom_fwd
from utils import (torch_type, data_type, ti2torch, torch2ti,
                   ti2torch_grad, torch2ti_grad,
                   get_ray_origin_and_direction_by_uv,
                   get_point_probability_density_from_2d_gaussian_normalized,
                   grad_point_probability_density_2d_normalized,
                   taichi_inverse_se3,
                   inverse_se3)
from GaussianPoint3D import GaussianPoint3D, project_point_to_camera
from SphericalHarmonics import SphericalHarmonics, vec16f
from typing import List, Tuple, Optional, Callable, Union
from dataclass_wizard import YAMLWizard


mat4x4f = ti.types.matrix(n=4, m=4, dtype=ti.f32)
mat4x3f = ti.types.matrix(n=4, m=3, dtype=ti.f32)

BOUNDARY_TILES = 3

@ti.kernel
def filter_point_in_camera(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    point_invalid_mask: ti.types.ndarray(ti.i8, ndim=1),  # (N)
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
        if point_invalid_mask[point_id] == 1:
            point_in_camera_mask[point_id] = ti.cast(0, ti.i8)
            continue
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
            pixel_u >= -16 * BOUNDARY_TILES and pixel_u < camera_width + 16 * BOUNDARY_TILES and \
                pixel_v >= -16 * BOUNDARY_TILES and pixel_v < camera_height + 16 * BOUNDARY_TILES:
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
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in ti.static(range(4))] for row in ti.static(range(4))])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
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
        tile_u = ti.cast(pixel_uv[0] // 16, ti.i32)
        tile_v = ti.cast(pixel_uv[1] // 16, ti.i32)
        encoded_tile_id = ti.cast(
            tile_u + tile_v * (camera_width // 16), ti.i32)
        sort_key = ti.cast(encoded_projected_depth, ti.i64) + \
            (ti.cast(encoded_tile_id, ti.i64) << 32)
        point_in_camera_sort_key[idx] = sort_key

HALF_NEIGHBOR_TILE_SIZE = 7 # in paper it is 8
@ti.kernel
def generate_num_overlap_tiles(
    num_overlap_tiles: ti.types.ndarray(ti.i64, ndim=1),  # (M)
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_uv_covariance: ti.types.ndarray(ti.f32, ndim=3),  # (M, 2, 2)
    point_alpha_after_activation: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    camera_width: ti.i32,  # required to be multiple of 16
    camera_height: ti.i32,
):
    for point_offset in range(num_overlap_tiles.shape[0]):
        uv = ti.math.vec2([point_uv[point_offset, 0],
                            point_uv[point_offset, 1]])
        uv_cov = ti.math.mat2([point_uv_covariance[point_offset, 0, 0], point_uv_covariance[point_offset, 0, 1],
                                point_uv_covariance[point_offset, 1, 0], point_uv_covariance[point_offset, 1, 1]])
        point_alpha_after_activation_value = point_alpha_after_activation[point_offset]
        
        center_tile_u = ti.cast(uv[0] // 16, ti.i32)
        center_tile_v = ti.cast(uv[1] // 16, ti.i32)
        overlap_tiles_count = 0
        # we define overlap as: the alpha at the center or four corners of the tile is larger than 1/255,
        # or the point is inside the 3x3 tile block
        for tile_u_offset in range(-HALF_NEIGHBOR_TILE_SIZE, HALF_NEIGHBOR_TILE_SIZE + 1):
            for tile_v_offset in ti.static(range(-HALF_NEIGHBOR_TILE_SIZE, HALF_NEIGHBOR_TILE_SIZE + 1)):
                tile_u = center_tile_u + tile_u_offset
                tile_v = center_tile_v + tile_v_offset
                if tile_u >= 0 and tile_u < camera_width // 16 and tile_v >= 0 and tile_v < camera_height // 16:
                    center_u = tile_u * 16 + 8
                    center_v = tile_v * 16 + 8
                    gaussian_alpha = get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([center_u, center_v]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    )
                    corner_u_0 = tile_u * 16
                    corner_v_0 = tile_v * 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_0, corner_v_0]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    corner_u_1 = tile_u * 16 + 16
                    corner_v_1 = tile_v * 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_1, corner_v_1]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    corner_u_2 = tile_u * 16
                    corner_v_2 = tile_v * 16 + 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_2, corner_v_2]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    corner_u_3 = tile_u * 16 + 16
                    corner_v_3 = tile_v * 16 + 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_3, corner_v_3]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    if gaussian_alpha * point_alpha_after_activation_value > 1 / 255. or \
                        (ti.abs(tile_u_offset) <= 1 and ti.abs(tile_v_offset) <= 1 and point_alpha_after_activation_value > 1 / 255.):
                        overlap_tiles_count += 1
        num_overlap_tiles[point_offset] = overlap_tiles_count
    
@ti.kernel
def generate_point_sort_key_by_num_overlap_tiles(
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_covariance: ti.types.ndarray(ti.f32, ndim=3),  # (M, 2, 2)
    point_alpha_after_activation: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    accumulated_num_overlap_tiles: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    point_offset_with_sort_key: ti.types.ndarray(ti.i32, ndim=1),  # (K), K = sum(num_overlap_tiles)
    point_in_camera_sort_key: ti.types.ndarray(ti.i64, ndim=1),  # (K), K = sum(num_overlap_tiles)
    camera_width: ti.i32,  # required to be multiple of 16
    camera_height: ti.i32,
    depth_to_sort_key_scale: ti.f32,
):
    for point_offset in range(accumulated_num_overlap_tiles.shape[0]):
        uv = ti.math.vec2([point_uv[point_offset, 0],
                            point_uv[point_offset, 1]])
        xyz_in_camera = ti.math.vec3(
            [point_in_camera[point_offset, 0], point_in_camera[point_offset, 1], point_in_camera[point_offset, 2]])
        uv_cov = ti.math.mat2([point_uv_covariance[point_offset, 0, 0], point_uv_covariance[point_offset, 0, 1],
                                point_uv_covariance[point_offset, 1, 0], point_uv_covariance[point_offset, 1, 1]])
        point_alpha_after_activation_value = point_alpha_after_activation[point_offset]

        point_depth = xyz_in_camera[2]
        encoded_projected_depth = ti.cast(
            point_depth * depth_to_sort_key_scale, ti.i32)       
        center_tile_u = ti.cast(uv[0] // 16, ti.i32)
        center_tile_v = ti.cast(uv[1] // 16, ti.i32)

        overlap_tiles_count = 0
        # we define overlap as: the alpha at the center or four corners of the tile is larger than 1/255,
        # or the point is inside the 3x3 tile block
        for tile_u_offset in range(-HALF_NEIGHBOR_TILE_SIZE, HALF_NEIGHBOR_TILE_SIZE + 1):
            for tile_v_offset in ti.static(range(-HALF_NEIGHBOR_TILE_SIZE, HALF_NEIGHBOR_TILE_SIZE + 1)):
                tile_u = center_tile_u + tile_u_offset
                tile_v = center_tile_v + tile_v_offset
                if tile_u >= 0 and tile_u < camera_width // 16 and tile_v >= 0 and tile_v < camera_height // 16:
                    center_u = tile_u * 16 + 8
                    center_v = tile_v * 16 + 8
                    gaussian_alpha = get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([center_u, center_v]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    )
                    corner_u_0 = tile_u * 16
                    corner_v_0 = tile_v * 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_0, corner_v_0]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    corner_u_1 = tile_u * 16 + 16
                    corner_v_1 = tile_v * 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_1, corner_v_1]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    corner_u_2 = tile_u * 16
                    corner_v_2 = tile_v * 16 + 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_2, corner_v_2]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    corner_u_3 = tile_u * 16 + 16
                    corner_v_3 = tile_v * 16 + 16
                    gaussian_alpha = ti.max(gaussian_alpha, get_point_probability_density_from_2d_gaussian_normalized(
                        xy=ti.math.vec2([corner_u_3, corner_v_3]),
                        gaussian_mean=uv,
                        gaussian_covariance=uv_cov,
                    ))
                    
                    if gaussian_alpha * point_alpha_after_activation_value > 1 / 255. or \
                        (ti.abs(tile_u_offset) <= 1 and ti.abs(tile_v_offset) <= 1 and point_alpha_after_activation_value > 1 / 255.):
                        key_idx = accumulated_num_overlap_tiles[point_offset] + overlap_tiles_count
                        encoded_tile_id = ti.cast(
                            tile_u + tile_v * (camera_width // 16), ti.i32)
                        sort_key = ti.cast(encoded_projected_depth, ti.i64) + \
                            (ti.cast(encoded_tile_id, ti.i64) << 32)
                        point_in_camera_sort_key[key_idx] = sort_key
                        point_offset_with_sort_key[key_idx] = point_offset
                        overlap_tiles_count += 1

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


@ti.func
def normalize_cov_rotation_in_pointcloud_features(
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, M)
    point_id: ti.i32,
):
    cov_rotation = ti.math.vec4(
        [pointcloud_features[point_id, offset] for offset in ti.static(range(4))])
    cov_rotation = ti.math.normalize(cov_rotation)
    for offset in ti.static(range(4)):
        pointcloud_features[point_id, offset] = cov_rotation[offset]


@ti.func
def load_point_cloud_row_into_gaussian_point_3d(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, M)
    point_id: ti.i32,
) -> GaussianPoint3D:
    translation = ti.Vector(
        [pointcloud[point_id, 0], pointcloud[point_id, 1], pointcloud[point_id, 2]])
    cov_rotation = ti.math.vec4(
        [pointcloud_features[point_id, offset] for offset in ti.static(range(4))])
    cov_scale = ti.math.vec3([pointcloud_features[point_id, offset]
                             for offset in ti.static(range(4, 4 + 3))])
    alpha = pointcloud_features[point_id, 7]
    r_feature = vec16f([pointcloud_features[point_id, offset]
                       for offset in ti.static(range(8, 8 + 16))])
    g_feature = vec16f([pointcloud_features[point_id, offset]
                       for offset in ti.static(range(24, 24 + 16))])
    b_feature = vec16f([pointcloud_features[point_id, offset]
                       for offset in ti.static(range(40, 40 + 16))])
    gaussian_point_3d = GaussianPoint3D(
        translation=translation,
        cov_rotation=cov_rotation,
        cov_scale=cov_scale,
        alpha=alpha,
        color_r=r_feature,
        color_g=g_feature,
        color_b=b_feature,
    )
    return gaussian_point_3d


@ti.kernel
def generate_point_attributes_in_camera_plane(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, M)
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
    point_id_list: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_covariance: ti.types.ndarray(ti.f32, ndim=3),  # (M, 2, 2)
    point_alpha_after_activation: ti.types.ndarray(ti.f32, ndim=1),  # (M)
):
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in ti.static(range(4))] for row in ti.static(range(4))])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
    for idx in range(point_id_list.shape[0]):
        point_id = point_id_list[idx]
        normalize_cov_rotation_in_pointcloud_features(
            pointcloud_features=pointcloud_features,
            point_id=point_id)
        gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=point_id)
        uv, xyz_in_camera = gaussian_point_3d.project_to_camera_position(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
        )
        uv_cov = gaussian_point_3d.project_to_camera_covariance(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
            translation_camera=xyz_in_camera,
        )
        point_uv[idx, 0], point_uv[idx, 1] = uv[0], uv[1]
        point_in_camera[idx, 0], point_in_camera[idx, 1], point_in_camera[idx,
                                                                          2] = xyz_in_camera[0], xyz_in_camera[1], xyz_in_camera[2]
        point_uv_covariance[idx, 0, 0], point_uv_covariance[idx, 0, 1], point_uv_covariance[idx, 1,
                                                                                            0], point_uv_covariance[idx, 1, 1] = uv_cov[0, 0], uv_cov[0, 1], uv_cov[1, 0], uv_cov[1, 1]
        point_alpha_after_activation[idx] = 1. / \
                (1. + ti.math.exp(-gaussian_point_3d.alpha))


@ti.kernel
def gaussian_point_rasterisation(
    camera_height: ti.i32,
    camera_width: ti.i32,
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, M)
    # (tiles_per_row * tiles_per_col)
    tile_points_start: ti.types.ndarray(ti.i32, ndim=1),
    # (tiles_per_row * tiles_per_col)
    tile_points_end: ti.types.ndarray(ti.i32, ndim=1),
    point_id_in_camera_list: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    point_offset_with_sort_key: ti.types.ndarray(ti.i32, ndim=1),  # (K) the offset of the point in point_id_in_camera_list
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_covariance: ti.types.ndarray(ti.f32, ndim=3),  # (M, 2, 2)
    rasterized_image: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 3)
    rasterized_depth: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    pixel_accumulated_alpha: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    # (H, W)
    pixel_offset_of_last_effective_point: ti.types.ndarray(ti.i32, ndim=2),
    pixel_valid_point_count: ti.types.ndarray(ti.i32, ndim=2),
):
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in ti.static(range(4))] for row in ti.static(range(4))])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
    T_pointcloud_camera = taichi_inverse_se3(T_camera_pointcloud_mat)
    ray_origin = ti.math.vec3(
        [T_pointcloud_camera[0, 3], T_pointcloud_camera[1, 3], T_pointcloud_camera[2, 3]])
    # taichi does not support thread block, so we just have a single thread for each pixel
    # hope the missing for shared block memory will not hurt the performance too much
    ti.loop_config(block_dim=256)
    for pixel_offset in ti.ndrange(camera_height * camera_width):
        tile_id = pixel_offset // 256
        tile_u = ti.cast(tile_id % (camera_width // 16), ti.i32)
        tile_v = ti.cast(tile_id // (camera_width // 16), ti.i32)
        pixel_offset_in_tile = pixel_offset - tile_id * 256
        pixel_offset_u_in_tile = pixel_offset_in_tile % 16
        pixel_offset_v_in_tile = pixel_offset_in_tile // 16
        pixel_u = tile_u * 16 + pixel_offset_u_in_tile
        pixel_v = tile_v * 16 + pixel_offset_v_in_tile
        start_offset = tile_points_start[tile_id]
        end_offset = tile_points_end[tile_id]
        T_i = 1.0
        accumulated_alpha = 0. # accumulated alpha is 1.0 - T_i
        accumulated_color = ti.math.vec3([0., 0., 0.])
        accumulated_depth = 0.
        depth_normalization_factor = 0.
        offset_of_last_effective_point = start_offset
        
        valid_point_count: ti.i32 = 0
        for idx_point_offset_with_sort_key in range(start_offset, end_offset):
            point_offset = point_offset_with_sort_key[idx_point_offset_with_sort_key]
            point_id = point_id_in_camera_list[point_offset]
            uv = ti.math.vec2([point_uv[point_offset, 0],
                              point_uv[point_offset, 1]])
            xyz_in_camera = ti.math.vec3(
                [point_in_camera[point_offset, 0], point_in_camera[point_offset, 1], point_in_camera[point_offset, 2]])
            depth = xyz_in_camera[2]
            uv_cov = ti.math.mat2([point_uv_covariance[point_offset, 0, 0], point_uv_covariance[point_offset, 0, 1],
                                  point_uv_covariance[point_offset, 1, 0], point_uv_covariance[point_offset, 1, 1]])
            gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
                pointcloud=pointcloud,
                pointcloud_features=pointcloud_features,
                point_id=point_id)

            gaussian_alpha = get_point_probability_density_from_2d_gaussian_normalized(
                xy=ti.math.vec2([pixel_u + 0.5, pixel_v + 0.5]),
                gaussian_mean=uv,
                gaussian_covariance=uv_cov,
            )
            point_alpha_after_activation = 1. / \
                (1. + ti.math.exp(-gaussian_point_3d.alpha))
            alpha = gaussian_alpha * point_alpha_after_activation
            # from paper: we skip any blending updates with 𝛼 < 𝜖 (we choose 𝜖 as 1
            # 255 ) and also clamp 𝛼 with 0.99 from above.
            # print(
            #     f"({pixel_v}, {pixel_u}, {point_offset}), alpha: {alpha}, accumulated_alpha: {accumulated_alpha}")
            if alpha < 1. / 255.:
                continue
            alpha = ti.min(alpha, 0.99)
            # from paper: before a Gaussian is included in the forward rasterization
            # pass, we compute the accumulated opacity if we were to include it
            # and stop front-to-back blending before it can exceed 0.9999.
            if 1 - (1 - accumulated_alpha) * (1 - alpha) >= 0.9999:
                break
            offset_of_last_effective_point = idx_point_offset_with_sort_key + 1
            
            ray_direction = gaussian_point_3d.translation - ray_origin
            color = gaussian_point_3d.get_color_by_ray(
                ray_origin=ray_origin,
                ray_direction=ray_direction,
            )
            # print(color)
            accumulated_color += color * alpha * T_i
            accumulated_depth += depth * alpha * T_i
            depth_normalization_factor += alpha * T_i
            T_i = T_i * (1 - alpha)
            accumulated_alpha = 1. - T_i
            valid_point_count += 1
        rasterized_image[pixel_v, pixel_u, 0] = accumulated_color[0]
        rasterized_image[pixel_v, pixel_u, 1] = accumulated_color[1]
        rasterized_image[pixel_v, pixel_u, 2] = accumulated_color[2]
        rasterized_depth[pixel_v, pixel_u] = accumulated_depth / ti.max(depth_normalization_factor, 1e-6)
        pixel_accumulated_alpha[pixel_v, pixel_u] = accumulated_alpha
        pixel_offset_of_last_effective_point[pixel_v,
                                             pixel_u] = offset_of_last_effective_point
        pixel_valid_point_count[pixel_v, pixel_u] = valid_point_count


@ti.func
def atomic_accumulate_grad_for_point(
    point_id: ti.i32,
    point_in_camera_grad: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    pointfeatures_grad: ti.types.ndarray(ti.f32, ndim=2),  # (M, K)
    viewspace_grad: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_viewspace_grad: ti.math.vec2,
    translation_grad: ti.math.vec3,
    gaussian_q_grad: ti.math.vec4,
    gaussian_s_grad: ti.math.vec3,
    gaussian_point_3d_alpha_grad: ti.f32,
    color_r_grad: vec16f,
    color_g_grad: vec16f,
    color_b_grad: vec16f,
):
    ti.atomic_add(viewspace_grad[point_id, 0], point_viewspace_grad[0])
    ti.atomic_add(viewspace_grad[point_id, 1], point_viewspace_grad[1])
    for offset in ti.static(range(3)):
        ti.atomic_add(point_in_camera_grad[point_id, offset],
                      translation_grad[offset])
    for offset in ti.static(range(4)):
        ti.atomic_add(pointfeatures_grad[point_id, offset],
                      gaussian_q_grad[offset])
    for offset in ti.static(range(3)):
        ti.atomic_add(pointfeatures_grad[point_id, 4 + offset],
                      gaussian_s_grad[offset])
    ti.atomic_add(pointfeatures_grad[point_id, 7],
                  gaussian_point_3d_alpha_grad)

    for offset in ti.static(range(16)):
        ti.atomic_add(pointfeatures_grad[point_id, 8 + offset],
                      color_r_grad[offset])
        ti.atomic_add(pointfeatures_grad[point_id, 24 + offset],
                      color_g_grad[offset])
        ti.atomic_add(pointfeatures_grad[point_id, 40 + offset],
                      color_b_grad[offset])


@ti.kernel
def gaussian_point_rasterisation_backward(
    camera_height: ti.i32,
    camera_width: ti.i32,
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, K)
    # (tiles_per_row * tiles_per_col)
    tile_points_start: ti.types.ndarray(ti.i32, ndim=1),
    tile_points_end: ti.types.ndarray(ti.i32, ndim=1),
    point_offset_with_sort_key: ti.types.ndarray(ti.i32, ndim=1),  # (K)
    point_id_in_camera_list: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    rasterized_image_grad: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 3)
    pixel_accumulated_alpha: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    # (H, W)
    pixel_offset_of_last_effective_point: ti.types.ndarray(ti.i32, ndim=2),
    grad_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    grad_pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, K)
    grad_uv: ti.types.ndarray(ti.f32, ndim=2),  # (N, 2)
    magnitude_grad_uv: ti.types.ndarray(ti.f32, ndim=2),  # (N, 2)
    magnitude_grad_viewspace_on_image: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 2)
    in_camera_grad_uv_cov_buffer: ti.types.ndarray(ti.f32, ndim=3),  # (M, 2, 2)
    in_camera_grad_color_buffer: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    in_camera_depth: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    in_camera_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    in_camera_num_affected_pixels: ti.types.ndarray(ti.i32, ndim=1),  # (M)
):
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in ti.static(range(4))] for row in ti.static(range(4))])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
    T_pointcloud_camera = taichi_inverse_se3(T_camera_pointcloud_mat)
    ray_origin = ti.math.vec3(
        [T_pointcloud_camera[0, 3], T_pointcloud_camera[1, 3], T_pointcloud_camera[2, 3]])

    ti.loop_config(block_dim=256)
    for pixel_offset in ti.ndrange(camera_height * camera_width):
        tile_id = pixel_offset // 256 # each block handles one tile, so tile_id is actually block_id
        thread_id = pixel_offset % 256
        tile_u = ti.cast(tile_id % (camera_width // 16), ti.i32)
        tile_v = ti.cast(tile_id // (camera_width // 16), ti.i32)
        
        start_offset = tile_points_start[tile_id]
        end_offset = tile_points_end[tile_id]
        tile_point_count = end_offset - start_offset

        
        tile_point_uv = ti.simt.block.SharedArray((256, 2), dtype=ti.f32) # 2KB shared memory
        tile_point_uv_cov = ti.simt.block.SharedArray((256, 2, 2), dtype=ti.f32) # 4KB shared memory
        tile_point_color = ti.simt.block.SharedArray((256, 3), dtype=ti.f32) # 3KB shared memory
        tile_point_alpha = ti.simt.block.SharedArray((256,), dtype=ti.f32) # 1KB shared memory
        tile_point_grad_uv = ti.simt.block.SharedArray((256, 2), dtype=ti.f32) # 2KB shared memory
        tile_point_abs_grad_uv = ti.simt.block.SharedArray((256, 2), dtype=ti.f32) # 2KB shared memory
        tile_point_grad_uv_cov = ti.simt.block.SharedArray((256, 2, 2), dtype=ti.f32) # 4KB shared memory
        tile_point_grad_color = ti.simt.block.SharedArray((256, 3), dtype=ti.f32) # 3KB shared memory
        tile_point_grad_alpha = ti.simt.block.SharedArray((256,), dtype=ti.f32) # 1KB shared memory
        tile_point_num_affected_pixels = ti.simt.block.SharedArray((256,), dtype=ti.i32) # 1KB shared memory
        

        pixel_offset_in_tile = pixel_offset - tile_id * 256
        pixel_offset_u_in_tile = pixel_offset_in_tile % 16
        pixel_offset_v_in_tile = pixel_offset_in_tile // 16
        pixel_u = tile_u * 16 + pixel_offset_u_in_tile
        pixel_v = tile_v * 16 + pixel_offset_v_in_tile
        last_effective_point = pixel_offset_of_last_effective_point[pixel_v, pixel_u]
        accumulated_alpha: ti.f32 = pixel_accumulated_alpha[pixel_v, pixel_u]
        T_i = 1.0 - accumulated_alpha # T_i = \prod_{j=1}^{i-1} (1 - a_j)
        # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} \sum_{j=i+1}^{n} c_j a_j T(j)
        # let w_i = \sum_{j=i+1}^{n} c_j a_j T(j)
        # we have w_n = 0, w_{i-1} = w_i + c_i a_i T(i)
        # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} w_i
        w_i = ti.math.vec3(0.0, 0.0, 0.0)
        
        pixel_rgb_grad = ti.math.vec3(
            rasterized_image_grad[pixel_v, pixel_u, 0], rasterized_image_grad[pixel_v, pixel_u, 1], rasterized_image_grad[pixel_v, pixel_u, 2])
        total_magnitude_grad_viewspace_on_image = ti.math.vec2(0.0, 0.0)
       
        # for inverse_point_offset in range(effective_point_count):
        # taichi only supports range() with start and end
        # for inverse_point_offset_base in range(0, tile_point_count, 256):
        num_point_blocks = (tile_point_count + 255) // 256
        for point_block_id in range(num_point_blocks):
            inverse_point_offset_base = point_block_id * 256
            block_end_idx_point_offset_with_sort_key = end_offset - inverse_point_offset_base
            block_start_idx_point_offset_with_sort_key = ti.max(block_end_idx_point_offset_with_sort_key - 256, 0)
            # in the later loop, we will handle the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key)
            # so we need to load the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key - 1]
            to_load_idx_point_offset_with_sort_key = block_end_idx_point_offset_with_sort_key - thread_id - 1
            if to_load_idx_point_offset_with_sort_key >= block_start_idx_point_offset_with_sort_key:
                to_load_point_offset = point_offset_with_sort_key[to_load_idx_point_offset_with_sort_key]
                to_load_point_id = point_id_in_camera_list[to_load_point_offset]
                to_load_gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
                    pointcloud=pointcloud,
                    pointcloud_features=pointcloud_features,
                    point_id=to_load_point_id)
                to_load_uv, to_load_translation_camera = to_load_gaussian_point_3d.project_to_camera_position(
                    T_camera_world=T_camera_pointcloud_mat,
                    projective_transform=camera_intrinsics_mat)       
                for i in ti.static(range(2)):
                    tile_point_uv[thread_id, i] = to_load_uv[i]
                    tile_point_grad_uv[thread_id, i] = 0.0
                    tile_point_abs_grad_uv[thread_id, i] = 0.0
                to_load_uv_cov = to_load_gaussian_point_3d.project_to_camera_covariance(
                    T_camera_world=T_camera_pointcloud_mat,
                    projective_transform=camera_intrinsics_mat,
                    translation_camera=to_load_translation_camera)
                for i in ti.static(range(2)):
                    for j in ti.static(range(2)):
                        tile_point_uv_cov[thread_id, i, j] = to_load_uv_cov[i, j]
                        tile_point_grad_uv_cov[thread_id, i, j] = 0.0
                to_load_ray_direction = to_load_gaussian_point_3d.translation - ray_origin
                to_load_color = to_load_gaussian_point_3d.get_color_by_ray(
                    ray_origin=ray_origin,
                    ray_direction=to_load_ray_direction,
                )
                for i in ti.static(range(3)):
                    tile_point_color[thread_id, i] = to_load_color[i]
                    tile_point_grad_color[thread_id, i] = 0.0
                
                tile_point_alpha[thread_id] = to_load_gaussian_point_3d.alpha
                tile_point_grad_alpha[thread_id] = 0.0
                
            ti.simt.block.sync()
            for inverse_point_offset_offset in range(256):
                inverse_point_offset = inverse_point_offset_base + inverse_point_offset_offset
                if inverse_point_offset >= tile_point_count:
                    break

                idx_point_offset_with_sort_key = end_offset - inverse_point_offset - 1
                if idx_point_offset_with_sort_key >= last_effective_point:
                    continue

                idx_point_offset_with_sort_key_in_block = inverse_point_offset_offset
                uv = ti.math.vec2(tile_point_uv[idx_point_offset_with_sort_key_in_block, 0], tile_point_uv[idx_point_offset_with_sort_key_in_block, 1])
                uv_cov = ti.math.mat2([
                    tile_point_uv_cov[idx_point_offset_with_sort_key_in_block, 0, 0], tile_point_uv_cov[idx_point_offset_with_sort_key_in_block, 0, 1],
                    tile_point_uv_cov[idx_point_offset_with_sort_key_in_block, 1, 0], tile_point_uv_cov[idx_point_offset_with_sort_key_in_block, 1, 1],
                ])


                point_offset = point_offset_with_sort_key[idx_point_offset_with_sort_key]
                point_id = point_id_in_camera_list[point_offset]
                gaussian_point_3d_alpha = tile_point_alpha[idx_point_offset_with_sort_key_in_block]

                # d_p_d_mean is (2,), d_p_d_cov is (2, 2), needs to be flattened to (4,)
                gaussian_alpha, d_p_d_mean, d_p_d_cov = grad_point_probability_density_2d_normalized(
                    xy=ti.math.vec2([pixel_u + 0.5, pixel_v + 0.5]),
                    gaussian_mean=uv,
                    gaussian_covariance=uv_cov,
                )
                point_alpha_after_activation = 1. / \
                    (1. + ti.math.exp(-gaussian_point_3d_alpha))
                prod_alpha = gaussian_alpha * point_alpha_after_activation
                # from paper: we skip any blending updates with 𝛼 < 𝜖 (we choose 𝜖 as 1
                # 255 ) and also clamp 𝛼 with 0.99 from above.
                if prod_alpha >= 1. / 255.:
                    alpha: ti.f32 = ti.min(prod_alpha, 0.99)
                    color = ti.math.vec3([
                        tile_point_color[idx_point_offset_with_sort_key_in_block, 0], 
                        tile_point_color[idx_point_offset_with_sort_key_in_block, 1], 
                        tile_point_color[idx_point_offset_with_sort_key_in_block, 2]])

                    T_i = T_i / (1. - alpha)
                    accumulated_alpha = 1. - T_i

                    # print(
                    #     f"({pixel_v}, {pixel_u}, {point_offset}, {point_offset - start_offset}), accumulated_alpha: {accumulated_alpha}")

                    d_pixel_rgb_d_color = alpha * T_i
                    point_grad_color = d_pixel_rgb_d_color * pixel_rgb_grad
                    
                    # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} w_i
                    alpha_grad_from_rgb = (color * T_i - w_i / (1. - alpha)) \
                        * pixel_rgb_grad
                    # w_{i-1} = w_i + c_i a_i T(i)
                    w_i += color * alpha * T_i
                    alpha_grad: ti.f32 = alpha_grad_from_rgb[0] + \
                        alpha_grad_from_rgb[1] + alpha_grad_from_rgb[2]
                    point_alpha_after_activation_grad = alpha_grad * gaussian_alpha
                    gaussian_point_3d_alpha_grad = point_alpha_after_activation_grad * \
                        (1. - point_alpha_after_activation) * \
                        point_alpha_after_activation
                    gaussian_alpha_grad = alpha_grad * point_alpha_after_activation
                    # gaussian_alpha_grad is dp
                    point_viewspace_grad = gaussian_alpha_grad * \
                        d_p_d_mean  # (2,) as the paper said, view space gradient is used for detect candidates for densification
                    total_magnitude_grad_viewspace_on_image += ti.abs(point_viewspace_grad)
                    point_uv_cov_grad = gaussian_alpha_grad * d_p_d_cov # (2, 2)
                    
                    # atomic accumulate on block shared memory shall be faster
                    for i in ti.static(range(2)):
                        ti.atomic_add(tile_point_grad_uv[idx_point_offset_with_sort_key_in_block, i], point_viewspace_grad[i])
                        ti.atomic_add(tile_point_abs_grad_uv[idx_point_offset_with_sort_key_in_block, i], ti.abs(point_viewspace_grad[i]))
                    for i in ti.static(range(2)):
                        for j in ti.static(range(2)):
                            ti.atomic_add(tile_point_grad_uv_cov[idx_point_offset_with_sort_key_in_block, i, j], point_uv_cov_grad[i, j])
                    for i in ti.static(range(3)):
                        ti.atomic_add(tile_point_grad_color[idx_point_offset_with_sort_key_in_block, i], point_grad_color[i])
                    ti.atomic_add(tile_point_grad_alpha[idx_point_offset_with_sort_key_in_block], gaussian_point_3d_alpha_grad)
                    ti.atomic_add(tile_point_num_affected_pixels[idx_point_offset_with_sort_key_in_block], 1)
            # end of the 256 block loop
            ti.simt.block.sync()
            # if the thread load the point, then it shall save the gradient of the point
            if to_load_idx_point_offset_with_sort_key >= block_start_idx_point_offset_with_sort_key:
                to_save_point_offset = point_offset_with_sort_key[to_load_idx_point_offset_with_sort_key]
                to_save_point_id = point_id_in_camera_list[to_save_point_offset]
                for i in ti.static(range(2)):
                    # no further process needed for abs grad, so directly save to global memory
                    ti.atomic_add(grad_uv[to_save_point_id, i], tile_point_grad_uv[thread_id, i])
                    ti.atomic_add(magnitude_grad_uv[to_save_point_id, i], tile_point_abs_grad_uv[thread_id, i])
                for i in ti.static(range(2)):
                    for j in ti.static(range(2)):
                        ti.atomic_add(in_camera_grad_uv_cov_buffer[to_save_point_offset, i, j], tile_point_grad_uv_cov[thread_id, i, j])
                for i in ti.static(range(3)):
                    ti.atomic_add(in_camera_grad_color_buffer[to_save_point_offset, i], tile_point_grad_color[thread_id, i])
                # no further process needed for alpha grad, so directly save to global memory
                ti.atomic_add(grad_pointcloud_features[to_save_point_id, 7], tile_point_grad_alpha[thread_id])
                # no further process needed for num_affected_pixels, so directly save to global memory
                ti.atomic_add(in_camera_num_affected_pixels[to_save_point_offset], tile_point_num_affected_pixels[thread_id])
            ti.simt.block.sync() # sync the block, so that the next block can read the data into shared memory without issue
        # end of the backward traversal loop, from last point to first point                    
        magnitude_grad_viewspace_on_image[pixel_v, pixel_u, 0] = total_magnitude_grad_viewspace_on_image[0]
        magnitude_grad_viewspace_on_image[pixel_v, pixel_u, 1] = total_magnitude_grad_viewspace_on_image[1]
    # end of per pixel loop

    # one more loop to compute the gradient from viewspace to 3D point
    for idx in range(point_id_in_camera_list.shape[0]):
        point_id = point_id_in_camera_list[idx]
        gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=point_id)
        point_grad_uv = ti.math.vec2(grad_uv[point_id, 0], grad_uv[point_id, 1])
        point_grad_uv_cov_flat = ti.math.vec4(
            in_camera_grad_uv_cov_buffer[idx, 0, 0],
            in_camera_grad_uv_cov_buffer[idx, 0, 1],
            in_camera_grad_uv_cov_buffer[idx, 1, 0],
            in_camera_grad_uv_cov_buffer[idx, 1, 1],
        )
        point_grad_color = ti.math.vec3(
            in_camera_grad_color_buffer[idx, 0],
            in_camera_grad_color_buffer[idx, 1],
            in_camera_grad_color_buffer[idx, 2],
        )
        uv, translation_camera = gaussian_point_3d.project_to_camera_position(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
        )
        depth = translation_camera[2]
        in_camera_depth[idx] = depth
        in_camera_uv[idx, 0] = uv[0]
        in_camera_uv[idx, 1] = uv[1]
        uv_cov = gaussian_point_3d.project_to_camera_covariance(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
            translation_camera=translation_camera)
        d_uv_d_translation = gaussian_point_3d.project_to_camera_position_jacobian(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
        )  # (2, 3)
        d_Sigma_prime_d_q, d_Sigma_prime_d_s = gaussian_point_3d.project_to_camera_covariance_jacobian(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
            translation_camera=translation_camera,
        )
        ray_direction = gaussian_point_3d.translation - ray_origin
        _, r_jacobian, g_jacobian, b_jacobian = gaussian_point_3d.get_color_with_jacobian_by_ray(
            ray_origin=ray_origin,
            ray_direction=ray_direction,
        )
        color_r_grad = point_grad_color[0] * r_jacobian
        color_g_grad = point_grad_color[1] * g_jacobian
        color_b_grad = point_grad_color[2] * b_jacobian
        translation_grad = point_grad_uv @ d_uv_d_translation

        # cov is Sigma
        gaussian_q_grad = point_grad_uv_cov_flat @ d_Sigma_prime_d_q
        gaussian_s_grad = point_grad_uv_cov_flat @ d_Sigma_prime_d_s

        for i in ti.static(range(3)):
            grad_pointcloud[point_id, i] = translation_grad[i]
        for i in ti.static(range(4)):
            grad_pointcloud_features[point_id, i] = gaussian_q_grad[i]
        for i in ti.static(range(3)):
            grad_pointcloud_features[point_id, i + 4] = gaussian_s_grad[i]
        for i in ti.static(range(16)):
            grad_pointcloud_features[point_id, i + 8] = color_r_grad[i]
            grad_pointcloud_features[point_id, i + 24] = color_g_grad[i]
            grad_pointcloud_features[point_id, i + 40] = color_b_grad[i]


class GaussianPointCloudRasterisation(torch.nn.Module):
    @dataclass
    class GaussianPointCloudRasterisationConfig(YAMLWizard):
        near_plane: float = 0.8
        far_plane: float = 1000.
        depth_to_sort_key_scale: float = 100.
        grad_color_factor = 5.
        grad_s_factor = 0.5
        grad_q_factor = 1.
        grad_alpha_factor = 1.

    @dataclass
    class GaussianPointCloudRasterisationInput:
        point_cloud: torch.Tensor  # Nx3
        point_cloud_features: torch.Tensor  # NxM
        point_invalid_mask: torch.Tensor  # N
        camera_info: CameraInfo
        T_pointcloud_camera: torch.Tensor  # 4x4 x to the right, y down, z forward
        color_max_sh_band: int = 2

    @dataclass
    class BackwardValidPointHookInput:
        point_id_in_camera_list: torch.Tensor  # M
        grad_point_in_camera: torch.Tensor  # Mx3
        grad_pointfeatures_in_camera: torch.Tensor  # Mx56
        grad_viewspace: torch.Tensor  # Mx2
        magnitude_grad_viewspace: torch.Tensor  # M x 2
        magnitude_grad_viewspace_on_image: torch.Tensor  # HxWx2
        num_overlap_tiles: torch.Tensor  # M
        num_affected_pixels: torch.Tensor  # M
        point_depth: torch.Tensor  # M
        point_uv_in_camera: torch.Tensor  # Mx2

    def __init__(
        self,
        config: GaussianPointCloudRasterisationConfig,
        backward_valid_point_hook: Optional[Callable[[
            BackwardValidPointHookInput], None]] = None,
    ):
        super().__init__()
        self.config = config

        class _module_function(torch.autograd.Function):

            @staticmethod
            @custom_fwd(cast_inputs=torch_type)
            def forward(ctx, pointcloud, pointcloud_features, point_invalid_mask, T_pointcloud_camera, camera_info, color_max_sh_band):
                point_in_camera_mask = torch.zeros(
                    size=(pointcloud.shape[0],), dtype=torch.int8, device=pointcloud.device)
                point_id = torch.arange(
                    pointcloud.shape[0], dtype=torch.int32, device=pointcloud.device)
                T_camera_pointcloud = inverse_se3(T_pointcloud_camera)
                filter_point_in_camera(
                    pointcloud=pointcloud,
                    point_invalid_mask=point_invalid_mask,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    T_camera_pointcloud=T_camera_pointcloud,
                    point_in_camera_mask=point_in_camera_mask,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    camera_height=camera_info.camera_height,
                    camera_width=camera_info.camera_width,
                )
                point_in_camera_mask = point_in_camera_mask.bool()
                point_id_in_camera_list = point_id[point_in_camera_mask].contiguous(
                )
                del point_id
                del point_in_camera_mask

                num_points_in_camera = point_id_in_camera_list.shape[0]

                point_uv = torch.zeros(
                    size=(num_points_in_camera, 2), dtype=torch.float32, device=pointcloud.device)
                point_alpha_after_activation = torch.zeros(
                    size=(num_points_in_camera,), dtype=torch.float32, device=pointcloud.device)
                point_in_camera = torch.zeros(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_uv_covariance = torch.zeros(
                    size=(num_points_in_camera, 2, 2), dtype=torch.float32, device=pointcloud.device)

                generate_point_attributes_in_camera_plane(
                    pointcloud=pointcloud,
                    pointcloud_features=pointcloud_features,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    T_camera_pointcloud=T_camera_pointcloud,
                    point_id_list=point_id_in_camera_list,
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_covariance=point_uv_covariance,
                    point_alpha_after_activation=point_alpha_after_activation,
                )

                num_overlap_tiles = torch.zeros_like(point_id_in_camera_list)
                generate_num_overlap_tiles(
                    num_overlap_tiles=num_overlap_tiles,
                    point_uv=point_uv,
                    point_uv_covariance=point_uv_covariance,
                    point_alpha_after_activation=point_alpha_after_activation,
                    camera_width=camera_info.camera_width,
                    camera_height=camera_info.camera_height,
                )
                accumulated_num_overlap_tiles = torch.cumsum(
                    num_overlap_tiles, dim=0)
                total_num_overlap_tiles = accumulated_num_overlap_tiles[-1]
                accumulated_num_overlap_tiles = torch.cat(
                    (torch.zeros(size=(1,), dtype=torch.int32, device=pointcloud.device), 
                     accumulated_num_overlap_tiles[:-1]))
                # del num_overlap_tiles
                point_in_camera_sort_key = torch.zeros(
                    size=(total_num_overlap_tiles,), dtype=torch.int64, device=pointcloud.device)
                point_offset_with_sort_key = torch.zeros(
                    size=(total_num_overlap_tiles,), dtype=torch.int32, device=pointcloud.device)
                generate_point_sort_key_by_num_overlap_tiles(
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_covariance=point_uv_covariance,
                    point_alpha_after_activation=point_alpha_after_activation,
                    accumulated_num_overlap_tiles=accumulated_num_overlap_tiles,
                    point_offset_with_sort_key=point_offset_with_sort_key,
                    point_in_camera_sort_key=point_in_camera_sort_key,
                    camera_width=camera_info.camera_width,
                    camera_height=camera_info.camera_height,
                    depth_to_sort_key_scale=self.config.depth_to_sort_key_scale,
                )

                point_in_camera_sort_key, permutation = point_in_camera_sort_key.sort()
                point_offset_with_sort_key = point_offset_with_sort_key[permutation].contiguous(
                )  # now the point_offset_with_sort_key is sorted by the sort_key
                del permutation
                tiles_per_row = camera_info.camera_width // 16
                tiles_per_col = camera_info.camera_height // 16
                tile_points_start = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                tile_points_end = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                find_tile_start_and_end(
                    point_in_camera_sort_key=point_in_camera_sort_key,
                    tile_points_start=tile_points_start,
                    tile_points_end=tile_points_end,
                )

                

                rasterized_image = torch.zeros(
                    camera_info.camera_height, camera_info.camera_width, 3, dtype=torch.float32, device=pointcloud.device)
                rasterized_depth = torch.zeros(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_accumulated_alpha = torch.zeros(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_offset_of_last_effective_point = torch.zeros(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                pixel_valid_point_count = torch.zeros(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                

                gaussian_point_rasterisation(
                    camera_height=camera_info.camera_height,
                    camera_width=camera_info.camera_width,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    T_camera_pointcloud=T_camera_pointcloud,
                    pointcloud=pointcloud,
                    pointcloud_features=pointcloud_features,
                    tile_points_start=tile_points_start,
                    tile_points_end=tile_points_end,
                    point_id_in_camera_list=point_id_in_camera_list,
                    point_offset_with_sort_key=point_offset_with_sort_key,
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_covariance=point_uv_covariance,
                    rasterized_image=rasterized_image,
                    rasterized_depth=rasterized_depth,
                    pixel_accumulated_alpha=pixel_accumulated_alpha,
                    pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                    pixel_valid_point_count=pixel_valid_point_count)
                ctx.save_for_backward(
                    pointcloud,
                    pointcloud_features,
                    point_offset_with_sort_key, # point_id_with_sort_key is sorted by tile and depth and has duplicated points, e.g. one points is belong to multiple tiles
                    point_id_in_camera_list, # point_in_camera_id does not have duplicated points
                    tile_points_start,
                    tile_points_end,
                    pixel_accumulated_alpha,
                    pixel_offset_of_last_effective_point,
                    T_pointcloud_camera,
                    T_camera_pointcloud,
                    num_overlap_tiles
                )
                ctx.camera_info = camera_info
                ctx.color_max_sh_band = color_max_sh_band
                # rasterized_image.requires_grad_(True)
                return rasterized_image, rasterized_depth, pixel_valid_point_count

            @staticmethod
            @custom_bwd
            def backward(ctx, grad_rasterized_image, grad_rasterized_depth, grad_pixel_valid_point_count):
                grad_pointcloud = grad_pointcloud_features = grad_T_pointcloud_camera = None
                if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                    pointcloud, pointcloud_features, point_offset_with_sort_key, point_id_in_camera_list, tile_points_start, tile_points_end, pixel_accumulated_alpha, pixel_offset_of_last_effective_point, T_pointcloud_camera, T_camera_pointcloud, num_overlap_tiles = ctx.saved_tensors
                    camera_info = ctx.camera_info
                    color_max_sh_band = ctx.color_max_sh_band
                    grad_rasterized_image = grad_rasterized_image.contiguous()
                    grad_pointcloud = torch.zeros_like(pointcloud)
                    grad_pointcloud_features = torch.zeros_like(
                        pointcloud_features)
                    
                    grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], 2), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], 2), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace_on_image = torch.zeros_like(
                        grad_rasterized_image[:, :, :2])
                    
                    in_camera_grad_uv_cov_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 2, 2), dtype=torch.float32, device=pointcloud.device)
                    in_camera_grad_color_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    in_camera_depth = torch.zeros(
                        size=(point_id_in_camera_list.shape[0],), dtype=torch.float32, device=pointcloud.device)
                    in_camera_uv = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 2), dtype=torch.float32, device=pointcloud.device)
                    in_camera_num_affected_pixels = torch.zeros(
                        size=(point_id_in_camera_list.shape[0],), dtype=torch.int32, device=pointcloud.device)
                    
                    gaussian_point_rasterisation_backward(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        camera_intrinsics=camera_info.camera_intrinsics,
                        T_camera_pointcloud=T_camera_pointcloud,
                        pointcloud=pointcloud,
                        pointcloud_features=pointcloud_features,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                        point_offset_with_sort_key=point_offset_with_sort_key,
                        point_id_in_camera_list=point_id_in_camera_list,
                        rasterized_image_grad=grad_rasterized_image,
                        pixel_accumulated_alpha=pixel_accumulated_alpha,
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                        grad_pointcloud=grad_pointcloud,
                        grad_pointcloud_features=grad_pointcloud_features,
                        grad_uv=grad_viewspace,
                        magnitude_grad_uv=magnitude_grad_viewspace,
                        magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image,
                        in_camera_grad_uv_cov_buffer=in_camera_grad_uv_cov_buffer,
                        in_camera_grad_color_buffer=in_camera_grad_color_buffer,
                        in_camera_depth=in_camera_depth,
                        in_camera_uv=in_camera_uv,
                        in_camera_num_affected_pixels=in_camera_num_affected_pixels,
                    )
                    del tile_points_start, tile_points_end, pixel_accumulated_alpha, pixel_offset_of_last_effective_point
                    grad_pointcloud_features = self._clear_grad_by_color_max_sh_band(
                        grad_pointcloud_features=grad_pointcloud_features,
                        color_max_sh_band=color_max_sh_band)
                    grad_pointcloud_features[:, :4] *= self.config.grad_q_factor
                    grad_pointcloud_features[:, 4:7] *= self.config.grad_s_factor
                    grad_pointcloud_features[:, 7] *= self.config.grad_alpha_factor
                    grad_pointcloud_features[:, 8:] *= self.config.grad_color_factor
                    
                    if backward_valid_point_hook is not None:
                        backward_valid_point_hook_input = GaussianPointCloudRasterisation.BackwardValidPointHookInput(
                            point_id_in_camera_list=point_id_in_camera_list,
                            grad_point_in_camera=grad_pointcloud[point_id_in_camera_list],
                            grad_pointfeatures_in_camera=grad_pointcloud_features[point_id_in_camera_list],
                            grad_viewspace=grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace=magnitude_grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image,
                            num_overlap_tiles=num_overlap_tiles,
                            num_affected_pixels=in_camera_num_affected_pixels,
                            point_uv_in_camera=in_camera_uv,
                            point_depth=in_camera_depth,
                        )
                        backward_valid_point_hook(
                            backward_valid_point_hook_input)
                    
                    
                return grad_pointcloud, grad_pointcloud_features, None, grad_T_pointcloud_camera, None, None

        self._module_function = _module_function

    def _clear_grad_by_color_max_sh_band(self, grad_pointcloud_features: torch.Tensor, color_max_sh_band: int):
        if color_max_sh_band == 0:
            grad_pointcloud_features[:, 8 + 1: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 1: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 1: 40 + 16] = 0.
        elif color_max_sh_band == 1:
            grad_pointcloud_features[:, 8 + 4: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 4: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 4: 40 + 16] = 0.
        elif color_max_sh_band == 2:
            grad_pointcloud_features[:, 8 + 9: 8 + 16] = 0.
            grad_pointcloud_features[:, 24 + 9: 24 + 16] = 0.
            grad_pointcloud_features[:, 40 + 9: 40 + 16] = 0.
        elif color_max_sh_band >= 3:
            pass
        return grad_pointcloud_features

    def forward(self, input_data: GaussianPointCloudRasterisationInput):
        pointcloud = input_data.point_cloud
        pointcloud_features = input_data.point_cloud_features
        point_invalid_mask = input_data.point_invalid_mask
        T_pointcloud_camera = input_data.T_pointcloud_camera
        color_max_sh_band = input_data.color_max_sh_band
        camera_info = input_data.camera_info
        assert camera_info.camera_width % 16 == 0
        assert camera_info.camera_height % 16 == 0
        return self._module_function.apply(
            pointcloud,
            pointcloud_features,
            point_invalid_mask,
            T_pointcloud_camera,
            camera_info,
            color_max_sh_band)
