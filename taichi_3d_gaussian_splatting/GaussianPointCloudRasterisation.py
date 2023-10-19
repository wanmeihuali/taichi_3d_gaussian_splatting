import torch
import taichi as ti
from dataclasses import dataclass
from .Camera import CameraInfo, CameraView
from torch.cuda.amp import custom_bwd, custom_fwd
from .utils import (torch_type, data_type, ti2torch, torch2ti,
                    ti2torch_grad, torch2ti_grad,
                    get_ray_origin_and_direction_by_uv,
                    get_point_probability_density_from_2d_gaussian_normalized,
                    grad_point_probability_density_2d_normalized,
                    taichi_inverse_SE3,
                    inverse_SE3_qt_torch,
                    get_point_conic,
                    get_point_probability_density_from_conic,
                    grad_point_probability_density_from_conic,
                    inverse_SE3)
from .GaussianPoint3D import GaussianPoint3D, project_point_to_camera, rotation_matrix_from_quaternion, transform_matrix_from_quaternion_and_translation
from .SphericalHarmonics import SphericalHarmonics, vec16f
from typing import List, Tuple, Optional, Callable, Union
from dataclass_wizard import YAMLWizard


mat4x4f = ti.types.matrix(n=4, m=4, dtype=ti.f32)
mat4x3f = ti.types.matrix(n=4, m=3, dtype=ti.f32)

BOUNDARY_TILES = 3
TILE_WIDTH = 16
TILE_HEIGHT = 16


@ti.kernel
def filter_point_in_camera(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    point_invalid_mask: ti.types.ndarray(ti.i8, ndim=1),  # (N)
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    # (N), every element is in [0, K-1] corresponding to the camera id
    point_object_id: ti.types.ndarray(ti.i32, ndim=1),
    q_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 4)
    t_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 3)
    point_in_camera_mask: ti.types.ndarray(ti.i8, ndim=1),  # (N), output
    near_plane: ti.f32,
    far_plane: ti.f32,
    camera_width: ti.i32,
    camera_height: ti.i32,
):
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in range(3)] for row in range(3)])

    # filter points in camera
    for point_id in range(pointcloud.shape[0]):
        if point_invalid_mask[point_id] == 1:
            point_in_camera_mask[point_id] = ti.cast(0, ti.i8)
            continue
        point_xyz = ti.Vector(
            [pointcloud[point_id, 0], pointcloud[point_id, 1], pointcloud[point_id, 2]])
        point_q_camera_pointcloud = ti.Vector(
            [q_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(4))])
        point_t_camera_pointcloud = ti.Vector(
            [t_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(3))])
        T_camera_pointcloud_mat = transform_matrix_from_quaternion_and_translation(
            q=point_q_camera_pointcloud,
            t=point_t_camera_pointcloud,
        )
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
            pixel_u >= -TILE_WIDTH * BOUNDARY_TILES and pixel_u < camera_width + TILE_WIDTH * BOUNDARY_TILES and \
                pixel_v >= -TILE_HEIGHT * BOUNDARY_TILES and pixel_v < camera_height + TILE_HEIGHT * BOUNDARY_TILES:
            point_in_camera_mask[point_id] = ti.cast(1, ti.i8)
        else:
            point_in_camera_mask[point_id] = ti.cast(0, ti.i8)


@ti.func
def get_bounding_box_by_point_and_radii(
    uv: ti.math.vec2,  # (2)
    radii: ti.f32,  # scalar
    camera_width: ti.i32,
    camera_height: ti.i32,
):
    radii = ti.max(radii, 1.0)  # avoid zero radii, at least 1 pixel
    min_u = ti.max(0.0, uv[0] - radii)
    max_u = uv[0] + radii
    min_v = ti.max(0.0, uv[1] - radii)
    max_v = uv[1] + radii
    min_tile_u = ti.cast(min_u // TILE_WIDTH, ti.i32)
    min_tile_u = ti.min(min_tile_u, camera_width // TILE_WIDTH)
    max_tile_u = ti.cast(max_u // TILE_WIDTH, ti.i32) + 1
    max_tile_u = ti.min(ti.max(max_tile_u, min_tile_u + 1),
                        camera_width // TILE_WIDTH)
    min_tile_v = ti.cast(min_v // TILE_HEIGHT, ti.i32)
    min_tile_v = ti.min(min_tile_v, camera_height // TILE_HEIGHT)
    max_tile_v = ti.cast(max_v // TILE_HEIGHT, ti.i32) + 1
    max_tile_v = ti.min(ti.max(max_tile_v, min_tile_v + 1),
                        camera_height // TILE_HEIGHT)
    return min_tile_u, max_tile_u, min_tile_v, max_tile_v


@ti.kernel
def generate_num_overlap_tiles(
    # (M)， output，a number for each point, count how many tiles the point can be projected in.
    num_overlap_tiles: ti.types.ndarray(ti.i32, ndim=1),
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_radii: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    camera_width: ti.i32,  # required to be multiple of 16
    camera_height: ti.i32,
):
    for point_offset in range(num_overlap_tiles.shape[0]):
        uv = ti.math.vec2([point_uv[point_offset, 0],
                           point_uv[point_offset, 1]])
        radii = point_radii[point_offset]

        min_tile_u, max_tile_u, min_tile_v, max_tile_v = get_bounding_box_by_point_and_radii(
            uv=uv,
            radii=radii,
            camera_width=camera_width,
            camera_height=camera_height,
        )
        overlap_tiles_count = (max_tile_u - min_tile_u) * \
            (max_tile_v - min_tile_v)
        num_overlap_tiles[point_offset] = overlap_tiles_count


@ti.kernel
def generate_point_sort_key_by_num_overlap_tiles(
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_radii: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    accumulated_num_overlap_tiles: ti.types.ndarray(ti.i64, ndim=1),  # (M)
    # (K), K = sum(num_overlap_tiles)
    point_offset_with_sort_key: ti.types.ndarray(ti.i32, ndim=1),
    # (K), K = sum(num_overlap_tiles)
    point_in_camera_sort_key: ti.types.ndarray(ti.i64, ndim=1),
    camera_width: ti.i32,  # required to be multiple of 16
    camera_height: ti.i32,
    depth_to_sort_key_scale: ti.f32,
):
    for point_offset in range(accumulated_num_overlap_tiles.shape[0]):
        uv = ti.math.vec2([point_uv[point_offset, 0],
                           point_uv[point_offset, 1]])
        xyz_in_camera = ti.math.vec3(
            [point_in_camera[point_offset, 0], point_in_camera[point_offset, 1], point_in_camera[point_offset, 2]])
        radii = point_radii[point_offset]
        min_tile_u, max_tile_u, min_tile_v, max_tile_v = get_bounding_box_by_point_and_radii(
            uv=uv,
            radii=radii,
            camera_width=camera_width,
            camera_height=camera_height,
        )

        point_depth = xyz_in_camera[2]
        encoded_projected_depth = ti.cast(
            point_depth * depth_to_sort_key_scale, ti.i32)
        for tile_u in range(min_tile_u, max_tile_u):
            for tile_v in range(min_tile_v, max_tile_v):
                overlap_tiles_count = (max_tile_v - min_tile_v) * \
                    (tile_u - min_tile_u) + (tile_v - min_tile_v)
                key_idx = accumulated_num_overlap_tiles[point_offset] + \
                    overlap_tiles_count
                encoded_tile_id = ti.cast(
                    tile_u + tile_v * (camera_width // TILE_WIDTH), ti.i32)
                sort_key = ti.cast(encoded_projected_depth, ti.i64) + \
                    (ti.cast(encoded_tile_id, ti.i64) << 32)
                point_in_camera_sort_key[key_idx] = sort_key
                point_offset_with_sort_key[key_idx] = point_offset


@ti.kernel
def find_tile_start_and_end(
    point_in_camera_sort_key: ti.types.ndarray(ti.i64, ndim=1),  # (M)
    # (tiles_per_row * tiles_per_col), for output
    tile_points_start: ti.types.ndarray(ti.i32, ndim=1),  # output
    # (tiles_per_row * tiles_per_col), for output
    tile_points_end: ti.types.ndarray(ti.i32, ndim=1),  # output
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
def generate_point_attributes_in_camera_plane(  # from 3d gaussian to 2d features, including color, alpha, 2d gaussion covariance
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    # (N, 56) 56 features (cov_rotation, xxx, r, g, b)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    # (N) [0, K-1] camera_id
    point_object_id: ti.types.ndarray(ti.i32, ndim=1),
    q_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 4)
    t_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 3)
    # (M) the point id in the view frustum.
    point_id_list: ti.types.ndarray(ti.i32, ndim=1),
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2) # output
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3) # output
    point_uv_conic: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)# output
    # (M)， output，alpha after sigmoid
    point_alpha_after_activation: ti.types.ndarray(ti.f32, ndim=1),
    point_color: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)# output
    # (M)# output, estimated eigenvalues, basically the size of gaussian
    point_radii: ti.types.ndarray(ti.f32, ndim=1),
):
    for idx in range(point_id_list.shape[0]):
        camera_intrinsics_mat = ti.Matrix(
            [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
        point_id = point_id_list[idx]
        normalize_cov_rotation_in_pointcloud_features(
            pointcloud_features=pointcloud_features,
            point_id=point_id)
        gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=point_id)

        point_q_camera_pointcloud = ti.Vector(
            [q_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(4))])
        point_t_camera_pointcloud = ti.Vector(
            [t_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(3))])
        T_camera_pointcloud_mat = transform_matrix_from_quaternion_and_translation(
            q=point_q_camera_pointcloud,
            t=point_t_camera_pointcloud,
        )
        T_pointcloud_camera = taichi_inverse_SE3(T_camera_pointcloud_mat)
        ray_origin = ti.math.vec3(
            [T_pointcloud_camera[0, 3], T_pointcloud_camera[1, 3], T_pointcloud_camera[2, 3]])

        uv, xyz_in_camera = gaussian_point_3d.project_to_camera_position(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
        )
        uv_cov = gaussian_point_3d.project_to_camera_covariance(
            T_camera_world=T_camera_pointcloud_mat,
            projective_transform=camera_intrinsics_mat,
            translation_camera=xyz_in_camera,
        )
        uv_conic = get_point_conic(uv_cov)

        point_uv[idx, 0], point_uv[idx, 1] = uv[0], uv[1]
        point_in_camera[idx, 0], point_in_camera[idx, 1], point_in_camera[idx,
                                                                          2] = xyz_in_camera[0], xyz_in_camera[1], xyz_in_camera[2]
        point_uv_conic[idx, 0], point_uv_conic[idx,
                                               1], point_uv_conic[idx, 2] = uv_conic.x, uv_conic.y, uv_conic.z
        point_alpha_after_activation[idx] = 1. / \
            (1. + ti.math.exp(-gaussian_point_3d.alpha))

        ray_direction = gaussian_point_3d.translation - ray_origin

        # get color by ray actually only cares about the direction of the ray, ray origin is not used
        color = gaussian_point_3d.get_color_by_ray(
            ray_origin=ray_origin,
            ray_direction=ray_direction,
        )
        point_color[idx, 0], point_color[idx,
                                         1], point_color[idx, 2] = color[0], color[1], color[2]
        large_eigen_values = (uv_cov[0, 0] + uv_cov[1, 1] +
                              ti.sqrt((uv_cov[0, 0] - uv_cov[1, 1]) * (uv_cov[0, 0] - uv_cov[1, 1]) + 4.0 * uv_cov[0, 1] * uv_cov[1, 0])) / 2.0
        # 3.0 is a value from experiment
        radii = ti.sqrt(large_eigen_values) * 3.0
        point_radii[idx] = radii


@ti.kernel
def gaussian_point_rasterisation(
    camera_height: ti.i32,
    camera_width: ti.i32,
    # (tiles_per_row * tiles_per_col)
    tile_points_start: ti.types.ndarray(ti.i32, ndim=1),
    # (tiles_per_row * tiles_per_col)
    tile_points_end: ti.types.ndarray(ti.i32, ndim=1),
    # (K) the offset of the point in point_id_in_camera_list
    point_offset_with_sort_key: ti.types.ndarray(ti.i32, ndim=1),
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_conic: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_alpha_after_activation: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    point_color: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    rasterized_image: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 3) # output
    # (H, W) # output, Note: think about handling the occlusion
    rasterized_depth: ti.types.ndarray(ti.f32, ndim=2),
    # (H, W) # output
    pixel_accumulated_alpha: ti.types.ndarray(ti.f32, ndim=2),
    # (H, W)
    # output
    pixel_offset_of_last_effective_point: ti.types.ndarray(ti.i32, ndim=2),
    pixel_valid_point_count: ti.types.ndarray(ti.i32, ndim=2),  # output
    rgb_only: ti.template(),  # input
):
    ti.loop_config(block_dim=(TILE_WIDTH * TILE_HEIGHT))
    for pixel_offset in ti.ndrange(camera_height * camera_width):  # 1920*1080
        # initialize
        # put each TILE_WIDTH * TILE_HEIGHT tile in the same CUDA thread group (block)
        tile_id = pixel_offset // (TILE_WIDTH * TILE_HEIGHT)
        # can wait for other threads in the same group, also have a shared memory.
        thread_id = pixel_offset % (TILE_WIDTH * TILE_HEIGHT)
        tile_u = ti.cast(tile_id % (camera_width // TILE_WIDTH),
                         ti.i32)  # tile position
        tile_v = ti.cast(tile_id // (camera_width // TILE_WIDTH), ti.i32)
        # pixel position in tile (The relative position of the pixel in the tile)
        pixel_offset_in_tile = pixel_offset - \
            tile_id * (TILE_WIDTH * TILE_HEIGHT)
        pixel_u = tile_u * TILE_WIDTH + pixel_offset_in_tile % TILE_WIDTH
        pixel_v = tile_v * TILE_HEIGHT + pixel_offset_in_tile // TILE_WIDTH
        start_offset = tile_points_start[tile_id]
        end_offset = tile_points_end[tile_id]
        # The initial value of accumulated alpha (initial value of accumulated multiplication)
        T_i = 1.0
        accumulated_color = ti.math.vec3([0., 0., 0.])
        accumulated_depth = 0.
        depth_normalization_factor = 0.
        offset_of_last_effective_point = start_offset
        valid_point_count: ti.i32 = 0

        # open the shared memory
        tile_point_uv = ti.simt.block.SharedArray(
            (2, ti.static(TILE_WIDTH * TILE_HEIGHT)), dtype=ti.f32)
        tile_point_uv_conic = ti.simt.block.SharedArray(
            (3, ti.static(TILE_WIDTH * TILE_HEIGHT)), dtype=ti.f32)
        tile_point_alpha = ti.simt.block.SharedArray(
            ti.static(TILE_WIDTH * TILE_HEIGHT), dtype=ti.f32)
        tile_point_color = ti.simt.block.SharedArray(
            (3, ti.static(TILE_WIDTH * TILE_HEIGHT)), dtype=ti.f32)
        tile_point_depth = ti.simt.block.SharedArray(
            ti.static(TILE_WIDTH * TILE_HEIGHT), dtype=ti.f32)

        num_points_in_tile = end_offset - start_offset
        num_point_groups = (num_points_in_tile + ti.static(TILE_WIDTH *
                            TILE_HEIGHT - 1)) // ti.static(TILE_WIDTH * TILE_HEIGHT)
        pixel_saturated = False
        # for idx_point_offset_with_sort_key in range(start_offset, end_offset):
        for point_group_id in range(num_point_groups):
            # The original implementation uses a predicate block the next update for shared memory until all threads finish the current update
            # but it is not supported by Taichi yet, and experiments show that it does not affect the performance
            """
            tile_saturated = ti.simt.block.sync_all_nonzero(predicate=ti.cast(
                pixel_saturated, ti.i32))
            if tile_saturated != 0:
                break
            """
            ti.simt.block.sync()
            # load point data into shared memory
            # [start_offset, end_offset)->[0, end_offset - start_offset)
            to_load_idx_point_offset_with_sort_key = start_offset + \
                point_group_id * \
                ti.static(TILE_WIDTH * TILE_HEIGHT) + thread_id
            if to_load_idx_point_offset_with_sort_key < end_offset:
                to_load_point_offset = point_offset_with_sort_key[to_load_idx_point_offset_with_sort_key]
                tile_point_uv[0, thread_id] = point_uv[to_load_point_offset, 0]
                tile_point_uv[1, thread_id] = point_uv[to_load_point_offset, 1]
                tile_point_uv_conic[0,
                                    thread_id] = point_uv_conic[to_load_point_offset, 0]
                tile_point_uv_conic[1,
                                    thread_id] = point_uv_conic[to_load_point_offset, 1]
                tile_point_uv_conic[2,
                                    thread_id] = point_uv_conic[to_load_point_offset, 2]
                if not rgb_only:
                    tile_point_depth[thread_id] = point_in_camera[to_load_point_offset, 2]
                tile_point_alpha[thread_id] = point_alpha_after_activation[to_load_point_offset]

                tile_point_color[0,
                                 thread_id] = point_color[to_load_point_offset, 0]
                tile_point_color[1,
                                 thread_id] = point_color[to_load_point_offset, 1]
                tile_point_color[2,
                                 thread_id] = point_color[to_load_point_offset, 2]

            ti.simt.block.sync()
            max_point_group_offset: ti.i32 = ti.min(
                ti.static(TILE_WIDTH * TILE_HEIGHT), num_points_in_tile - point_group_id * ti.static(TILE_WIDTH * TILE_HEIGHT))
            for point_group_offset in range(max_point_group_offset):
                if pixel_saturated:
                    break
                # forward rendering process
                idx_point_offset_with_sort_key: ti.i32 = start_offset + \
                    point_group_id * \
                    ti.static(TILE_WIDTH * TILE_HEIGHT) + point_group_offset

                uv = ti.math.vec2(
                    [tile_point_uv[0, point_group_offset], tile_point_uv[1, point_group_offset]])
                uv_conic = ti.math.vec3([tile_point_uv_conic[0, point_group_offset], tile_point_uv_conic[1, point_group_offset],
                                         tile_point_uv_conic[2, point_group_offset]])
                point_alpha_after_activation_value = tile_point_alpha[point_group_offset]
                color = ti.math.vec3([tile_point_color[0, point_group_offset],
                                     tile_point_color[1, point_group_offset], tile_point_color[2, point_group_offset]])

                gaussian_alpha = get_point_probability_density_from_conic(
                    xy=ti.math.vec2([pixel_u + 0.5, pixel_v + 0.5]),
                    gaussian_mean=uv,
                    conic=uv_conic,
                )
                alpha = gaussian_alpha * point_alpha_after_activation_value
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
                next_T_i = T_i * (1 - alpha)
                if next_T_i < 0.0001:
                    pixel_saturated = True
                    continue  # somehow faster than directly breaking
                offset_of_last_effective_point = idx_point_offset_with_sort_key + 1
                accumulated_color += color * alpha * T_i

                if not rgb_only:
                    # Weighted depth for all valid points.
                    depth = tile_point_depth[point_group_offset]
                    accumulated_depth += depth * alpha * T_i
                    depth_normalization_factor += alpha * T_i
                    valid_point_count += 1
                T_i = next_T_i
            # end of point group loop

        # end of point group id loop

        rasterized_image[pixel_v, pixel_u, 0] = accumulated_color[0]
        rasterized_image[pixel_v, pixel_u, 1] = accumulated_color[1]
        rasterized_image[pixel_v, pixel_u, 2] = accumulated_color[2]
        if not rgb_only:
            rasterized_depth[pixel_v, pixel_u] = accumulated_depth / \
                ti.max(depth_normalization_factor, 1e-6)
            pixel_accumulated_alpha[pixel_v, pixel_u] = 1. - T_i
            pixel_offset_of_last_effective_point[pixel_v,
                                                 pixel_u] = offset_of_last_effective_point
            pixel_valid_point_count[pixel_v, pixel_u] = valid_point_count
    # end of pixel loop


@ti.kernel
def gaussian_point_rasterisation_backward(
    camera_height: ti.i32,
    camera_width: ti.i32,
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, K)
    point_object_id: ti.types.ndarray(ti.i32, ndim=1),  # (N)
    q_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 4)
    t_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 3)
    t_pointcloud_camera: ti.types.ndarray(ti.f32, ndim=2),  # (K, 3)
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

    in_camera_grad_uv_cov_buffer: ti.types.ndarray(ti.f32, ndim=2),
    in_camera_grad_color_buffer: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)

    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_conic: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_alpha_after_activation: ti.types.ndarray(ti.f32, ndim=1),  # (M)
    point_color: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)

    need_extra_info: ti.template(),
    magnitude_grad_viewspace: ti.types.ndarray(ti.f32, ndim=1),  # (N)
    # (H, W, 2)
    magnitude_grad_viewspace_on_image: ti.types.ndarray(ti.f32, ndim=3),
    # (M, 2, 2)
    in_camera_num_affected_pixels: ti.types.ndarray(ti.i32, ndim=1),  # (M)
):
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])

    ti.loop_config(block_dim=(TILE_HEIGHT * TILE_WIDTH))
    for pixel_offset in ti.ndrange(camera_height * camera_width):
        # each block handles one tile, so tile_id is actually block_id
        tile_id = pixel_offset // (TILE_HEIGHT * TILE_WIDTH)
        thread_id = pixel_offset % (TILE_HEIGHT * TILE_WIDTH)
        tile_u = ti.cast(tile_id % (camera_width // TILE_WIDTH), ti.i32)
        tile_v = ti.cast(tile_id // (camera_width // TILE_WIDTH), ti.i32)

        start_offset = tile_points_start[tile_id]
        end_offset = tile_points_end[tile_id]
        tile_point_count = end_offset - start_offset

        tile_point_uv = ti.simt.block.SharedArray(
            (2, ti.static(TILE_HEIGHT * TILE_WIDTH)), dtype=ti.f32)  # 2KB shared memory
        tile_point_uv_conic = ti.simt.block.SharedArray(
            (3, ti.static(TILE_HEIGHT * TILE_WIDTH)), dtype=ti.f32)  # 4KB shared memory
        tile_point_color = ti.simt.block.SharedArray(
            (3, ti.static(TILE_HEIGHT * TILE_WIDTH)), dtype=ti.f32)  # 3KB shared memory
        tile_point_alpha = ti.simt.block.SharedArray(
            (ti.static(TILE_HEIGHT * TILE_WIDTH),), dtype=ti.f32)  # 1KB shared memory

        pixel_offset_in_tile = pixel_offset - \
            tile_id * ti.static(TILE_HEIGHT * TILE_WIDTH)
        pixel_offset_u_in_tile = pixel_offset_in_tile % TILE_WIDTH
        pixel_offset_v_in_tile = pixel_offset_in_tile // TILE_WIDTH
        pixel_u = tile_u * TILE_WIDTH + pixel_offset_u_in_tile
        pixel_v = tile_v * TILE_HEIGHT + pixel_offset_v_in_tile
        last_effective_point = pixel_offset_of_last_effective_point[pixel_v, pixel_u]
        accumulated_alpha: ti.f32 = pixel_accumulated_alpha[pixel_v, pixel_u]
        T_i = 1.0 - accumulated_alpha  # T_i = \prod_{j=1}^{i-1} (1 - a_j)
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
        # for inverse_point_offset_base in range(0, tile_point_count, TILE_HEIGHT * TILE_WIDTH):
        num_point_blocks = (tile_point_count + TILE_HEIGHT *
                            TILE_WIDTH - 1) // (TILE_HEIGHT * TILE_WIDTH)
        for point_block_id in range(num_point_blocks):
            inverse_point_offset_base = point_block_id * \
                (TILE_HEIGHT * TILE_WIDTH)
            block_end_idx_point_offset_with_sort_key = end_offset - inverse_point_offset_base
            block_start_idx_point_offset_with_sort_key = ti.max(
                block_end_idx_point_offset_with_sort_key - (TILE_HEIGHT * TILE_WIDTH), 0)
            # in the later loop, we will handle the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key)
            # so we need to load the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key - 1]
            to_load_idx_point_offset_with_sort_key = block_end_idx_point_offset_with_sort_key - thread_id - 1
            if to_load_idx_point_offset_with_sort_key >= block_start_idx_point_offset_with_sort_key:
                to_load_point_offset = point_offset_with_sort_key[to_load_idx_point_offset_with_sort_key]
                to_load_uv = ti.math.vec2(
                    [point_uv[to_load_point_offset, 0], point_uv[to_load_point_offset, 1]])

                for i in ti.static(range(2)):
                    tile_point_uv[i, thread_id] = to_load_uv[i]

                for i in ti.static(range(3)):
                    tile_point_uv_conic[i,
                                        thread_id] = point_uv_conic[to_load_point_offset, i]
                for i in ti.static(range(3)):
                    tile_point_color[i,
                                     thread_id] = point_color[to_load_point_offset, i]

                tile_point_alpha[thread_id] = point_alpha_after_activation[to_load_point_offset]

            ti.simt.block.sync()
            max_inverse_point_offset_offset = ti.min(
                (TILE_HEIGHT * TILE_WIDTH), tile_point_count - inverse_point_offset_base)
            for inverse_point_offset_offset in range(max_inverse_point_offset_offset):
                inverse_point_offset = inverse_point_offset_base + inverse_point_offset_offset

                idx_point_offset_with_sort_key = end_offset - inverse_point_offset - 1
                if idx_point_offset_with_sort_key >= last_effective_point:
                    continue

                idx_point_offset_with_sort_key_in_block = inverse_point_offset_offset
                uv = ti.math.vec2(tile_point_uv[0, idx_point_offset_with_sort_key_in_block],
                                  tile_point_uv[1, idx_point_offset_with_sort_key_in_block])
                uv_conic = ti.math.vec3([
                    tile_point_uv_conic[0,
                                        idx_point_offset_with_sort_key_in_block],
                    tile_point_uv_conic[1,
                                        idx_point_offset_with_sort_key_in_block],
                    tile_point_uv_conic[2,
                                        idx_point_offset_with_sort_key_in_block],
                ])

                point_alpha_after_activation_value = tile_point_alpha[
                    idx_point_offset_with_sort_key_in_block]

                # d_p_d_mean is (2,), d_p_d_cov is (2, 2), needs to be flattened to (4,)
                gaussian_alpha, d_p_d_mean, d_p_d_cov = grad_point_probability_density_from_conic(
                    xy=ti.math.vec2([pixel_u + 0.5, pixel_v + 0.5]),
                    gaussian_mean=uv,
                    conic=uv_conic,
                )
                prod_alpha = gaussian_alpha * point_alpha_after_activation_value
                # from paper: we skip any blending updates with 𝛼 < 𝜖 (we choose 𝜖 as 1
                # 255 ) and also clamp 𝛼 with 0.99 from above.
                if prod_alpha >= 1. / 255.:
                    alpha: ti.f32 = ti.min(prod_alpha, 0.99)
                    color = ti.math.vec3([
                        tile_point_color[0,
                                         idx_point_offset_with_sort_key_in_block],
                        tile_point_color[1,
                                         idx_point_offset_with_sort_key_in_block],
                        tile_point_color[2, idx_point_offset_with_sort_key_in_block]])

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
                    alpha_grad: ti.f32 = alpha_grad_from_rgb.sum()
                    point_alpha_after_activation_grad = alpha_grad * gaussian_alpha
                    gaussian_point_3d_alpha_grad = point_alpha_after_activation_grad * \
                        (1. - point_alpha_after_activation_value) * \
                        point_alpha_after_activation_value
                    gaussian_alpha_grad = alpha_grad * point_alpha_after_activation_value
                    # gaussian_alpha_grad is dp
                    point_viewspace_grad = gaussian_alpha_grad * \
                        d_p_d_mean  # (2,) as the paper said, view space gradient is used for detect candidates for densification
                    total_magnitude_grad_viewspace_on_image += ti.abs(
                        point_viewspace_grad)
                    point_uv_cov_grad = gaussian_alpha_grad * \
                        d_p_d_cov  # (2, 2)

                    point_offset = point_offset_with_sort_key[idx_point_offset_with_sort_key]
                    point_id = point_id_in_camera_list[point_offset]
                    # atomic accumulate on block shared memory shall be faster
                    for i in ti.static(range(2)):
                        ti.atomic_add(
                            grad_uv[point_id, i], point_viewspace_grad[i])
                    ti.atomic_add(in_camera_grad_uv_cov_buffer[point_offset, 0],
                                  point_uv_cov_grad[0, 0])
                    ti.atomic_add(in_camera_grad_uv_cov_buffer[point_offset, 1],
                                  point_uv_cov_grad[0, 1])
                    ti.atomic_add(in_camera_grad_uv_cov_buffer[point_offset, 2],
                                  point_uv_cov_grad[1, 1])

                    for i in ti.static(range(3)):
                        ti.atomic_add(
                            in_camera_grad_color_buffer[point_offset, i], point_grad_color[i])
                    ti.atomic_add(
                        grad_pointcloud_features[point_id, 7], gaussian_point_3d_alpha_grad)

                    if need_extra_info:
                        magnitude_point_grad_viewspace = ti.sqrt(
                            point_viewspace_grad[0] ** 2 + point_viewspace_grad[1] ** 2)
                        ti.atomic_add(
                            magnitude_grad_viewspace[point_id], magnitude_point_grad_viewspace)
                        ti.atomic_add(
                            in_camera_num_affected_pixels[point_offset], 1)
            # end of the TILE_WIDTH * TILE_HEIGHT block loop
            ti.simt.block.sync()
        # end of the backward traversal loop, from last point to first point
        if need_extra_info:
            magnitude_grad_viewspace_on_image[pixel_v, pixel_u,
                                              0] = total_magnitude_grad_viewspace_on_image[0]
            magnitude_grad_viewspace_on_image[pixel_v, pixel_u,
                                              1] = total_magnitude_grad_viewspace_on_image[1]
    # end of per pixel loop

    # one more loop to compute the gradient from viewspace to 3D point
    for idx in range(point_id_in_camera_list.shape[0]):
        point_id = point_id_in_camera_list[idx]
        gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
            pointcloud=pointcloud,
            pointcloud_features=pointcloud_features,
            point_id=point_id)
        point_grad_uv = ti.math.vec2(
            grad_uv[point_id, 0], grad_uv[point_id, 1])
        point_grad_uv_cov_flat = ti.math.vec4(
            in_camera_grad_uv_cov_buffer[idx, 0],
            in_camera_grad_uv_cov_buffer[idx, 1],
            in_camera_grad_uv_cov_buffer[idx, 1],
            in_camera_grad_uv_cov_buffer[idx, 2],
        )
        point_grad_color = ti.math.vec3(
            in_camera_grad_color_buffer[idx, 0],
            in_camera_grad_color_buffer[idx, 1],
            in_camera_grad_color_buffer[idx, 2],
        )
        point_q_camera_pointcloud = ti.Vector(
            [q_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(4))])
        point_t_camera_pointcloud = ti.Vector(
            [t_camera_pointcloud[point_object_id[point_id], idx] for idx in ti.static(range(3))])
        ray_origin = ti.Vector(
            [t_pointcloud_camera[point_object_id[point_id], idx] for idx in ti.static(range(3))])
        T_camera_pointcloud_mat = transform_matrix_from_quaternion_and_translation(
            q=point_q_camera_pointcloud,
            t=point_t_camera_pointcloud,
        )
        translation_camera = ti.Vector([
            point_in_camera[idx, j] for j in ti.static(range(3))])
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
        rgb_only: bool = False
        grad_color_factor = 5.
        grad_high_order_color_factor = 1.
        grad_s_factor = 0.5
        grad_q_factor = 1.
        grad_alpha_factor = 20.

    @dataclass
    class GaussianPointCloudRasterisationInput:
        point_cloud: torch.Tensor  # Nx3
        point_cloud_features: torch.Tensor  # NxM
        # (N,), we allow points belong to different objects,
        # different objects may have different camera poses.
        # By moving camera, we can actually handle moving rigid objects.
        # if no moving objects, then everything belongs to the same object with id 0.
        # it shall works better once we also optimize for camera pose.
        point_object_id: torch.Tensor
        point_invalid_mask: torch.Tensor  # N
        camera_info: CameraInfo
        # Kx4, x to the right, y down, z forward, K is the number of objects
        q_pointcloud_camera: torch.Tensor
        # Kx3, x to the right, y down, z forward, K is the number of objects
        t_pointcloud_camera: torch.Tensor
        color_max_sh_band: int = 2

    @dataclass
    class BackwardValidPointHookInput:
        point_id_in_camera_list: torch.Tensor  # M
        grad_point_in_camera: torch.Tensor  # Mx3
        grad_pointfeatures_in_camera: torch.Tensor  # Mx56
        grad_viewspace: torch.Tensor  # Mx2
        magnitude_grad_viewspace: torch.Tensor  # M
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
            def forward(ctx,
                        pointcloud,
                        pointcloud_features,
                        point_invalid_mask,
                        point_object_id,
                        q_pointcloud_camera,
                        t_pointcloud_camera,
                        camera_info,
                        color_max_sh_band,
                        ):
                point_in_camera_mask = torch.zeros(
                    size=(pointcloud.shape[0],), dtype=torch.int8, device=pointcloud.device)
                point_id = torch.arange(
                    pointcloud.shape[0], dtype=torch.int32, device=pointcloud.device)
                q_camera_pointcloud, t_camera_pointcloud = inverse_SE3_qt_torch(
                    q=q_pointcloud_camera, t=t_pointcloud_camera)
                # Step 1: filter points
                filter_point_in_camera(
                    pointcloud=pointcloud,
                    point_invalid_mask=point_invalid_mask,
                    point_object_id=point_object_id,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    q_camera_pointcloud=q_camera_pointcloud,
                    t_camera_pointcloud=t_camera_pointcloud,
                    point_in_camera_mask=point_in_camera_mask,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    camera_height=camera_info.camera_height,
                    camera_width=camera_info.camera_width,
                )
                point_in_camera_mask = point_in_camera_mask.bool()

                # Get id based on the camera_mask
                point_id_in_camera_list = point_id[point_in_camera_mask].contiguous(
                )
                del point_id
                del point_in_camera_mask

                # Number of points in camera
                num_points_in_camera = point_id_in_camera_list.shape[0]

                # Allocate memory
                point_uv = torch.empty(
                    size=(num_points_in_camera, 2), dtype=torch.float32, device=pointcloud.device)
                point_alpha_after_activation = torch.empty(
                    size=(num_points_in_camera,), dtype=torch.float32, device=pointcloud.device)
                point_in_camera = torch.empty(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_uv_conic = torch.empty(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_color = torch.zeros(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_radii = torch.empty(
                    size=(num_points_in_camera,), dtype=torch.float32, device=pointcloud.device)

                # Step 2: get 2d features
                generate_point_attributes_in_camera_plane(
                    pointcloud=pointcloud,
                    pointcloud_features=pointcloud_features,
                    point_object_id=point_object_id,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    point_id_list=point_id_in_camera_list,
                    q_camera_pointcloud=q_camera_pointcloud,
                    t_camera_pointcloud=t_camera_pointcloud,
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_conic=point_uv_conic,
                    point_alpha_after_activation=point_alpha_after_activation,
                    point_color=point_color,
                    point_radii=point_radii,
                )

                # Step 3: get how many tiles overlapped, in order to allocate memory
                num_overlap_tiles = torch.empty_like(point_id_in_camera_list)
                generate_num_overlap_tiles(
                    num_overlap_tiles=num_overlap_tiles,
                    point_uv=point_uv,
                    point_radii=point_radii,
                    camera_width=camera_info.camera_width,
                    camera_height=camera_info.camera_height,
                )
                # Calculate pre-sum of number_overlap_tiles
                accumulated_num_overlap_tiles = torch.cumsum(
                    num_overlap_tiles, dim=0)
                if len(accumulated_num_overlap_tiles) > 0:
                    total_num_overlap_tiles = accumulated_num_overlap_tiles[-1]
                else:
                    total_num_overlap_tiles = 0
                # The space of each point.
                accumulated_num_overlap_tiles = torch.cat(
                    (torch.zeros(size=(1,), dtype=torch.int32, device=pointcloud.device),
                     accumulated_num_overlap_tiles[:-1]))

                # del num_overlap_tiles

                # 64-bits key
                point_in_camera_sort_key = torch.empty(
                    size=(total_num_overlap_tiles,), dtype=torch.int64, device=pointcloud.device)
                # Corresponding to the original position, the record is the point offset in the frustum (engineering optimization)
                point_offset_with_sort_key = torch.empty(
                    size=(total_num_overlap_tiles,), dtype=torch.int32, device=pointcloud.device)

                # Step 4: calclualte key
                if point_in_camera_sort_key.shape[0] > 0:
                    generate_point_sort_key_by_num_overlap_tiles(
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_radii=point_radii,
                        accumulated_num_overlap_tiles=accumulated_num_overlap_tiles,  # input
                        point_offset_with_sort_key=point_offset_with_sort_key,  # output
                        point_in_camera_sort_key=point_in_camera_sort_key,  # output
                        camera_width=camera_info.camera_width,
                        camera_height=camera_info.camera_height,
                        depth_to_sort_key_scale=self.config.depth_to_sort_key_scale,
                    )

                point_in_camera_sort_key, permutation = point_in_camera_sort_key.sort()
                point_offset_with_sort_key = point_offset_with_sort_key[permutation].contiguous(
                )  # now the point_offset_with_sort_key is sorted by the sort_key
                del permutation

                tiles_per_row = camera_info.camera_width // TILE_WIDTH
                tiles_per_col = camera_info.camera_height // TILE_HEIGHT
                tile_points_start = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                tile_points_end = torch.zeros(size=(
                    tiles_per_row * tiles_per_col,), dtype=torch.int32, device=pointcloud.device)
                # Find tile's start and end.
                if point_in_camera_sort_key.shape[0] > 0:
                    find_tile_start_and_end(
                        point_in_camera_sort_key=point_in_camera_sort_key,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                    )

                # Allocate space for the image.
                rasterized_image = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, 3, dtype=torch.float32, device=pointcloud.device)
                rasterized_depth = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_accumulated_alpha = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_offset_of_last_effective_point = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                pixel_valid_point_count = torch.empty(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.int32, device=pointcloud.device)
                # print(f"num_points: {pointcloud.shape[0]}, num_points_in_camera: {num_points_in_camera}, num_points_rendered: {point_in_camera_sort_key.shape[0]}")

                # Step 5: render
                if point_in_camera_sort_key.shape[0] > 0:
                    gaussian_point_rasterisation(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                        point_offset_with_sort_key=point_offset_with_sort_key,
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_uv_conic=point_uv_conic,
                        point_alpha_after_activation=point_alpha_after_activation,
                        point_color=point_color,
                        rasterized_image=rasterized_image,
                        rgb_only=self.config.rgb_only,
                        rasterized_depth=rasterized_depth,
                        pixel_accumulated_alpha=pixel_accumulated_alpha,
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                        pixel_valid_point_count=pixel_valid_point_count)
                ctx.save_for_backward(
                    pointcloud,
                    pointcloud_features,
                    # point_id_with_sort_key is sorted by tile and depth and has duplicated points, e.g. one points is belong to multiple tiles
                    point_offset_with_sort_key,
                    point_id_in_camera_list,  # point_in_camera_id does not have duplicated points
                    tile_points_start,
                    tile_points_end,
                    pixel_accumulated_alpha,
                    pixel_offset_of_last_effective_point,
                    num_overlap_tiles,
                    point_object_id,
                    q_pointcloud_camera,
                    q_camera_pointcloud,
                    t_pointcloud_camera,
                    t_camera_pointcloud,
                    point_uv,
                    point_in_camera,
                    point_uv_conic,
                    point_alpha_after_activation,
                    point_color,
                )
                ctx.camera_info = camera_info
                ctx.color_max_sh_band = color_max_sh_band
                # rasterized_image.requires_grad_(True)
                return rasterized_image, rasterized_depth, pixel_valid_point_count

            @staticmethod
            def backward(ctx, grad_rasterized_image, grad_rasterized_depth, grad_pixel_valid_point_count):
                grad_pointcloud = grad_pointcloud_features = grad_q_pointcloud_camera = grad_t_pointcloud_camera = None
                if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                    pointcloud, \
                        pointcloud_features, \
                        point_offset_with_sort_key, \
                        point_id_in_camera_list, \
                        tile_points_start, \
                        tile_points_end, \
                        pixel_accumulated_alpha, \
                        pixel_offset_of_last_effective_point, \
                        num_overlap_tiles, \
                        point_object_id, \
                        q_pointcloud_camera, \
                        q_camera_pointcloud, \
                        t_pointcloud_camera, \
                        t_camera_pointcloud, \
                        point_uv, \
                        point_in_camera, \
                        point_uv_conic, \
                        point_alpha_after_activation, \
                        point_color = ctx.saved_tensors
                    camera_info = ctx.camera_info
                    color_max_sh_band = ctx.color_max_sh_band
                    grad_rasterized_image = grad_rasterized_image.contiguous()
                    grad_pointcloud = torch.zeros_like(pointcloud)
                    grad_pointcloud_features = torch.zeros_like(
                        pointcloud_features)

                    grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], 2), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace = torch.zeros(
                        size=(pointcloud.shape[0], ), dtype=torch.float32, device=pointcloud.device)
                    magnitude_grad_viewspace_on_image = torch.empty_like(
                        grad_rasterized_image[:, :, :2])

                    in_camera_grad_uv_cov_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    in_camera_grad_color_buffer = torch.zeros(
                        size=(point_id_in_camera_list.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    in_camera_num_affected_pixels = torch.zeros(
                        size=(point_id_in_camera_list.shape[0],), dtype=torch.int32, device=pointcloud.device)

                    gaussian_point_rasterisation_backward(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        camera_intrinsics=camera_info.camera_intrinsics.contiguous(),
                        point_object_id=point_object_id.contiguous(),
                        q_camera_pointcloud=q_camera_pointcloud.contiguous(),
                        t_camera_pointcloud=t_camera_pointcloud.contiguous(),
                        t_pointcloud_camera=t_pointcloud_camera.contiguous(),
                        pointcloud=pointcloud.contiguous(),
                        pointcloud_features=pointcloud_features.contiguous(),
                        tile_points_start=tile_points_start.contiguous(),
                        tile_points_end=tile_points_end.contiguous(),
                        point_offset_with_sort_key=point_offset_with_sort_key.contiguous(),
                        point_id_in_camera_list=point_id_in_camera_list.contiguous(),
                        rasterized_image_grad=grad_rasterized_image.contiguous(),
                        pixel_accumulated_alpha=pixel_accumulated_alpha.contiguous(),
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point.contiguous(),
                        grad_pointcloud=grad_pointcloud.contiguous(),
                        grad_pointcloud_features=grad_pointcloud_features.contiguous(),
                        grad_uv=grad_viewspace.contiguous(),
                        in_camera_grad_uv_cov_buffer=in_camera_grad_uv_cov_buffer.contiguous(),
                        in_camera_grad_color_buffer=in_camera_grad_color_buffer.contiguous(),
                        point_uv=point_uv.contiguous(),
                        point_in_camera=point_in_camera.contiguous(),
                        point_uv_conic=point_uv_conic.contiguous(),
                        point_alpha_after_activation=point_alpha_after_activation.contiguous(),
                        point_color=point_color.contiguous(),
                        need_extra_info=True,
                        magnitude_grad_viewspace=magnitude_grad_viewspace.contiguous(),
                        magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image.contiguous(),
                        in_camera_num_affected_pixels=in_camera_num_affected_pixels.contiguous(),
                    )
                    del tile_points_start, tile_points_end, pixel_accumulated_alpha, pixel_offset_of_last_effective_point
                    grad_pointcloud_features = self._clear_grad_by_color_max_sh_band(
                        grad_pointcloud_features=grad_pointcloud_features,
                        color_max_sh_band=color_max_sh_band)
                    grad_pointcloud_features[:,
                                             :4] *= self.config.grad_q_factor
                    grad_pointcloud_features[:,
                                             4:7] *= self.config.grad_s_factor
                    grad_pointcloud_features[:,
                                             7] *= self.config.grad_alpha_factor

                    # 8, 24, 40 are the zero order coefficients of the SH basis
                    grad_pointcloud_features[:,
                                             8] *= self.config.grad_color_factor
                    grad_pointcloud_features[:,
                                             24] *= self.config.grad_color_factor
                    grad_pointcloud_features[:,
                                             40] *= self.config.grad_color_factor
                    # other coefficients are the higher order coefficients of the SH basis
                    grad_pointcloud_features[:,
                                             9:24] *= self.config.grad_high_order_color_factor
                    grad_pointcloud_features[:,
                                             25:40] *= self.config.grad_high_order_color_factor
                    grad_pointcloud_features[:,
                                             41:] *= self.config.grad_high_order_color_factor

                    if backward_valid_point_hook is not None:
                        backward_valid_point_hook_input = GaussianPointCloudRasterisation.BackwardValidPointHookInput(
                            point_id_in_camera_list=point_id_in_camera_list,
                            grad_point_in_camera=grad_pointcloud[point_id_in_camera_list],
                            grad_pointfeatures_in_camera=grad_pointcloud_features[
                                point_id_in_camera_list],
                            grad_viewspace=grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace=magnitude_grad_viewspace[point_id_in_camera_list],
                            magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image,
                            num_overlap_tiles=num_overlap_tiles,
                            num_affected_pixels=in_camera_num_affected_pixels,
                            point_uv_in_camera=point_uv,
                            point_depth=point_in_camera[:, 2],
                        )
                        backward_valid_point_hook(
                            backward_valid_point_hook_input)
                """_summary_
                pointcloud,
                        pointcloud_features,
                        point_invalid_mask,
                        point_object_id,
                        q_pointcloud_camera,
                        t_pointcloud_camera,
                        camera_info,
                        color_max_sh_band,

                Returns:
                    _type_: _description_
                """

                return grad_pointcloud, \
                    grad_pointcloud_features, \
                    None, \
                    None, \
                    grad_q_pointcloud_camera, \
                    grad_t_pointcloud_camera, \
                    None, None

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
        point_object_id = input_data.point_object_id
        q_pointcloud_camera = input_data.q_pointcloud_camera
        t_pointcloud_camera = input_data.t_pointcloud_camera
        color_max_sh_band = input_data.color_max_sh_band
        camera_info = input_data.camera_info
        assert camera_info.camera_width % TILE_WIDTH == 0
        assert camera_info.camera_height % TILE_HEIGHT == 0
        return self._module_function.apply(
            pointcloud,
            pointcloud_features,
            point_invalid_mask,
            point_object_id,
            q_pointcloud_camera,
            t_pointcloud_camera,
            camera_info,
            color_max_sh_band,
        )
