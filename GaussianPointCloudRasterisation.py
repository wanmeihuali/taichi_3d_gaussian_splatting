import numpy as np
import torch
import taichi as ti
from dataclasses import dataclass
from Camera import CameraInfo, CameraView
from torch.cuda.amp import custom_bwd, custom_fwd
from utils import (torch_type, data_type, ti2torch, torch2ti,
                   ti2torch_grad, torch2ti_grad,
                   get_ray_origin_and_direction_by_uv,
                   get_point_probability_density_from_2d_gaussian,
                   grad_point_probability_density_2d,
                   inverse_se3)
from GaussianPoint3D import GaussianPoint3D, project_point_to_camera
from SphericalHarmonics import SphericalHarmonics, vec16f
from typing import List, Tuple, Optional, Callable, Union


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
    point_in_camera_id: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_covariance: ti.types.ndarray(ti.f32, ndim=3),  # (M, 2, 2)
):
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in ti.static(range(4))] for row in ti.static(range(4))])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
    for idx in range(point_in_camera_id.shape[0]):
        point_id = point_in_camera_id[idx]
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
    point_in_camera_id: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    point_uv: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
    point_in_camera: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    point_uv_covariance: ti.types.ndarray(ti.f32, ndim=3),  # (M, 2, 2)
    rasterized_image: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 3)
    pixel_accumulated_alpha: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    # (H, W)
    pixel_offset_of_last_effective_point: ti.types.ndarray(ti.i32, ndim=2),
):
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in ti.static(range(4))] for row in ti.static(range(4))])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
    # taichi does not support thread block, so we just have a single thread for each pixel
    # hope the missing for shared block memory will not hurt the performance too much
    for pixel_v, pixel_u in ti.ndrange(camera_height, camera_width):
        tile_u = ti.cast(pixel_u // 16, ti.i32)
        tile_v = ti.cast(pixel_v // 16, ti.i32)
        tile_id = tile_u + tile_v * (camera_width // 16)
        start_offset = tile_points_start[tile_id]
        end_offset = tile_points_end[tile_id]
        accumulated_alpha = 0.
        accumulated_color = ti.math.vec3([0., 0., 0.])
        offset_of_last_effective_point = start_offset
        ray_origin, ray_direction = get_ray_origin_and_direction_by_uv(
            pixel_u=pixel_u,
            pixel_v=pixel_v,
            camera_intrinsics=camera_intrinsics_mat,
            T_camera_pointcloud=T_camera_pointcloud_mat,
        )
        for point_offset in range(start_offset, end_offset):
            point_id = point_in_camera_id[point_offset]
            uv = ti.math.vec2([point_uv[point_offset, 0],
                              point_uv[point_offset, 1]])
            xyz_in_camera = ti.math.vec3(
                [point_in_camera[point_offset, 0], point_in_camera[point_offset, 1], point_in_camera[point_offset, 2]])
            uv_cov = ti.math.mat2([point_uv_covariance[point_offset, 0, 0], point_uv_covariance[point_offset, 0, 1],
                                  point_uv_covariance[point_offset, 1, 0], point_uv_covariance[point_offset, 1, 1]])
            gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
                pointcloud=pointcloud,
                pointcloud_features=pointcloud_features,
                point_id=point_id)

            gaussian_alpha = get_point_probability_density_from_2d_gaussian(
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
            if accumulated_alpha + alpha > 0.9999:
                break
            offset_of_last_effective_point = point_offset + 1
            color = gaussian_point_3d.get_color_by_ray(
                ray_origin=ray_origin,
                ray_direction=ray_direction,
            )
            accumulated_color = accumulated_color + \
                alpha * (1. - accumulated_alpha) * color
            accumulated_alpha += alpha
        rasterized_image[pixel_v, pixel_u, 0] = accumulated_color[0]
        rasterized_image[pixel_v, pixel_u, 1] = accumulated_color[1]
        rasterized_image[pixel_v, pixel_u, 2] = accumulated_color[2]
        pixel_accumulated_alpha[pixel_v, pixel_u] = accumulated_alpha
        pixel_offset_of_last_effective_point[pixel_v,
                                             pixel_u] = offset_of_last_effective_point


@ti.func
def atomic_accumulate_grad_for_point(
    point_offset: ti.i32,
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
    ti.atomic_add(viewspace_grad[point_offset, 0], point_viewspace_grad[0])
    ti.atomic_add(viewspace_grad[point_offset, 1], point_viewspace_grad[1])
    for offset in ti.static(range(3)):
        ti.atomic_add(point_in_camera_grad[point_offset, offset],
                      translation_grad[offset])
    for offset in ti.static(range(4)):
        ti.atomic_add(pointfeatures_grad[point_offset, offset],
                      gaussian_q_grad[offset])
    for offset in ti.static(range(3)):
        ti.atomic_add(pointfeatures_grad[point_offset, 4 + offset],
                      gaussian_s_grad[offset])
    ti.atomic_add(pointfeatures_grad[point_offset, 7],
                  gaussian_point_3d_alpha_grad)

    for offset in ti.static(range(16)):
        ti.atomic_add(pointfeatures_grad[point_offset, 8 + offset],
                      color_r_grad[offset])
        ti.atomic_add(pointfeatures_grad[point_offset, 24 + offset],
                      color_g_grad[offset])
        ti.atomic_add(pointfeatures_grad[point_offset, 40 + offset],
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
    point_in_camera_id: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    rasterized_image_grad: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 3)
    pixel_accumulated_alpha: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    # (H, W)
    pixel_offset_of_last_effective_point: ti.types.ndarray(ti.i32, ndim=2),
    point_in_camera_grad: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    pointfeatures_grad: ti.types.ndarray(ti.f32, ndim=2),  # (M, K)
    viewspace_grad: ti.types.ndarray(ti.f32, ndim=2),  # (M, 2)
):
    T_camera_pointcloud_mat = ti.Matrix(
        [[T_camera_pointcloud[row, col] for col in ti.static(range(4))] for row in ti.static(range(4))])
    camera_intrinsics_mat = ti.Matrix(
        [[camera_intrinsics[row, col] for col in ti.static(range(3))] for row in ti.static(range(3))])
    # taichi does not support thread block, so we just have a single thread for each pixel
    # hope the missing for shared block memory will not hurt the performance too much
    for pixel_v, pixel_u in ti.ndrange(camera_height, camera_width):
        tile_u = ti.cast(pixel_u // 16, ti.i32)
        tile_v = ti.cast(pixel_v // 16, ti.i32)
        tile_id = tile_u + tile_v * (camera_width // 16)
        start_offset = tile_points_start[tile_id]
        last_effective_point = pixel_offset_of_last_effective_point[pixel_v, pixel_u]
        effective_point_count = last_effective_point - start_offset
        accumulated_alpha: ti.f32 = pixel_accumulated_alpha[pixel_v, pixel_u]
        pixel_rgb_grad = ti.math.vec3(
            rasterized_image_grad[pixel_v, pixel_u, 0], rasterized_image_grad[pixel_v, pixel_u, 1], rasterized_image_grad[pixel_v, pixel_u, 2])
        ray_origin, ray_direction = get_ray_origin_and_direction_by_uv(
            pixel_u=pixel_u,
            pixel_v=pixel_v,
            camera_intrinsics=camera_intrinsics_mat,
            T_camera_pointcloud=T_camera_pointcloud_mat,
        )
        for inverse_point_offset in range(effective_point_count):
            point_offset = last_effective_point - inverse_point_offset - 1
            point_id = point_in_camera_id[point_offset]
            gaussian_point_3d = load_point_cloud_row_into_gaussian_point_3d(
                pointcloud=pointcloud,
                pointcloud_features=pointcloud_features,
                point_id=point_id)
            uv, translation_camera = gaussian_point_3d.project_to_camera_position(
                T_camera_world=T_camera_pointcloud_mat,
                projective_transform=camera_intrinsics_mat,
            )
            d_uv_d_translation = gaussian_point_3d.project_to_camera_position_jacobian(
                T_camera_world=T_camera_pointcloud_mat,
                projective_transform=camera_intrinsics_mat,
            )  # (2, 3)
            uv_cov = gaussian_point_3d.project_to_camera_covariance(
                T_camera_world=T_camera_pointcloud_mat,
                projective_transform=camera_intrinsics_mat,
                translation_camera=translation_camera)

            # d_Sigma_prime_d_q is 4x4, d_Sigma_prime_d_s is 4x3
            d_Sigma_prime_d_q, d_Sigma_prime_d_s = gaussian_point_3d.project_to_camera_covariance_jacobian(
                T_camera_world=T_camera_pointcloud_mat,
                projective_transform=camera_intrinsics_mat,
                translation_camera=translation_camera,
            )

            # d_p_d_mean is (2,), d_p_d_cov is (2, 2), needs to be flattened to (4,)
            gaussian_alpha, d_p_d_mean, d_p_d_cov = grad_point_probability_density_2d(
                xy=ti.math.vec2([pixel_u + 0.5, pixel_v + 0.5]),
                gaussian_mean=uv,
                gaussian_covariance=uv_cov,
            )
            d_p_d_cov_flat = ti.math.vec4(
                [d_p_d_cov[0, 0], d_p_d_cov[0, 1], d_p_d_cov[1, 0], d_p_d_cov[1, 1]])
            point_alpha_after_activation = 1. / \
                (1. + ti.math.exp(-gaussian_point_3d.alpha))
            prod_alpha = gaussian_alpha * point_alpha_after_activation
            # from paper: we skip any blending updates with 𝛼 < 𝜖 (we choose 𝜖 as 1
            # 255 ) and also clamp 𝛼 with 0.99 from above.
            if prod_alpha >= 1. / 255.:
                alpha: ti.f32 = ti.min(prod_alpha, 0.99)
                color, r_jacobian, g_jacobian, b_jacobian = gaussian_point_3d.get_color_with_jacobian_by_ray(
                    ray_origin=ray_origin,
                    ray_direction=ray_direction,
                )

                # TODO: have no idea why taichi does not allow the following code under debug. However, it works under release mode.
                accumulated_alpha = accumulated_alpha - alpha
                # print(
                #     f"({pixel_v}, {pixel_u}, {point_offset}, {point_offset - start_offset}), accumulated_alpha: {accumulated_alpha}")

                d_pixel_rgb_d_color = alpha * (1. - accumulated_alpha)
                # all vec16f
                color_r_grad = d_pixel_rgb_d_color * \
                    pixel_rgb_grad[0] * r_jacobian
                color_g_grad = d_pixel_rgb_d_color * \
                    pixel_rgb_grad[1] * g_jacobian
                color_b_grad = d_pixel_rgb_d_color * \
                    pixel_rgb_grad[2] * b_jacobian

                alpha_grad_from_rgb = (1. - accumulated_alpha) * \
                    color * pixel_rgb_grad
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
                translation_grad = point_viewspace_grad @ d_uv_d_translation
                # cov is Sigma
                gaussian_q_grad = gaussian_alpha_grad * \
                    d_p_d_cov_flat @ d_Sigma_prime_d_q
                gaussian_s_grad = gaussian_alpha_grad * \
                    d_p_d_cov_flat @ d_Sigma_prime_d_s
                atomic_accumulate_grad_for_point(
                    point_offset=point_offset,
                    point_in_camera_grad=point_in_camera_grad,
                    pointfeatures_grad=pointfeatures_grad,
                    viewspace_grad=viewspace_grad,
                    point_viewspace_grad=point_viewspace_grad,
                    translation_grad=translation_grad,
                    gaussian_q_grad=gaussian_q_grad,
                    gaussian_s_grad=gaussian_s_grad,
                    gaussian_point_3d_alpha_grad=gaussian_point_3d_alpha_grad,
                    color_r_grad=color_r_grad,
                    color_g_grad=color_g_grad,
                    color_b_grad=color_b_grad,
                )


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
        T_pointcloud_camera: torch.Tensor  # 4x4 x to the right, y down, z forward
        color_max_sh_band: int = 2

    @dataclass
    class BackwardValidPointHookInput:
        point_in_camera_id: torch.Tensor  # M
        grad_point_in_camera: torch.Tensor  # Mx3
        grad_pointfeatures_in_camera: torch.Tensor  # Mx56
        grad_viewspace: torch.Tensor  # Mx2

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
            def forward(ctx, pointcloud, pointcloud_features, T_pointcloud_camera, camera_info, color_max_sh_band):
                point_in_camera_mask = torch.zeros(
                    size=(pointcloud.shape[0],), dtype=torch.int8, device=pointcloud.device)
                point_id = torch.arange(
                    pointcloud.shape[0], dtype=torch.int32, device=pointcloud.device)
                T_camera_pointcloud = inverse_se3(T_pointcloud_camera)
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
                point_in_camera_mask = point_in_camera_mask.bool()
                point_in_camera_id = point_id[point_in_camera_mask].contiguous(
                )
                del point_id
                del point_in_camera_mask
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
                point_in_camera_sort_key, permutation = point_in_camera_sort_key.sort()
                point_in_camera_id = point_in_camera_id[permutation].contiguous(
                )  # now the point_in_camera_id is sorted by the sort_key
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

                num_points_in_camera = point_in_camera_id.shape[0]

                # in paper, these data are computed on the fly and saved in shared block memory.
                # however, taichi does not support shared block memory for ndarray, so we save them in global memory
                point_uv = torch.zeros(
                    size=(num_points_in_camera, 2), dtype=torch.float32, device=pointcloud.device)
                point_in_camera = torch.zeros(
                    size=(num_points_in_camera, 3), dtype=torch.float32, device=pointcloud.device)
                point_uv_covariance = torch.zeros(
                    size=(num_points_in_camera, 2, 2), dtype=torch.float32, device=pointcloud.device)

                generate_point_attributes_in_camera_plane(
                    pointcloud=pointcloud,
                    pointcloud_features=pointcloud_features,
                    camera_intrinsics=camera_info.camera_intrinsics,
                    T_camera_pointcloud=T_camera_pointcloud,
                    point_in_camera_id=point_in_camera_id,
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_covariance=point_uv_covariance,
                )

                rasterized_image = torch.zeros(
                    camera_info.camera_height, camera_info.camera_width, 3, dtype=torch.float32, device=pointcloud.device)
                pixel_accumulated_alpha = torch.zeros(
                    camera_info.camera_height, camera_info.camera_width, dtype=torch.float32, device=pointcloud.device)
                pixel_offset_of_last_effective_point = torch.zeros(
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
                    point_in_camera_id=point_in_camera_id,
                    point_uv=point_uv,
                    point_in_camera=point_in_camera,
                    point_uv_covariance=point_uv_covariance,
                    rasterized_image=rasterized_image,
                    pixel_accumulated_alpha=pixel_accumulated_alpha,
                    pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point)
                ctx.save_for_backward(
                    pointcloud,
                    pointcloud_features,
                    point_in_camera_id,
                    tile_points_start,
                    tile_points_end,
                    pixel_accumulated_alpha,
                    pixel_offset_of_last_effective_point,
                    T_pointcloud_camera,
                    T_camera_pointcloud
                )
                ctx.camera_info = camera_info
                ctx.color_max_sh_band = color_max_sh_band
                # rasterized_image.requires_grad_(True)
                return rasterized_image

            @staticmethod
            @custom_bwd
            def backward(ctx, grad_rasterized_image):
                grad_pointcloud = grad_pointcloud_features = grad_T_pointcloud_camera = None
                if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                    pointcloud, pointcloud_features, point_in_camera_id, tile_points_start, tile_points_end, pixel_accumulated_alpha, pixel_offset_of_last_effective_point, T_pointcloud_camera, T_camera_pointcloud = ctx.saved_tensors
                    camera_info = ctx.camera_info
                    color_max_sh_band = ctx.color_max_sh_band
                    grad_rasterized_image = grad_rasterized_image.contiguous()
                    grad_point_in_camera = torch.zeros(
                        size=(point_in_camera_id.shape[0], 3), dtype=torch.float32, device=pointcloud.device)
                    grad_pointfeatures = torch.zeros(
                        size=(point_in_camera_id.shape[0], pointcloud_features.shape[1]), dtype=torch.float32, device=pointcloud.device)
                    grad_viewspace = torch.zeros(
                        size=(point_in_camera_id.shape[0], 2), dtype=torch.float32, device=pointcloud.device)
                    gaussian_point_rasterisation_backward(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        camera_intrinsics=camera_info.camera_intrinsics,
                        T_camera_pointcloud=T_camera_pointcloud,
                        pointcloud=pointcloud,
                        pointcloud_features=pointcloud_features,
                        tile_points_start=tile_points_start,
                        point_in_camera_id=point_in_camera_id,
                        rasterized_image_grad=grad_rasterized_image,
                        pixel_accumulated_alpha=pixel_accumulated_alpha,
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                        point_in_camera_grad=grad_point_in_camera,
                        pointfeatures_grad=grad_pointfeatures,
                        viewspace_grad=grad_viewspace,
                    )
                    del tile_points_start, tile_points_end, pixel_accumulated_alpha, pixel_offset_of_last_effective_point
                    if backward_valid_point_hook is not None:
                        backward_valid_point_hook_input = GaussianPointCloudRasterisation.BackwardValidPointHookInput(
                            point_in_camera_id=point_in_camera_id,
                            grad_point_in_camera=grad_point_in_camera,
                            grad_pointfeatures_in_camera=grad_pointfeatures,
                            grad_viewspace=grad_viewspace,
                        )
                        backward_valid_point_hook(
                            backward_valid_point_hook_input)
                    grad_pointfeatures = self._clear_grad_by_color_max_sh_band(
                        grad_pointcloud_features=grad_pointfeatures,
                        color_max_sh_band=color_max_sh_band)
                    grad_pointcloud = torch.zeros_like(pointcloud)
                    grad_pointcloud_features = torch.zeros_like(
                        pointcloud_features)
                    grad_pointcloud[point_in_camera_id] = grad_point_in_camera
                    grad_pointcloud_features[point_in_camera_id] = \
                        grad_pointfeatures
                return grad_pointcloud, grad_pointcloud_features, grad_T_pointcloud_camera, None, None

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
        T_pointcloud_camera = input_data.T_pointcloud_camera
        color_max_sh_band = input_data.color_max_sh_band
        camera_info = input_data.camera_info
        assert camera_info.camera_width % 16 == 0
        assert camera_info.camera_height % 16 == 0
        return self._module_function.apply(pointcloud, pointcloud_features, T_pointcloud_camera, camera_info, color_max_sh_band)
