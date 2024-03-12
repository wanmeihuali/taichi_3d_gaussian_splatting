# Given a rendered scene, how can I estimate the camera pose of a novel image?

import argparse
import json
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import filter_point_in_camera, generate_point_attributes_in_camera_plane, \
    generate_num_overlap_tiles, generate_point_sort_key_by_num_overlap_tiles, find_tile_start_and_end, gaussian_point_rasterisation, load_point_cloud_row_into_gaussian_point_3d
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.GaussianPoint3D import rotation_matrix_from_quaternion, transform_matrix_from_quaternion_and_translation
from taichi_3d_gaussian_splatting.utils import grad_point_probability_density_from_conic, inverse_SE3_qt_torch, SE3_to_quaternion_and_translation_torch,\
    quaternion_to_rotation_matrix_torch, perturb_pose_quaternion_translation_torch
from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
# %%
import os
import PIL.Image
import torchvision
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer
from dataclass_wizard import YAMLWizard
from typing import List, Tuple, Optional, Callable, Union
import torch.nn.functional as F

mat4x0f = ti.types.matrix(n=1, m=4, dtype=ti.f32)
mat3x0f = ti.types.matrix(n=1, m=4, dtype=ti.f32)
mat1x4f = ti.types.matrix(n=1, m=4, dtype=ti.f32)
mat2x3f = ti.types.matrix(n=2, m=3, dtype=ti.f32)
mat2x4f = ti.types.matrix(n=2, m=4, dtype=ti.f32)
mat3x3f = ti.types.matrix(n=3, m=3, dtype=ti.f32)
mat3x4f = ti.types.matrix(n=3, m=4, dtype=ti.f32)

@ti.kernel
def gaussian_point_rasterisation_backward_with_pose(
    camera_height: ti.i32,
    camera_width: ti.i32,
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
    pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, K)
    point_object_id: ti.types.ndarray(ti.i32, ndim=1),  # (N)
    q_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 4)
    t_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (K, 3)
    q_pointcloud_camera: ti.types.ndarray(ti.f32, ndim=2),  # (1, 4)
    t_pointcloud_camera: ti.types.ndarray(ti.f32, ndim=2),  # (1, 3)
    grad_q: ti.types.ndarray(ti.f32, ndim=1),  # (4)
    grad_t: ti.types.ndarray(ti.f32, ndim=1),  # (3)
    # (tiles_per_row * tiles_per_col)
    tile_points_start: ti.types.ndarray(ti.i32, ndim=1),
    tile_points_end: ti.types.ndarray(ti.i32, ndim=1),
    point_offset_with_sort_key: ti.types.ndarray(ti.i32, ndim=1),  # (K)
    point_id_in_camera_list: ti.types.ndarray(ti.i32, ndim=1),  # (M)
    rasterized_image_grad: ti.types.ndarray(ti.f32, ndim=3),  # (H, W, 3)
    enable_depth_grad: ti.template(),
    rasterized_depth_grad: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    accumulated_alpha_grad: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    pixel_accumulated_alpha: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    rasterized_depth: ti.types.ndarray(ti.f32, ndim=2),  # (H, W)
    # (H, W)
    pixel_offset_of_last_effective_point: ti.types.ndarray(ti.i32, ndim=2),
    grad_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
    grad_pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, K)
    grad_uv: ti.types.ndarray(ti.f32, ndim=2),  # (N, 2)

    in_camera_grad_uv_cov_buffer: ti.types.ndarray(ti.f32, ndim=2),
    in_camera_grad_color_buffer: ti.types.ndarray(ti.f32, ndim=2),  # (M, 3)
    in_camera_grad_depth_buffer: ti.types.ndarray(ti.f32, ndim=1),  # (M)

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

    ti.loop_config(block_dim=256)
    for pixel_offset in ti.ndrange(camera_height * camera_width):
        # each block handles one tile, so tile_id is actually block_id
        tile_id = pixel_offset // 256
        thread_id = pixel_offset % 256
        tile_u = ti.cast(tile_id % (camera_width // 16), ti.i32)
        tile_v = ti.cast(tile_id // (camera_width // 16), ti.i32)

        start_offset = tile_points_start[tile_id]
        end_offset = tile_points_end[tile_id]
        tile_point_count = end_offset - start_offset

        tile_point_uv = ti.simt.block.SharedArray(
            (2, 256), dtype=ti.f32)  # 2KB shared memory
        tile_point_uv_conic = ti.simt.block.SharedArray(
            (3, 256), dtype=ti.f32)  # 4KB shared memory
        tile_point_color = ti.simt.block.SharedArray(
            (3, 256), dtype=ti.f32)  # 3KB shared memory
        tile_point_alpha = ti.simt.block.SharedArray(
            (256,), dtype=ti.f32)  # 1KB shared memory
        tile_point_depth = ti.simt.block.SharedArray(
            (ti.static(256 if enable_depth_grad else 0),), dtype=ti.f32)  # 1KB shared memory

        pixel_offset_in_tile = pixel_offset - tile_id * 256
        pixel_offset_u_in_tile = pixel_offset_in_tile % 16
        pixel_offset_v_in_tile = pixel_offset_in_tile // 16
        pixel_u = tile_u * 16 + pixel_offset_u_in_tile
        pixel_v = tile_v * 16 + pixel_offset_v_in_tile
        last_effective_point = pixel_offset_of_last_effective_point[pixel_v, pixel_u]
        org_accumulated_alpha: ti.f32 = pixel_accumulated_alpha[pixel_v, pixel_u]
        accumulated_alpha: ti.f32 = pixel_accumulated_alpha[pixel_v, pixel_u]
        accumulated_alpha_grad_value: ti.f32 = accumulated_alpha_grad[pixel_v, pixel_u]
        d_pixel: ti.f32 = rasterized_depth[pixel_v, pixel_u]
        T_i = 1.0 - accumulated_alpha  # T_i = \prod_{j=1}^{i-1} (1 - a_j)
        # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} \sum_{j=i+1}^{n} c_j a_j T(j)
        # let w_i = \sum_{j=i+1}^{n} c_j a_j T(j)
        # we have w_n = 0, w_{i-1} = w_i + c_i a_i T(i)
        # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} w_i
        w_i = ti.math.vec3(0.0, 0.0, 0.0)
        depth_w_i = 0.0
        acc_alpha_w_i = 0.0

        pixel_rgb_grad = ti.math.vec3(
            rasterized_image_grad[pixel_v, pixel_u, 0], rasterized_image_grad[pixel_v, pixel_u, 1], rasterized_image_grad[pixel_v, pixel_u, 2])
        pixel_depth_grad = rasterized_depth_grad[pixel_v,
                                                 pixel_u] if enable_depth_grad else 0.0
        total_magnitude_grad_viewspace_on_image = ti.math.vec2(0.0, 0.0)

        # for inverse_point_offset in range(effective_point_count):
        # taichi only supports range() with start and end
        # for inverse_point_offset_base in range(0, tile_point_count, 256):
        num_point_blocks = (tile_point_count + 255) // 256
        for point_block_id in range(num_point_blocks):
            inverse_point_offset_base = point_block_id * 256
            block_end_idx_point_offset_with_sort_key = end_offset - inverse_point_offset_base
            block_start_idx_point_offset_with_sort_key = ti.max(
                block_end_idx_point_offset_with_sort_key - 256, 0)
            # in the later loop, we will handle the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key)
            # so we need to load the points in [block_start_idx_point_offset_with_sort_key, block_end_idx_point_offset_with_sort_key - 1]
            to_load_idx_point_offset_with_sort_key = block_end_idx_point_offset_with_sort_key - thread_id - 1
            if to_load_idx_point_offset_with_sort_key >= block_start_idx_point_offset_with_sort_key:
                to_load_point_offset = point_offset_with_sort_key[to_load_idx_point_offset_with_sort_key]
                to_load_uv = ti.math.vec2(
                    [point_uv[to_load_point_offset, 0], point_uv[to_load_point_offset, 1]])

                if enable_depth_grad:
                    tile_point_depth[thread_id] = point_in_camera[to_load_point_offset, 2]

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
                256, tile_point_count - inverse_point_offset_base)
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

                    # accumulated_alpha_i = 1. - T_i #alpha after passing current point
                    # Transmittance before passing current point
                    T_i = T_i / (1. - alpha)
                    accumulated_alpha = 1. - T_i  # accumulated alha before passing current point

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

                    if enable_depth_grad:
                        depth_i = tile_point_depth[idx_point_offset_with_sort_key_in_block]
                        alpha_grad_from_depth = (depth_i * T_i - depth_w_i / (1. - alpha)) \
                            * pixel_depth_grad
                        depth_w_i += depth_i * alpha * T_i
                        alpha_grad += alpha_grad_from_depth

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
                    if enable_depth_grad:
                        point_depth_grad = alpha * T_i * pixel_depth_grad
                        ti.atomic_add(
                            in_camera_grad_depth_buffer[point_offset], point_depth_grad)

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
            # end of the 256 block loop
            ti.simt.block.sync()
        # end of the backward traversal loop, from last point to first point
        if need_extra_info:
            magnitude_grad_viewspace_on_image[pixel_v, pixel_u,
                                              0] = total_magnitude_grad_viewspace_on_image[0]
            magnitude_grad_viewspace_on_image[pixel_v, pixel_u,
                                              1] = total_magnitude_grad_viewspace_on_image[1]
    # end of per pixel loop

    q_pointcloud_camera_taichi = ti.math.vec4(
        q_pointcloud_camera[0, 0], q_pointcloud_camera[0, 1], q_pointcloud_camera[0, 2], q_pointcloud_camera[0, 3],)
    # q_pointcloud_camera_taichi = ti.Vector(
    #                 [q_pointcloud_camera[:, idx] for idx in ti.static(range(4))])
    # t_pointcloud_camera_taichi = ti.Vector(
    #                     [t_pointcloud_camera[:, idx] for idx in ti.static(range(3))])
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
        point_grad_depth = in_camera_grad_depth_buffer[idx] if enable_depth_grad else 0.

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

        # -------------------------------------------------------------
        # Pose optimization code
        w_t_w_point = gaussian_point_3d.translation
        R_guess = rotation_matrix_from_quaternion(q_pointcloud_camera_taichi)

        d_uv_d_translation_camera = project_to_camera_relative_position_jacobian(
            w_t_w_point, T_camera_pointcloud_mat, camera_intrinsics_mat)
        dR_dqx, dR_dqy, dR_dqz, dR_dqw = quaternion_to_rotation_matrix_torch_jacobian(
            (q_pointcloud_camera[0,0], q_pointcloud_camera[0,1], q_pointcloud_camera[0,2], q_pointcloud_camera[0,3]))
        # x, y, z: coordinate of points in camera frame
        dx_dq = mat1x4f([[dR_dqx[0, 0]*w_t_w_point[0] + dR_dqx[1, 0]*w_t_w_point[1] + dR_dqx[2, 0]*w_t_w_point[2]],
                              [dR_dqy[0, 0]*w_t_w_point[0] + dR_dqy[1, 0] *
                                  w_t_w_point[1] + dR_dqy[2, 0]*w_t_w_point[2]],
                              [dR_dqz[0, 0]*w_t_w_point[0] + dR_dqz[1, 0] *
                                  w_t_w_point[1] + dR_dqz[2, 0]*w_t_w_point[2]],
                              [dR_dqw[0, 0]*w_t_w_point[0] + dR_dqw[1, 0]*w_t_w_point[1] + dR_dqw[2, 0]*w_t_w_point[2]]])

        dy_dq = mat1x4f([[dR_dqx[0, 1]*w_t_w_point[0] + dR_dqx[1, 1]*w_t_w_point[1] + dR_dqx[2, 1]*w_t_w_point[2]],
                              [dR_dqy[0, 1]*w_t_w_point[0] + dR_dqy[1, 1] *
                                  w_t_w_point[1] + dR_dqy[2, 1]*w_t_w_point[2]],
                              [dR_dqz[0, 1]*w_t_w_point[0] + dR_dqz[1, 1] *
                                  w_t_w_point[1] + dR_dqz[2, 1]*w_t_w_point[2]],
                              [dR_dqw[0, 1]*w_t_w_point[0] + dR_dqw[1, 1]*w_t_w_point[1] + dR_dqw[2, 1]*w_t_w_point[2]]])

        dz_dq = mat1x4f([[dR_dqx[0, 2]*w_t_w_point[0] + dR_dqx[1, 2]*w_t_w_point[1] + dR_dqx[2, 2]*w_t_w_point[2]],
                              [dR_dqy[0, 2]*w_t_w_point[0] + dR_dqy[1, 2] *
                                  w_t_w_point[1] + dR_dqy[2, 2]*w_t_w_point[2]],
                              [dR_dqz[0, 2]*w_t_w_point[0] + dR_dqz[1, 2] *
                                  w_t_w_point[1] + dR_dqz[2, 2]*w_t_w_point[2]],
                              [dR_dqw[0, 2]*w_t_w_point[0] + dR_dqw[1, 2]*w_t_w_point[1] + dR_dqw[2, 2]*w_t_w_point[2]]])

        d_translation_camera_d_q = mat3x4f([[dx_dq[0,0], dx_dq[0,1], dx_dq[0,2], dx_dq[0,3]],
                                            [dy_dq[0,0], dy_dq[0,1], dy_dq[0,2], dy_dq[0,3]],
                                            [dz_dq[0,0], dz_dq[0,1], dz_dq[0,2], dz_dq[0,3]]])

        dxyz_d_t_world_camera = -R_guess.transpose()
        point_grad_q = d_uv_d_translation_camera @ d_translation_camera_d_q
        point_grad_t = d_uv_d_translation_camera @ dxyz_d_t_world_camera

        multiply = point_grad_uv @ point_grad_q
        grad_q[0] = grad_q[0] + multiply[0]
        grad_q[1] = grad_q[1] + multiply[1]
        grad_q[2] = grad_q[2] + multiply[2]
        grad_q[3] = grad_q[3] + multiply[3]
        
        multiply_t = point_grad_uv @ point_grad_t
        grad_t[0] = grad_t[0] + multiply_t[0]
        grad_t[1] = grad_t[1] + multiply_t[1]
        grad_t[2] = grad_t[2] + multiply_t[2]

        # ------------------------------------------------------------

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

        translation_grad = ti.math.vec3([0., 0., 0.])
        if enable_depth_grad:
            d_depth_d_translation = gaussian_point_3d.depth_jacobian(
                T_camera_world=T_camera_pointcloud_mat,
            )
            translation_grad = point_grad_uv @ d_uv_d_translation + \
                point_grad_depth * d_depth_d_translation
        else:
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


@ti.kernel
def torchImage2tiImage(field: ti.template(), data: ti.types.ndarray()):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        field[col, data.shape[0] - row -
              1] = ti.math.vec3(data[row, col, 0], data[row, col, 1], data[row, col, 2])

@ti.func
def quaternion_to_rotation_matrix_torch_jacobian(q):
    qx, qy, qz, qw = q
    dR_dqx = mat3x3f([
        [0, 2*qy, 2*qz],
        [2*qy, -4*qx, -2*qw],
        [2*qz, 2*qw, -4*qx]
    ])
    dR_dqy = mat3x3f([
        [-4*qy, 2*qx, 2*qw],
        [2*qx, 0, 2*qz],
        [-2*qw, 2*qz, -4*qy]
    ])
    dR_dqz = mat3x3f([
        [-4*qz, -2*qw, 2*qx],
        [2*qw, -4*qz, 2*qy],
        [2*qx, 2*qy, 0]
    ])
    dR_dqw = mat3x3f([
        [0, -2*qz, 2*qy],
        [2*qz, 0, -2*qx],
        [-2*qy, 2*qx, 0]
    ])
    return dR_dqx, dR_dqy, dR_dqz, dR_dqw

@ti.func
def project_to_camera_relative_position_jacobian(
    w_t_w_points,
    T_camera_world,
    projective_transform,
):

    w_t_w_pointst_homogeneous = ti.math.vec4(
            [w_t_w_points[0], w_t_w_points[1], w_t_w_points[2], 1])
    t = T_camera_world @ w_t_w_pointst_homogeneous
    K = projective_transform

    d_uv_d_translation_camera = mat2x3f([
        [K[0, 0] / t[2], K[0, 1] / t[2], (-K[0, 0] * t[0] - K[0, 1] * t[1]) / (t[2] * t[2])],
        [K[1, 0] / t[2], K[1, 1] / t[2], (-K[1, 0] * t[0] - K[1, 1] * t[1]) / (t[2] * t[2])]])

    return d_uv_d_translation_camera


class PoseModel(torch.nn.Module):
    @dataclass
    class PoseModelConfig(YAMLWizard):
        near_plane: float = 0.8
        far_plane: float = 1000.
        depth_to_sort_key_scale: float = 100.
        rgb_only: bool = False
        grad_color_factor = 5.
        grad_high_order_color_factor = 1.
        grad_s_factor = 0.5
        grad_q_factor = 1.
        grad_alpha_factor = 20.
        enable_depth_grad = True

    @dataclass
    class PoseModelInput:
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
        config: PoseModelConfig,
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

                q_pointcloud_camera = F.normalize(q_pointcloud_camera, p=2, dim=-1)
                
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

                tiles_per_row = camera_info.camera_width // 16
                tiles_per_col = camera_info.camera_height // 16
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
                    # import ipdb;ipdb.set_trace()
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
                    rasterized_depth,
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
                return rasterized_image, rasterized_depth, pixel_valid_point_count, pixel_accumulated_alpha

            @staticmethod
            def backward(ctx, grad_rasterized_image, grad_rasterized_depth,
                         grad_pixel_valid_point_count, grad_pixel_accumulated_alpha):
                grad_pointcloud = grad_pointcloud_features = grad_q_pointcloud_camera = grad_t_pointcloud_camera = None
                if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                    pointcloud, \
                        pointcloud_features, \
                        point_offset_with_sort_key, \
                        point_id_in_camera_list, \
                        tile_points_start, \
                        tile_points_end, \
                        pixel_accumulated_alpha, \
                        rasterized_depth, \
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
                    enable_depth_grad = self.config.enable_depth_grad
                    if enable_depth_grad:
                        grad_rasterized_depth = grad_rasterized_depth.contiguous()
                        in_camera_grad_depth_buffer = torch.zeros(
                            size=(point_id_in_camera_list.shape[0], ), dtype=torch.float32, device=pointcloud.device)
                    else:  # taichi does not support None for tensor, so we use an empty tensor instead
                        grad_rasterized_depth = torch.empty(
                            size=(0, 0, ), dtype=torch.float32, device=pointcloud.device)
                        in_camera_grad_depth_buffer = torch.empty(
                            size=(0, ), dtype=torch.float32, device=pointcloud.device)
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

                    grad_q = torch.zeros(
                        size=(4,), dtype=torch.float32, device=pointcloud.device)
                    grad_t = torch.zeros(
                        size=(3,), dtype=torch.float32, device=pointcloud.device)

    
                    grad_q = torch.squeeze(grad_q)
                    
                    grad_q_taichi = ti.math.vec4([0.,0.,0.,0.])
                    #grad_q_taichi.from_torch(grad_q.clone().detach())
                    grad_t = torch.squeeze(grad_t)
                    grad_t_taichi = ti.math.vec3([0.,0.,0.])                 
                    gaussian_point_rasterisation_backward_with_pose(
                        camera_height=camera_info.camera_height,
                        camera_width=camera_info.camera_width,
                        camera_intrinsics=camera_info.camera_intrinsics,
                        point_object_id=point_object_id,
                        q_camera_pointcloud=q_camera_pointcloud,
                        t_camera_pointcloud=t_camera_pointcloud,
                        q_pointcloud_camera=q_pointcloud_camera.contiguous(),
                        t_pointcloud_camera=t_pointcloud_camera.contiguous(),
                        grad_q=grad_q,
                        grad_t=grad_t,
                        pointcloud=pointcloud,
                        pointcloud_features=pointcloud_features,
                        tile_points_start=tile_points_start,
                        tile_points_end=tile_points_end,
                        point_offset_with_sort_key=point_offset_with_sort_key,
                        point_id_in_camera_list=point_id_in_camera_list,
                        rasterized_image_grad=grad_rasterized_image,
                        enable_depth_grad=enable_depth_grad,
                        rasterized_depth_grad=grad_rasterized_depth,
                        accumulated_alpha_grad=grad_pixel_accumulated_alpha,
                        pixel_accumulated_alpha=pixel_accumulated_alpha,
                        rasterized_depth=rasterized_depth,
                        pixel_offset_of_last_effective_point=pixel_offset_of_last_effective_point,
                        grad_pointcloud=grad_pointcloud,
                        grad_pointcloud_features=grad_pointcloud_features,
                        grad_uv=grad_viewspace,
                        in_camera_grad_uv_cov_buffer=in_camera_grad_uv_cov_buffer,
                        in_camera_grad_color_buffer=in_camera_grad_color_buffer,
                        in_camera_grad_depth_buffer=in_camera_grad_depth_buffer,
                        point_uv=point_uv,
                        point_in_camera=point_in_camera,
                        point_uv_conic=point_uv_conic,
                        point_alpha_after_activation=point_alpha_after_activation,
                        point_color=point_color,
                        need_extra_info=True,
                        magnitude_grad_viewspace=magnitude_grad_viewspace,
                        magnitude_grad_viewspace_on_image=magnitude_grad_viewspace_on_image,
                        in_camera_num_affected_pixels=in_camera_num_affected_pixels,
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
                        point_id_in_camera_list = point_id_in_camera_list.contiguous().long()
                        backward_valid_point_hook_input = PoseModel.BackwardValidPointHookInput(
                            point_id_in_camera_list=point_id_in_camera_list,
                            grad_point_in_camera=grad_pointcloud[point_id_in_camera_list.long(
                            )],
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

                grad_q_pointcloud_camera = grad_q.view(1, -1)
                grad_t_pointcloud_camera = grad_t.view(1, -1)
                # same as inputs of forward method
                               
                return grad_pointcloud, \
                    grad_pointcloud_features, \
                    None, \
                    None, \
                    grad_q_pointcloud_camera, \
                    grad_t_pointcloud_camera, \
                    None, \
                    None

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

    def forward(self, input_data: PoseModelInput):
        pointcloud = input_data.point_cloud
        pointcloud_features = input_data.point_cloud_features
        point_invalid_mask = input_data.point_invalid_mask
        point_object_id = input_data.point_object_id
        q_pointcloud_camera = input_data.q_pointcloud_camera
        t_pointcloud_camera = input_data.t_pointcloud_camera
        color_max_sh_band = input_data.color_max_sh_band
        camera_info = input_data.camera_info
        assert camera_info.camera_width % 16 == 0
        assert camera_info.camera_height % 16 == 0
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


class PoseEstimator():
    @dataclass
    class PoseEstimatorConfig:
        device: str = "cuda"
        image_height: int = 405
        image_width: int = 720
        camera_intrinsics: torch.Tensor = torch.tensor([[400, 0, 360],
                                                        [0, 400, 202.5],
                                                        [0, 0, 1]], device="cuda")
        parquet_path: str = None
        initial_guess_T_pointcloud_camera = torch.tensor([
            [0.059764470905065536, 0.4444755017757416, -0.8937951326370239, 0.],
            [0.9982125163078308, -0.026611410081386566, 0.05351284518837929, 0.],
            [0.0, -0.8953956365585327, -0.44527140259742737, 0.],
            [0.0, 0.0, 0.0, 1.0]])
        image_path_list: List[str] = None
        json_file_path: str = None
        
    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, output_path, config) -> None:
        self.config = config
        self.output_path = output_path
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16

        scene = GaussianPointCloudScene.from_parquet(
            self.config.parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
        self.scene = self._merge_scenes([scene])
        self.scene = self.scene.to(self.config.device)

        # Initial guess

        self.initial_guess_T_pointcloud_camera = self.config.initial_guess_T_pointcloud_camera.to(
            self.config.device)
        initial_guess_T_pointcloud_camera = self.initial_guess_T_pointcloud_camera.unsqueeze(
            0)
        self.initial_guess_q_pointcloud_camera, self.initial_guess_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
            initial_guess_T_pointcloud_camera)

        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(
                self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_id=0,
        )

        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(
            self.config.image_width, self.config.image_height))

        self.rasteriser = PoseModel(
            config=PoseModel.PoseModelConfig(
                near_plane=0.001,
                far_plane=1000.,
                depth_to_sort_key_scale=100.))

    def _merge_scenes(self, scene_list):
        # the config does not matter here, only for training

        merged_point_cloud = torch.cat(
            [scene.point_cloud for scene in scene_list], dim=0)
        merged_point_cloud_features = torch.cat(
            [scene.point_cloud_features for scene in scene_list], dim=0)
        num_of_points_list = [scene.point_cloud.shape[0]
                              for scene in scene_list]
        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
        end_offset_list = np.cumsum(num_of_points_list).tolist()
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                center=scene_list[idx].point_cloud.mean(dim=0),
                visible=True
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        point_object_id = torch.zeros(
            (merged_point_cloud.shape[0],), dtype=torch.int32, device=self.config.device)
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            point_object_id=point_object_id,
            config=GaussianPointCloudScene.PointCloudSceneConfig(
                max_num_points_ratio=None
            ))
        return merged_scene

    def start(self):
        d = self.config.image_path_list
        count = 0
        with open(self.config.json_file_path) as f:
            d = json.load(f)
            for view in d:
                # Load groundtruth image
                ground_truth_image_path = view["image_path"]
                print(f"Loading image {ground_truth_image_path}")
                ground_truth_image_numpy = np.array(
                    PIL.Image.open(ground_truth_image_path))
                ground_truth_image = torchvision.transforms.functional.to_tensor(
                    ground_truth_image_numpy)

                ground_truth_image, resized_camera_info, _ = GaussianPointCloudTrainer._downsample_image_and_camera_info(ground_truth_image,
                                                                                                                        None,
                                                                                                                        self.camera_info,
                                                                                                                        1)
                ground_truth_image = ground_truth_image.cuda()
                
                self.camera_info = resized_camera_info
                groundtruth_T_pointcloud_camera = torch.tensor(
                    view["T_pointcloud_camera"],
                    device="cuda")

                groundtruth_T_pointcloud_camera = groundtruth_T_pointcloud_camera.unsqueeze(
                    0)
                groundtruth_q_pointcloud_camera, groundtruth_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
                    groundtruth_T_pointcloud_camera)
                print(f"Ground truth q: \n\t {groundtruth_q_pointcloud_camera}")
                print(f"Ground truth t: \n\t {groundtruth_t_pointcloud_camera}")
                initial_guess_q_pointcloud_camera, initial_guess_t_pointcloud_camera = perturb_pose_quaternion_translation_torch(groundtruth_q_pointcloud_camera,\
                    groundtruth_t_pointcloud_camera,\
                        0.05, 0.3)
                initial_guess_q_pointcloud_camera = torch.tensor([ [0.6749, 0.5794,  -0.3524, -0.2909]], device="cuda")
                initial_guess_t_pointcloud_camera = torch.tensor([ [0.2528, 0.1397,  -0.0454]], device="cuda")
                initial_guess_q_pointcloud_camera.requires_grad = True
                initial_guess_t_pointcloud_camera.requires_grad = True
                print(f"Ground truth transformation world to camera, in camera frame: \n\t {groundtruth_T_pointcloud_camera}")
                print(f"Initial guess q: \n\t {initial_guess_q_pointcloud_camera}")
                print(f"Initial guess t: \n\t {initial_guess_t_pointcloud_camera}")
                
                # Save groundtruth image
                im = PIL.Image.fromarray((ground_truth_image_numpy).astype(np.uint8))
                if not os.path.exists(os.path.join(self.output_path,f'groundtruth/')):
                    os.makedirs(os.path.join(self.output_path,'groundtruth/'))
                im.save(os.path.join(self.output_path,f'groundtruth/groundtruth_{count}.png'))
                
                # Optimization starts
                optimizer_q = torch.optim.Adam(
                   [initial_guess_q_pointcloud_camera], lr=0.001)
                optimizer_t = torch.optim.Adam(
                    [initial_guess_t_pointcloud_camera], lr=0.001)
                
                num_epochs = 20000
                for epoch in range(num_epochs):          
                    optimizer_q.zero_grad()
                    optimizer_t.zero_grad()
                    
                    predicted_image, _, _, _ = self.rasteriser(
                        PoseModel.PoseModelInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,                    
                            q_pointcloud_camera=initial_guess_q_pointcloud_camera,
                            t_pointcloud_camera=initial_guess_t_pointcloud_camera,
                            camera_info=self.camera_info,
                            color_max_sh_band=3,
                        )
                    )
                    predicted_image = predicted_image.permute(2, 0, 1)
                    L1 = torch.abs(predicted_image - ground_truth_image).mean()
                    L1.backward()

                    if not torch.isnan(initial_guess_t_pointcloud_camera.grad).any():                      
                        torch.nn.utils.clip_grad_norm_(
                            initial_guess_t_pointcloud_camera, max_norm=1.0)
                        optimizer_t.step()
                    else:
                        print("Skipped epoch ", epoch)
                        print(previous_initial_guess_t_pointcloud_camera)
                        print(previous_initial_guess_t_pointcloud_camera)
                        # image_np = predicted_image.cpu().detach().numpy()
                        # im = PIL.Image.fromarray(
                        #     (image_np.transpose(1, 2, 0)*255).astype(np.uint8))
                        # if not os.path.exists(os.path.join(self.output_path, f'epochs/')):
                        #     os.makedirs(os.path.join(self.output_path, 'epochs/'))
                        # im.save(os.path.join(self.output_path,
                        #         f'epochs/epoch_{epoch}_problematic.png'))
                    
                    if not torch.isnan(initial_guess_q_pointcloud_camera.grad).any():
                        torch.nn.utils.clip_grad_norm_(
                            initial_guess_q_pointcloud_camera, max_norm=1.0)
                        optimizer_q.step()
                        
                        
                    if (epoch + 1) % 50 == 0 and epoch > 100:
                        with torch.no_grad():
                            print(f"============== epoch {epoch + 1} ==========================")
                            print(f"loss:{L1}")
                            q_pointcloud_camera = F.normalize(initial_guess_q_pointcloud_camera, p=2, dim=-1)
                            R = quaternion_to_rotation_matrix_torch(
                                q_pointcloud_camera)
                            print("Estimated rotation")
                            print(R)
                            print(f"Estimated translation: \n\t {initial_guess_t_pointcloud_camera}")
                            print(f"Gradient translation: \n\t {initial_guess_t_pointcloud_camera.grad}")
                            print("Ground truth transformation world to camera, in camera frame:")
                            print(groundtruth_T_pointcloud_camera)
                            image_np = predicted_image.cpu().detach().numpy()
                            im = PIL.Image.fromarray(
                                (image_np.transpose(1, 2, 0)*255).astype(np.uint8))
                            if not os.path.exists(os.path.join(self.output_path, f'epochs/')):
                                os.makedirs(os.path.join(self.output_path, 'epochs/'))
                            im.save(os.path.join(self.output_path,
                                    f'epochs/epoch_{epoch}.png'))
                            np.savetxt(os.path.join(self.output_path, f'epochs/epoch_{epoch}_q.txt'), q_pointcloud_camera.cpu().detach().numpy())
                            np.savetxt(os.path.join(self.output_path, f'epochs/epoch_{epoch}_t.txt'), initial_guess_t_pointcloud_camera.cpu().detach().numpy())
                    previous_initial_guess_t_pointcloud_camera = initial_guess_t_pointcloud_camera.clone().detach()
                    previous_grad_t_pointcloud_camera = initial_guess_t_pointcloud_camera.grad
                break # Only optimize on the first image
                count += 1

parser = argparse.ArgumentParser(description='Parquet file path')
parser.add_argument('--parquet_path', type=str, help='Parquet file path')
# parser.add_argument('--images-path', type=str, help='Images file path')
parser.add_argument('--json_file_path', type=str, help='Json trajectory file path')
parser.add_argument('--output_path', type=str, help='Output folder path')

# # Let's keep things easy with just 1 picture for now
# image_path = [
#     "/media/scratch1/mroncoroni/colmap_projects/replica/room_1_high_quality_500_frames/images/frame000000.jpg"]
args = parser.parse_args()

print("Opening parquet file ", args.parquet_path)
parquet_path_list = args.parquet_path
ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
visualizer = PoseEstimator(args.output_path, config=PoseEstimator.PoseEstimatorConfig(
    parquet_path=parquet_path_list,
    json_file_path=args.json_file_path
))
visualizer.start()
