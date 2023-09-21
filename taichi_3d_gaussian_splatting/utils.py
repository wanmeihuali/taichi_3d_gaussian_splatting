import numpy as np
import taichi as ti
import taichi.math as tm
import torch
from .Camera import CameraInfo
from typing import Tuple

data_type = ti.f32
torch_type = torch.float32


@ti.func
def intersect_ray_with_ellipsoid(
    ray_origin: tm.vec3,
    ray_direction: tm.vec3,
    ellipsoid_R: tm.mat3,  # R
    ellipsoid_t: tm.vec3,
    ellipsoid_S: tm.vec3,
    eps: ti.f32 = 1e-5
):
    """ intersect a ray with an ellipsoid

    Args:
        ray_origin (tm.vec3): the origin of the ray in the world space
        ray_direction (tm.vec3): the direction of the ray in the world space
        ellipsoid_R (tm.mat3): the rotation matrix of the ellipsoid
        ellipsoid_S (tm.vec3): the scale of the ellipsoid
        eps (ti.f32, optional): _description_. Defaults to 1e-5.

    Returns:
        (ti.i32, tm.vec3): whether the ray intersects with the ellipsoid, and the intersection point in the world space
    """
    o = ray_origin
    d = ray_direction
    t = ellipsoid_t
    R = ellipsoid_R
    S = ellipsoid_S
    has_intersection = False
    intersection_point = tm.vec3(0.0, 0.0, 0.0)

    inv_transform_matrix = tm.mat3([
        [1 / S[0], 0, 0],
        [0, 1 / S[1], 0],
        [0, 0, 1 / S[2]]
    ]) @ (R.transpose())
    o_transformed = inv_transform_matrix @ (o - t)
    d_transformed = inv_transform_matrix @ d

    A = tm.dot(d_transformed, d_transformed)
    if abs(A) < eps:
        A = eps

    B = 2 * tm.dot(o_transformed, d_transformed)
    C = tm.dot(o_transformed, o_transformed) - 1

    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        has_intersection = False
    else:
        if abs(discriminant) < eps:
            discriminant = 0

        sqrt_discriminant = ti.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)

        if t1 < 0 and t2 < 0:
            has_intersection = False
        else:
            t_intersect = t1 if t1 >= 0 else t2
            if abs(t1 - t2) < eps:
                t_intersect = ti.min(t1, t2)

            intersection_point_transformed = o_transformed + t_intersect * d_transformed
            transform_mat = R @ tm.mat3([
                [S[0], 0, 0],
                [0, S[1], 0],
                [0, 0, S[2]]])
            intersection_point = transform_mat @ intersection_point_transformed + t

            has_intersection = True
    return has_intersection, intersection_point


@ti.func
def get_point_to_line_vector(
    point: tm.vec3,
    line_origin: tm.vec3,
    line_direction: tm.vec3
):
    """ given a point and a line, return the vector from the point to the line

    Args:
        point (tm.vec3): the point, x, y, z
        line_origin (tm.vec3): the origin of the line(ray) in the same space as the point, x, y, z
        line_direction (tm.vec3): the direction of the line(ray), x, y, z

    Returns:
        tm.vec3: the vector from the point to the line
    """
    p = point
    o = line_origin
    d = line_direction
    op = p - o
    scale_factor = ti.math.dot(op, d) / ti.math.dot(d, d)
    q = o + scale_factor * d
    qp = p - q
    return qp


def get_ray_origin_and_direction_from_camera(
    T_pointcloud_camera: torch.Tensor,
    camera_info: CameraInfo
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ get the ray origin and direction for each pixel from the camera. ray starts from the camera center and goes through 
    the center of each pixel

    Args:
        T_pointcloud_camera (torch.Tensor): 4x4 SE(3) matrix, transforms points from the camera frame to the pointcloud frame
        camera_info (CameraInfo): the camera info, contain camera_width, camera_height, camera_intrinsics(3x3)

    Returns:
        (torch.Tensor, torch.Tensor): the ray origin and direction for each ray, ray_origin: (3,), direction: (H, W, 3)
    """
    T_camera_pointcloud = inverse_SE3(T_pointcloud_camera)
    ray_origin = T_pointcloud_camera[:3, 3]
    """ consider how we get a point(x, y, z)'s position in the camera frame:
    p_in_camera_frame = T_camera_pointcloud * [x, y, z, 1]^T
    let p_in_camera_frame be [x', y', z', 1]^T
    |u|   |fx  0  cx|   |x'/z'|                     
    |v| = |0   fy cy| * |y'/z'|, where u, v are the pixel coordinates in the image
    |1|   |0   0  1 |   | 1  |
    now, we want to get the direction of the ray, which is the vector from the camera center to the point
    we take z' = 1, so we have:
    |x'|   |fx  0  cx|^-1   |u|
    |y'| = |0   fy cy|    * |v|
    |1 |   |0   0  1 |      |1|
    , where x', y' 1 are the direction in the camera frame
    then we can get the direction of the ray by:
    direction = T_pointcloud_camera[:3, :3] * [x', y', 1]^T
    """
    pixel_v, pixel_u = torch.meshgrid(torch.arange(
        camera_info.camera_height), torch.arange(camera_info.camera_width))
    pixel_u, pixel_v = pixel_u.float(), pixel_v.float()
    pixel_u += 0.5  # add 0.5 to make the pixel coordinates be the center of the pixel
    pixel_v += 0.5  # add 0.5 to make the pixel coordinates be the center of the pixel
    pixel_uv_1 = torch.stack(
        [pixel_u, pixel_v, torch.ones_like(pixel_u)], dim=-1)  # (H, W, 3)
    pixel_uv_1 = pixel_uv_1.reshape(-1,
                                    3).to(camera_info.camera_intrinsics.device)
    fx = camera_info.camera_intrinsics[0, 0]
    fy = camera_info.camera_intrinsics[1, 1]
    cx = camera_info.camera_intrinsics[0, 2]
    cy = camera_info.camera_intrinsics[1, 2]
    inv_camera_intrinsics = torch.tensor([
        [1 / fx, 0, -cx / fx],
        [0, 1 / fy, -cy / fy],
        [0, 0, 1]], dtype=camera_info.camera_intrinsics.dtype, device=camera_info.camera_intrinsics.device)
    # (3, H*W)
    pixel_direction_in_camera = inv_camera_intrinsics @ pixel_uv_1.T
    direction = T_pointcloud_camera[:3, :3] @ \
        pixel_direction_in_camera  # (3, H*W)
    direction = direction.T.reshape(
        camera_info.camera_height, camera_info.camera_width, 3)  # (H, W, 3)
    direction = direction / \
        torch.norm(direction, dim=-1, keepdim=True)  # (H, W, 3)
    return ray_origin, direction


@ti.func
def get_ray_origin_and_direction_by_uv(
    pixel_u: ti.i32,
    pixel_v: ti.i32,
    camera_intrinsics: ti.math.mat3,
    T_camera_pointcloud: ti.math.mat4,
):
    pixel_uv = ti.math.vec2(pixel_u, pixel_v)
    pixel_uv_center = pixel_uv + 0.5
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    pixel_direction_in_cameraspace = ti.math.vec3(
        [(pixel_uv_center.x - cx) / fx, (pixel_uv_center.y - cy) / fy, 1])
    T_pointcloud_camera = taichi_inverse_SE3(T_camera_pointcloud)
    ray_origin = ti.math.vec3(
        [T_pointcloud_camera[0, 3], T_pointcloud_camera[1, 3], T_pointcloud_camera[2, 3]])
    R_pointcloud_camera = ti.math.mat3([
        [T_pointcloud_camera[0, 0], T_pointcloud_camera[0, 1],
            T_pointcloud_camera[0, 2]],
        [T_pointcloud_camera[1, 0], T_pointcloud_camera[1, 1],
            T_pointcloud_camera[1, 2]],
        [T_pointcloud_camera[2, 0], T_pointcloud_camera[2, 1], T_pointcloud_camera[2, 2]]
    ])
    ray_direction = R_pointcloud_camera @ pixel_direction_in_cameraspace
    ray_direction = ti.math.normalize(ray_direction)
    return ray_origin, ray_direction


@ti.func
def quaternion_multiply(q1: ti.math.vec4, q2: ti.math.vec4) -> ti.math.vec4:
    return ti.math.vec4([
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    ])


@ti.func
def quaternion_conjugate(q: ti.math.vec4) -> ti.math.vec4:
    return ti.math.vec4([-q.x, -q.y, -q.z, q.w])


@ti.func
def quaternion_rotate(q: ti.math.vec4, v: ti.math.vec3) -> ti.math.vec3:
    qv = ti.math.vec4([v.x, v.y, v.z, 0.0])
    ret4 = quaternion_multiply(
        q, quaternion_multiply(qv, quaternion_conjugate(q)))
    return ti.math.vec3([ret4.x, ret4.y, ret4.z])


@ti.func
def get_point_probability_density_from_2d_gaussian(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    gaussian_covariance: ti.math.mat2,
) -> ti.f32:
    xy_mean = xy - gaussian_mean
    det_cov = gaussian_covariance.determinant()
    inv_cov = (1. / det_cov) * \
        ti.math.mat2([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    xy_mean_T_cov_inv = xy_mean @ inv_cov
    xy_mean_T_cov_inv_xy_mean = xy_mean_T_cov_inv @ xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    return ti.exp(exponent) / (2 * np.pi * ti.sqrt(det_cov))


@ti.func
def get_point_probability_density_from_2d_gaussian_normalized(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    gaussian_covariance: ti.math.mat2,
) -> ti.f32:
    xy_mean = xy - gaussian_mean
    det_cov = gaussian_covariance.determinant()
    inv_cov = (1. / det_cov) * \
        ti.math.mat2([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    xy_mean_T_cov_inv = xy_mean @ inv_cov
    xy_mean_T_cov_inv_xy_mean = xy_mean_T_cov_inv @ xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    return ti.exp(exponent)


@ti.func
def get_point_conic(
    gaussian_covariance: ti.math.mat2,
) -> ti.math.vec3:
    det_cov = gaussian_covariance.determinant()
    inv_cov = (1. / det_cov) * \
        ti.math.mat2([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    conic = ti.math.vec3([inv_cov[0, 0], inv_cov[0, 1], inv_cov[1, 1]])
    return conic


@ti.func
def get_point_probability_density_from_conic(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    conic: ti.math.vec3,
) -> ti.f32:
    xy_mean = xy - gaussian_mean
    exponent = -0.5 * (xy_mean.x * xy_mean.x * conic.x + xy_mean.y * xy_mean.y * conic.z) \
        - xy_mean.x * xy_mean.y * conic.y
    return ti.exp(exponent)


@ti.func
def grad_point_probability_density_2d(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    gaussian_covariance: ti.math.mat2,
):
    xy_mean = xy - gaussian_mean
    det_cov = gaussian_covariance.determinant()
    inv_cov = (1. / det_cov) * \
        ti.math.mat2([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    cov_inv_xy_mean = inv_cov @ xy_mean
    xy_mean_T_cov_inv_xy_mean = xy_mean @ cov_inv_xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    p = ti.exp(exponent) / (2 * np.pi * ti.sqrt(det_cov))
    d_p_d_mean = p * cov_inv_xy_mean
    xy_mean_outer_xy_mean = xy_mean.outer_product(xy_mean)
    d_p_d_cov = -0.5 * p * (inv_cov - inv_cov @
                            xy_mean_outer_xy_mean @ inv_cov)
    return p, d_p_d_mean, d_p_d_cov


@ti.func
def grad_point_probability_density_2d_normalized(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    gaussian_covariance: ti.math.mat2,
):
    xy_mean = xy - gaussian_mean
    det_cov = gaussian_covariance.determinant()
    inv_cov = (1. / det_cov) * \
        ti.math.mat2([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    cov_inv_xy_mean = inv_cov @ xy_mean
    xy_mean_T_cov_inv_xy_mean = xy_mean @ cov_inv_xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    p = ti.exp(exponent)
    d_p_d_mean = p * cov_inv_xy_mean
    xy_mean_outer_xy_mean = xy_mean.outer_product(xy_mean)
    d_p_d_cov = 0.5 * p * (inv_cov @
                           xy_mean_outer_xy_mean @ inv_cov)
    return p, d_p_d_mean, d_p_d_cov


@ti.func
def grad_point_probability_density_from_conic(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    conic: ti.math.vec3,
):
    xy_mean = xy - gaussian_mean
    inv_cov = ti.math.mat2([[conic.x, conic.y], [conic.y, conic.z]])
    cov_inv_xy_mean = inv_cov @ xy_mean
    xy_mean_T_cov_inv_xy_mean = xy_mean @ cov_inv_xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    p = ti.exp(exponent)
    d_p_d_mean = p * cov_inv_xy_mean
    xy_mean_outer_xy_mean = xy_mean.outer_product(xy_mean)
    d_p_d_cov = 0.5 * p * (inv_cov @
                           xy_mean_outer_xy_mean @ inv_cov)
    return p, d_p_d_mean, d_p_d_cov


@ti.func
def ti_sigmoid(x):
    return 1 / (1 + ti.exp(-x))


@ti.func
def ti_sigmoid_with_jacobian(x):
    s = ti_sigmoid(x)
    return s, s * (1 - s)


@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]


@ti.kernel
def ti2torch(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = field[I]


@ti.kernel
def ti2torch_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        grad[I] = field.grad[I]


@ti.kernel
def torch2ti_grad(field: ti.template(), grad: ti.types.ndarray()):
    for I in ti.grouped(grad):
        field.grad[I] = grad[I]


def inverse_SE3(transform: torch.Tensor):
    R = transform[:3, :3]
    t = transform[:3, 3]
    inverse_transform = torch.zeros_like(transform)
    inverse_transform[:3, :3] = R.T
    inverse_transform[:3, 3] = -R.T @ t
    inverse_transform[3, 3] = 1
    return inverse_transform


def quaternion_conjugate_torch(
    q: torch.Tensor,  # (batch_size, 4), (x, y, z, w)
):
    return torch.cat([-q[..., 0:3], q[..., 3:4]], dim=-1)


def quaternion_multiply_torch(
    q0: torch.Tensor,  # (batch_size, 4)
    q1: torch.Tensor,  # (batch_size, 4)
):
    x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    return torch.stack([x, y, z, w], dim=-1)


def quaternion_rotate_torch(
    q: torch.Tensor,  # (batch_size, 4)
    v: torch.Tensor,  # (batch_size, 3)
):
    q = q / torch.norm(q, dim=-1, keepdim=True)
    v = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
    q_conj = quaternion_conjugate_torch(q)
    return quaternion_multiply_torch(
        quaternion_multiply_torch(q, v), q_conj)[..., :3]


def inverse_SE3_qt_torch(
    q: torch.Tensor,  # (batch_size, 4)
    t: torch.Tensor,  # (batch_size, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_inv = quaternion_conjugate_torch(q)
    t_inv = -quaternion_rotate_torch(q_inv, t)
    return q_inv, t_inv


def rotation_matrix_to_quaternion_torch(
    R: torch.Tensor  # (batch_size, 3, 3)
) -> torch.Tensor:
    q = torch.zeros(R.shape[0], 4, device=R.device,
                    dtype=R.dtype)  # (batch_size, 4) x, y, z, w
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q0_mask = trace > 0
    q1_mask = (R[..., 0, 0] > R[..., 1, 1]) & (
        R[..., 0, 0] > R[..., 2, 2]) & ~q0_mask
    q2_mask = (R[..., 1, 1] > R[..., 2, 2]) & ~q0_mask & ~q1_mask
    q3_mask = ~q0_mask & ~q1_mask & ~q2_mask
    if q0_mask.any():
        R_for_q0 = R[q0_mask]
        S_for_q0 = 0.5 / torch.sqrt(1 + trace[q0_mask])
        q[q0_mask, 3] = 0.25 / S_for_q0
        q[q0_mask, 0] = (R_for_q0[..., 2, 1] - R_for_q0[..., 1, 2]) * S_for_q0
        q[q0_mask, 1] = (R_for_q0[..., 0, 2] - R_for_q0[..., 2, 0]) * S_for_q0
        q[q0_mask, 2] = (R_for_q0[..., 1, 0] - R_for_q0[..., 0, 1]) * S_for_q0

    if q1_mask.any():
        R_for_q1 = R[q1_mask]
        S_for_q1 = 2.0 * \
            torch.sqrt(1 + R_for_q1[..., 0, 0] -
                       R_for_q1[..., 1, 1] - R_for_q1[..., 2, 2])
        q[q1_mask, 0] = 0.25 * S_for_q1
        q[q1_mask, 1] = (R_for_q1[..., 0, 1] + R_for_q1[..., 1, 0]) / S_for_q1
        q[q1_mask, 2] = (R_for_q1[..., 0, 2] + R_for_q1[..., 2, 0]) / S_for_q1
        q[q1_mask, 3] = (R_for_q1[..., 2, 1] - R_for_q1[..., 1, 2]) / S_for_q1

    if q2_mask.any():
        R_for_q2 = R[q2_mask]
        S_for_q2 = 2.0 * \
            torch.sqrt(1 + R_for_q2[..., 1, 1] -
                       R_for_q2[..., 0, 0] - R_for_q2[..., 2, 2])
        q[q2_mask, 0] = (R_for_q2[..., 0, 1] + R_for_q2[..., 1, 0]) / S_for_q2
        q[q2_mask, 1] = 0.25 * S_for_q2
        q[q2_mask, 2] = (R_for_q2[..., 1, 2] + R_for_q2[..., 2, 1]) / S_for_q2
        q[q2_mask, 3] = (R_for_q2[..., 0, 2] - R_for_q2[..., 2, 0]) / S_for_q2

    if q3_mask.any():
        R_for_q3 = R[q3_mask]
        S_for_q3 = 2.0 * \
            torch.sqrt(1 + R_for_q3[..., 2, 2] -
                       R_for_q3[..., 0, 0] - R_for_q3[..., 1, 1])
        q[q3_mask, 0] = (R_for_q3[..., 0, 2] + R_for_q3[..., 2, 0]) / S_for_q3
        q[q3_mask, 1] = (R_for_q3[..., 1, 2] + R_for_q3[..., 2, 1]) / S_for_q3
        q[q3_mask, 2] = 0.25 * S_for_q3
        q[q3_mask, 3] = (R_for_q3[..., 1, 0] - R_for_q3[..., 0, 1]) / S_for_q3
    return q


def SE3_to_quaternion_and_translation_torch(
    transform: torch.Tensor,  # (batch_size, 4, 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    q = rotation_matrix_to_quaternion_torch(R)
    return q, t


@ti.func
def taichi_inverse_SE3(transform: ti.math.mat4):
    R_T = ti.math.mat3([
        [transform[0, 0], transform[1, 0], transform[2, 0]],
        [transform[0, 1], transform[1, 1], transform[2, 1]],
        [transform[0, 2], transform[1, 2], transform[2, 2]]
    ])
    t = ti.math.vec3([transform[0, 3], transform[1, 3], transform[2, 3]])
    inverse_transform_t = -R_T @ t
    inverse_transform = ti.math.mat4([
        [R_T[0, 0], R_T[0, 1], R_T[0, 2], inverse_transform_t[0]],
        [R_T[1, 0], R_T[1, 1], R_T[1, 2], inverse_transform_t[1]],
        [R_T[2, 0], R_T[2, 1], R_T[2, 2], inverse_transform_t[2]],
        [0, 0, 0, 1]
    ])
    return inverse_transform


def torch_single_point_alpha_forward(
    point_xyz: torch.Tensor,  # (3,)
    point_q: torch.Tensor,  # (4,)
    point_s: torch.Tensor,  # (3,)
    T_camera_pointcloud: torch.Tensor,  # (4, 4)
    camera_intrinsics: torch.Tensor,  # (3, 3)
    point_alpha: torch.Tensor,  # (1,)
    pixel_uv: torch.Tensor,  # (1,)
):
    xyz1 = torch.cat(
        [point_xyz, torch.ones_like(point_xyz[:1])], dim=0)  # (4,)
    xyz1_camera = T_camera_pointcloud @ xyz1  # (4,)
    xyz_camera = xyz1_camera[:3]
    xyz_camera.register_hook(lambda grad: print(
        f"torch xyz_camera grad: {grad}"))
    uv1 = camera_intrinsics @ xyz_camera
    uv = uv1[:2] / uv1[2]
    print(uv)
    # add hook to get the gradient of uv
    uv.register_hook(lambda grad: print(f"torch uv grad: {grad}"))
    exp_s = torch.exp(point_s)  # (3,)
    S = torch.diag(exp_s)  # (3, 3)
    R = quaternion_to_rotation_matrix_torch(point_q)  # (3, 3)
    Sigma = R @ S @ S @ R.T  # (3, 3)
    print(Sigma)
    z_camrea = xyz1_camera[2]
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    J = torch.tensor([
        fx / z_camrea, 0, -fx * xyz1_camera[0] / (z_camrea * z_camrea),
        0, fy / z_camrea, -fy * xyz1_camera[1] / (z_camrea * z_camrea)
    ]).reshape(2, 3)  # (2, 3)
    W = T_camera_pointcloud[:3, :3]  # (3, 3)
    cov = J @ W @ Sigma @ W.T @ J.T  # (2, 2)
    print(cov)
    cov.register_hook(lambda grad: print(f"torch cov grad: {grad}"))
    # for 2d gaussian center at uv with covariance cov, the probability density of pixel_uv is:
    det_cov = cov.det()
    inv_cov = torch.inverse(cov)
    pixel_uv_center = pixel_uv.float() + 0.5
    p = torch.exp(-0.5 * (pixel_uv_center - uv).T @ inv_cov @
                  (pixel_uv_center - uv))  # (1,)
    print("torch p: ", p)
    print("torch point_alpha: ", point_alpha)
    alpha = torch.sigmoid(point_alpha) * p  # (1,)
    return alpha


def torch_single_point_forward(
    point_xyz: torch.Tensor,  # (3,)
    point_q: torch.Tensor,  # (4,)
    point_s: torch.Tensor,  # (3,)
    T_camera_pointcloud: torch.Tensor,  # (4, 4)
    camera_intrinsics: torch.Tensor,  # (3, 3)
    ray_origin: torch.Tensor,  # (3,)
    ray_direction: torch.Tensor,  # (3,)
    point_alpha: torch.Tensor,  # (1,)
    color_r_sh: torch.Tensor,  # (16,)
    color_g_sh: torch.Tensor,  # (16,)
    color_b_sh: torch.Tensor,  # (16,)
    pixel_uv: torch.Tensor,  # (1,)
    accumuated_alpha: float,
):
    alpha = torch_single_point_alpha_forward(
        point_xyz=point_xyz,
        point_q=point_q,
        point_s=point_s,
        T_camera_pointcloud=T_camera_pointcloud,
        camera_intrinsics=camera_intrinsics,
        point_alpha=point_alpha,
        pixel_uv=pixel_uv,
    )
    color_r = alpha * \
        get_spherical_harmonic_from_xyz_torch(ray_direction) @ color_r_sh
    color_g = alpha * \
        get_spherical_harmonic_from_xyz_torch(ray_direction) @ color_g_sh
    color_b = alpha * \
        get_spherical_harmonic_from_xyz_torch(ray_direction) @ color_b_sh
    rgb = torch.stack([color_r, color_g, color_b], dim=0)
    pixel_rgb = alpha * (1 - accumuated_alpha) * rgb
    return pixel_rgb


def quaternion_to_rotation_matrix_torch(q):
    """
    Convert a quaternion into a full three-dimensional rotation matrix.

    Input:
    :param q: A tensor of size (B, 4), where B is batch size and quaternion is in format (x, y, z, w).

    Output:
    :return: A tensor of size (B, 3, 3), where B is batch size.
    """
    # Ensure quaternion has four components
    assert q.shape[-1] == 4, "Input quaternion should have 4 components!"

    x, y, z, w = q.unbind(-1)

    # Compute quaternion norms
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    # Normalize input quaternions
    q = q / q_norm

    # Compute the quaternion outer product
    q_outer = torch.einsum('...i,...j->...ij', q, q)

    # Compute rotation matrix
    rot_matrix = torch.empty(
        (*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    rot_matrix[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrix[..., 0, 1] = 2 * (x*y - z*w)
    rot_matrix[..., 0, 2] = 2 * (x*z + y*w)
    rot_matrix[..., 1, 0] = 2 * (x*y + z*w)
    rot_matrix[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrix[..., 1, 2] = 2 * (y*z - x*w)
    rot_matrix[..., 2, 0] = 2 * (x*z - y*w)
    rot_matrix[..., 2, 1] = 2 * (y*z + x*w)
    rot_matrix[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot_matrix


def get_spherical_harmonic_from_xyz_torch(
    xyz: torch.Tensor  # (3,)
):
    xyz /= torch.norm(xyz)
    x, y, z = xyz[0], xyz[1], xyz[2]
    l0m0 = 0.28209479177387814
    l1m1 = -0.48860251190291987 * y
    l1m0 = 0.48860251190291987 * z
    l1p1 = -0.48860251190291987 * x
    l2m2 = 1.0925484305920792 * x * y
    l2m1 = -1.0925484305920792 * y * z
    l2m0 = 0.94617469575755997 * z * z - 0.31539156525251999
    l2p1 = -1.0925484305920792 * x * z
    l2p2 = 0.54627421529603959 * x * x - 0.54627421529603959 * y * y
    l3m3 = 0.59004358992664352 * y * (-3.0 * x * x + y * y)
    l3m2 = 2.8906114426405538 * x * y * z
    l3m1 = 0.45704579946446572 * y * (1.0 - 5.0 * z * z)
    l3m0 = 0.3731763325901154 * z * (5.0 * z * z - 3.0)
    l3p1 = 0.45704579946446572 * x * (1.0 - 5.0 * z * z)
    l3p2 = 1.4453057213202769 * z * (x * x - y * y)
    l3p3 = 0.59004358992664352 * x * (-x * x + 3.0 * y * y)
    return torch.tensor([
        l0m0, l1m1, l1m0, l1p1, l2m2, l2m1, l2m0, l2p1, l2p2, l3m3, l3m2, l3m1, l3m0, l3p1, l3p2, l3p3])
