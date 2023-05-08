import taichi as ti
import taichi.math as tm
import unittest
from utils import intersect_ray_with_ellipsoid, get_ray_origin_and_direction_from_camera
from Camera import CameraInfo
import torch
import numpy as np


def intersect_ray_ellipsoid_np(o, d, S, R, t, eps=1e-9):
    S_matrix = np.diag(S)

    inv_transform_matrix = np.linalg.inv(R @ S_matrix)

    o_transformed = np.dot(inv_transform_matrix, o - t)
    d_transformed = np.dot(inv_transform_matrix, d)

    A = np.dot(d_transformed, d_transformed)
    if np.abs(A) < eps:
        A = eps

    B = 2 * np.dot(o_transformed, d_transformed)
    C = np.dot(o_transformed, o_transformed) - 1

    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        return False, None
    elif np.abs(discriminant) < eps:
        discriminant = 0

    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-B - sqrt_discriminant) / (2 * A)
    t2 = (-B + sqrt_discriminant) / (2 * A)

    if t1 < 0 and t2 < 0:
        return False, None

    t_intersect = t1 if t1 >= 0 else t2
    if np.abs(t1 - t2) < eps:
        t_intersect = min(t1, t2)

    intersection_point_transformed = o_transformed + t_intersect * d_transformed
    intersection_point = np.dot(
        R @ S_matrix, intersection_point_transformed) + t

    return True, intersection_point


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(ti.cpu, debug=True)

    def test_intersect_ray_with_ellipsoid(self):
        o_ti = tm.vec3.field(shape=())
        d_ti = tm.vec3.field(shape=())
        S_ti = tm.vec3.field(shape=())
        R_ti = tm.mat3.field(shape=())
        t_ti = tm.vec3.field(shape=())
        has_intersection_ti = ti.field(dtype=ti.i32, shape=())
        intersection_point_ti = tm.vec3.field(shape=())

        @ti.kernel
        def run_intersect_ray_with_ellipsoid():
            has_intersection_ti[None], intersection_point_ti[None] = intersect_ray_with_ellipsoid(
                ray_origin=o_ti[None],
                ray_direction=d_ti[None],
                ellipsoid_R=R_ti[None],
                ellipsoid_S=S_ti[None],
                ellipsoid_t=t_ti[None],
                eps=eps
            )
        total_has_intersection = 0
        total_has_intersection_np = 0
        for i in range(10000):
            o = np.random.rand(3)
            d = np.random.rand(3)
            S = np.random.rand(3)
            # R = np.random.rand(3, 3)
            # R is SO(3)
            q = np.random.rand(4)
            q = q / np.linalg.norm(q)
            R = np.array([
                [1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] *
                                                           q[2] - q[3] * q[0]), 2 * (q[1] * q[3] + q[2] * q[0])],
                [2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[1] *
                                                           q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[1] * q[0])],
                [2 * (q[1] * q[3] - q[2] * q[0]), 2 * (q[2] * q[3] +
                                                       q[1] * q[0]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])]
            ])
            t = np.random.rand(3)
            eps = 1e-9

            o_ti[None] = o
            d_ti[None] = d
            S_ti[None] = S
            t_ti[None] = t
            R_ti[None] = tm.mat3(R)

            run_intersect_ray_with_ellipsoid()
            has_intersection_np, intersection_point_np = intersect_ray_ellipsoid_np(
                o, d, S, R, t, eps
            )
            has_intersection = has_intersection_ti[None]
            intersection_point = intersection_point_ti[None]

            total_has_intersection += has_intersection if has_intersection else 0
            total_has_intersection_np += has_intersection_np if has_intersection_np else 0
            if has_intersection and has_intersection_np:
                self.assertTrue(
                    np.allclose(intersection_point,
                                intersection_point_np, atol=1e-3),
                    msg=f"o={o}, d={d}, S={S}, R={R}, t={t}, eps={eps}, intersection_point={intersection_point}, intersection_point_np={intersection_point_np}"
                )
        has_intersection_ratio = total_has_intersection / 10000
        total_has_intersection_np_ratio = total_has_intersection_np / 10000
        self.assertTrue(
            np.allclose(has_intersection_ratio,
                        total_has_intersection_np_ratio, atol=1e-2),
            msg=f"has_intersection_ratio={has_intersection_ratio}, total_has_intersection_np_ratio={total_has_intersection_np_ratio}"
        )


class TestGetRayOriginAndDirectionFromCamera(unittest.TestCase):

    def setUp(self):
        self.camera_info = CameraInfo(
            camera_id=0,
            camera_width=640,
            camera_height=480,
            camera_intrinsics=torch.tensor([
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]
            ], dtype=torch.float32)
        )
        self.T_pointcloud_camera = torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

    def test_ray_origin_shape(self):
        ray_origin, _ = get_ray_origin_and_direction_from_camera(
            self.T_pointcloud_camera, self.camera_info)
        self.assertEqual(ray_origin.shape, (3,))

    def test_direction_shape(self):
        _, direction = get_ray_origin_and_direction_from_camera(
            self.T_pointcloud_camera, self.camera_info)
        self.assertEqual(
            direction.shape, (self.camera_info.camera_height, self.camera_info.camera_width, 3))

    def test_ray_origin_values(self):
        ray_origin, _ = get_ray_origin_and_direction_from_camera(
            self.T_pointcloud_camera, self.camera_info)
        expected_ray_origin = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(ray_origin, expected_ray_origin))
