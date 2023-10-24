import taichi as ti
import taichi.math as tm
import unittest
from taichi_3d_gaussian_splatting.utils import (intersect_ray_with_ellipsoid, get_ray_origin_and_direction_from_camera,
                                                get_point_probability_density_from_2d_gaussian, grad_point_probability_density_2d,
                                                quaternion_to_rotation_matrix_torch,
                                                get_ray_origin_and_direction_by_uv, inverse_SE3_qt_torch, rotation_matrix_to_quaternion_torch)
from taichi_3d_gaussian_splatting.Camera import CameraInfo
import torch
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation


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

    def test_rotation_matrix_to_quaternion_torch(self):
        q_list = np.random.rand(100, 4)
        q_list = q_list / np.linalg.norm(q_list, axis=1, keepdims=True)
        # to wxyz
        rotations = Rotation.from_quat(q_list)
        matrix_list = rotations.as_matrix()
        q_torch = rotation_matrix_to_quaternion_torch(
            torch.tensor(matrix_list))

        self.assertTrue(np.allclose(q_torch.numpy(), q_list))

    def test_inverse_SE3_qt_torch(self):
        q_list = np.random.rand(100, 4)
        q_list = q_list / np.linalg.norm(q_list, axis=1, keepdims=True)
        # to wxyz
        rotations = Rotation.from_quat(q_list)
        matrix_list = rotations.as_matrix()
        t_list = np.random.rand(100, 3)
        SE3_np = np.eye(4).reshape(1, 4, 4).repeat(100, axis=0)
        SE3_np[:, :3, :3] = matrix_list
        SE3_np[:, :3, 3] = t_list
        q_list_torch = torch.tensor(q_list)
        t_list_torch = torch.tensor(t_list)
        q_inv_torch, t_inv_torch = inverse_SE3_qt_torch(
            q_list_torch, t_list_torch)
        SE3_inv_np = np.linalg.inv(SE3_np)
        SE3_inv_torch = torch.eye(4).reshape(1, 4, 4).repeat(100, 1, 1)
        SE3_inv_torch[:, :3, :3] = quaternion_to_rotation_matrix_torch(
            q_inv_torch)
        SE3_inv_torch[:, :3, 3] = t_inv_torch
        self.assertTrue(np.allclose(SE3_inv_np, SE3_inv_torch.numpy()))


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
        ti.init(ti.gpu)

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

    def test_correstness(self):
        for _ in range(100):
            q = torch.rand(4)
            q = q / torch.norm(q)
            R = torch.tensor(Rotation.from_quat(q).as_matrix())
            t = torch.rand(3)
            T_pointcloud_camera = torch.eye(4)
            T_pointcloud_camera[:3, :3] = R
            T_pointcloud_camera[:3, 3] = t
            camera_info = CameraInfo(
                camera_id=0,
                camera_width=640,
                camera_height=480,
                camera_intrinsics=torch.tensor([
                    [500, 0, 320],
                    [0, 500, 240],
                    [0, 0, 1]
                ], dtype=torch.float32)
            )
            ray_origin, ray_direction = get_ray_origin_and_direction_from_camera(
                T_pointcloud_camera, camera_info)
            self.assertEqual(ray_origin.shape, (3,))
            self.assertEqual(ray_direction.shape, (480, 640, 3))
            rand_depth = torch.rand(480, 640) * 10 + 5
            point_cloud = ray_origin.unsqueeze(0).unsqueeze(0) + \
                ray_direction * rand_depth.unsqueeze(-1)
            point_cloud = point_cloud.reshape(-1, 3)
            T_camera_pointcloud = torch.inverse(T_pointcloud_camera)
            # pointcloud: xyz->xyz1
            homogeneous_point_cloud = torch.cat(
                [point_cloud, torch.ones_like(point_cloud[:, :1])], dim=-1)
            camera_point_cloud = torch.einsum(
                "ij,bj->bi", T_camera_pointcloud, homogeneous_point_cloud)
            camera_point_cloud = camera_point_cloud[:, :3]
            self.assertEqual(camera_point_cloud.shape, (480 * 640, 3))
            uv1 = torch.einsum(
                "ij,bj->bi", camera_info.camera_intrinsics, camera_point_cloud) / camera_point_cloud[:, 2].unsqueeze(-1)

            u = uv1[:, 0].reshape(480, 640)  # col idx + 0.5
            v = uv1[:, 1].reshape(480, 640)  # row idx + 0.5

            u_expected = torch.arange(
                0, 640, dtype=torch.float32).unsqueeze(0).repeat(480, 1) + 0.5
            print(u_expected.shape)
            v_expected = torch.arange(
                0, 480, dtype=torch.float32).unsqueeze(-1).repeat(1, 640) + 0.5
            print(v_expected.shape)
            self.assertTrue(torch.allclose(u, u_expected, atol=1e-2),
                            msg=f"u={u}, u_expected={u_expected}")
            self.assertTrue(torch.allclose(v, v_expected, atol=1e-2),
                            msg=f"v={v}, v_expected={v_expected}")

    def test_get_ray_origin_and_direction_by_uv(self):
        ray_origin = ti.math.vec3.field(shape=())
        ray_direction = ti.math.vec3.field(
            shape=(self.camera_info.camera_height, self.camera_info.camera_width))
        camera_intrinsics_ti = ti.math.mat3.field(shape=())
        camera_intrinsics_ti.from_numpy(
            self.camera_info.camera_intrinsics.numpy())
        T_camera_pointcloud_ti = ti.math.mat4.field(shape=())
        T_camera_pointcloud_ti.from_numpy(
            np.linalg.inv(self.T_pointcloud_camera.numpy()))

        @ti.kernel
        def test_kernel():
            for pixel_v, pixel_u in ray_direction:
                pixel_origin, pixel_direction = get_ray_origin_and_direction_by_uv(
                    pixel_u, pixel_v, camera_intrinsics=camera_intrinsics_ti[None], T_camera_pointcloud=T_camera_pointcloud_ti[None])
                ray_origin[None] = pixel_origin
                ray_direction[pixel_v, pixel_u] = pixel_direction
        test_kernel()
        ray_origin_torch, ray_direction_torch = get_ray_origin_and_direction_from_camera(
            self.T_pointcloud_camera, self.camera_info)
        self.assertEqual(ray_origin[None].to_numpy().shape, (3,))
        self.assertEqual(ray_origin_torch.shape, (3,))
        self.assertEqual(ray_direction.to_numpy(
        ).shape, (self.camera_info.camera_height, self.camera_info.camera_width, 3))
        self.assertEqual(ray_direction_torch.shape,
                         (self.camera_info.camera_height, self.camera_info.camera_width, 3))
        self.assertTrue(np.allclose(
            ray_origin[None].to_numpy(), ray_origin_torch.numpy(), atol=1e-2),
            msg=f"ti={ray_origin[None].to_numpy()}, torch={ray_origin_torch.numpy()}")
        self.assertTrue(np.allclose(ray_direction.to_numpy(),
                        ray_direction_torch.numpy(),
                        atol=1e-2),
                        msg=f"ti={ray_direction.to_numpy()}, torch={ray_direction_torch.numpy()}")


class Test2DGaussianPDF(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(ti.cpu, debug=True)

    def test_get_point_probability_density_from_2d_gaussian(self):
        xy = np.array([3.0, 4.0], dtype=np.float32)
        gaussian_mean = np.array([2.0, 3.0], dtype=np.float32)
        gaussian_covariance = np.array(
            [[1.0, 0.3],
             [0.3, 1.5]],
            dtype=np.float32
        )
        xy_ti = tm.vec2.field(shape=())
        xy_ti.from_numpy(xy)
        gaussian_mean_ti = tm.vec2.field(shape=())
        gaussian_mean_ti.from_numpy(gaussian_mean)
        gaussian_covariance_ti = tm.mat2.field(shape=())
        gaussian_covariance_ti.from_numpy(gaussian_covariance)

        @ti.kernel
        def test_kernel() -> ti.f32:
            ti_result = get_point_probability_density_from_2d_gaussian(
                xy_ti[None], gaussian_mean_ti[None], gaussian_covariance_ti[None])
            return ti_result
        ti_result = test_kernel()

        numpy_result = multivariate_normal.pdf(
            xy, mean=gaussian_mean, cov=gaussian_covariance)

        self.assertAlmostEqual(ti_result, numpy_result, places=5,
                               msg=f"Expected: {numpy_result}, Actual: {ti_result}"
                               )

        def gradient_mean(x, mean, cov):
            inv_cov = np.linalg.inv(cov)
            diff = x - mean
            pdf = multivariate_normal.pdf(x, mean=mean, cov=cov)
            d_pdf_d_mean = pdf * (inv_cov @ diff)
            return d_pdf_d_mean

        def gradient_cov(x, mean, cov):
            inv_cov = np.linalg.inv(cov)
            diff = x - mean
            diff_outer = np.outer(diff, diff)
            pdf = multivariate_normal.pdf(x, mean=mean, cov=cov)

            gradient = -0.5 * pdf * (inv_cov - inv_cov @ diff_outer @ inv_cov)
            return gradient

        ti_d_pdf_d_mean = tm.vec2.field(shape=())
        ti_d_pdf_d_cov = tm.mat2.field(shape=())

        @ti.kernel
        def test_gradient_kernel():
            _, ti_d_pdf_d_mean[None], ti_d_pdf_d_cov[None] = grad_point_probability_density_2d(
                xy_ti[None], gaussian_mean_ti[None], gaussian_covariance_ti[None])
        test_gradient_kernel()
        np_d_pdf_d_mean = gradient_mean(xy, gaussian_mean, gaussian_covariance)
        np_d_pdf_d_cov = gradient_cov(xy, gaussian_mean, gaussian_covariance)
        self.assertTrue(np.allclose(ti_d_pdf_d_mean[None].to_numpy(), np_d_pdf_d_mean, atol=1e-5),
                        msg=f"Expected: {np_d_pdf_d_mean}, Actual: {ti_d_pdf_d_mean[None].to_numpy()}")
        self.assertTrue(np.allclose(ti_d_pdf_d_cov[None].to_numpy(), np_d_pdf_d_cov, atol=1e-5),
                        msg=f"Expected: {np_d_pdf_d_cov}, Actual: {ti_d_pdf_d_cov[None].to_numpy()}")
