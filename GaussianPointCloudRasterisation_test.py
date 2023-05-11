import numpy as np
import unittest
import taichi as ti
import torch
from GaussianPointCloudRasterisation import (
    find_tile_start_and_end, load_point_cloud_row_into_gaussian_point_3d, GaussianPointCloudRasterisation)
from GaussianPoint3D import GaussianPoint3D, mat2x3f
from SphericalHarmonics import SphericalHarmonics
from utils import grad_point_probability_density_2d, torch_single_point_alpha_forward, torch_single_point_forward, get_point_probability_density_from_2d_gaussian
from Camera import CameraInfo
from tqdm import tqdm


class TestFindTileStartAndEnd(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(ti.gpu, debug=True)

    def test_find_tile_start_and_end(self):
        point_in_camera_sort_key = torch.tensor(
            [
                0x100000000, 0x100000001, 0x200000000, 0x200000001, 0x200000002,
                0x300000000, 0x300000001
            ],
            dtype=torch.int64,
            device=torch.device("cuda:0")
        )

        tile_points_start = torch.zeros(
            4, dtype=torch.int32, device=point_in_camera_sort_key.device)
        tile_points_end = torch.zeros(
            4, dtype=torch.int32, device=point_in_camera_sort_key.device)

        find_tile_start_and_end(
            point_in_camera_sort_key,
            tile_points_start,
            tile_points_end
        )

        tile_points_start_expected = torch.tensor(
            [0, 0, 2, 5], dtype=torch.int32, device=point_in_camera_sort_key.device)
        tile_points_end_expected = torch.tensor(
            [0, 2, 5, 7], dtype=torch.int32, device=point_in_camera_sort_key.device)

        self.assertTrue(
            torch.all(tile_points_start == tile_points_start_expected),
            f"Expected: {tile_points_start_expected}, Actual: {tile_points_start}"
        )
        self.assertTrue(
            torch.all(tile_points_end == tile_points_end_expected),
            f"Expected: {tile_points_end_expected}, Actual: {tile_points_end}"
        )


class TestLoadPointCloudRowIntoGaussianPoint3D(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(ti.gpu, debug=True)

    def test_load_point_cloud_row_into_gaussian_point_3d(self):
        N = 5
        M = 56
        # pointcloud = np.random.rand(N, 3).astype(np.float32)
        # pointcloud_features = np.random.rand(N, M).astype(np.float32)
        pointcloud = torch.rand(N, 3, dtype=torch.float32,
                                device=torch.device("cuda:0"))
        pointcloud_features = torch.rand(
            N, M, dtype=torch.float32, device=torch.device("cuda:0"))
        point_id = 2

        @ti.kernel
        def test_kernel(
            pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (N, 3)
            pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (N, M)
            point_id: ti.i32,
        ) -> GaussianPoint3D:
            return load_point_cloud_row_into_gaussian_point_3d(
                pointcloud=pointcloud,
                pointcloud_features=pointcloud_features,
                point_id=point_id
            )

        result = test_kernel(pointcloud, pointcloud_features, point_id)

        expected = GaussianPoint3D(
            translation=pointcloud[point_id],
            cov_rotation=pointcloud_features[point_id, :4],
            cov_scale=pointcloud_features[point_id, 4:7],
            alpha=pointcloud_features[point_id, 7],
            color_r=pointcloud_features[point_id, 8:24],
            color_g=pointcloud_features[point_id, 24:40],
            color_b=pointcloud_features[point_id, 40:56],
        )

        self.assertTrue(
            np.allclose(result.translation, expected.translation),
            f"Expected: {expected.translation}, Actual: {result.translation}",
        )
        self.assertTrue(
            np.allclose(result.cov_rotation, expected.cov_rotation),
            f"Expected: {expected.cov_rotation}, Actual: {result.cov_rotation}",
        )
        self.assertTrue(
            np.allclose(result.cov_scale, expected.cov_scale),
            f"Expected: {expected.cov_scale}, Actual: {result.cov_scale}",
        )


class TestRasterisation(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(ti.gpu, device_memory_GB=20)

    def test_rasterisation_basic(self):
        # GaussianPointCloudRasterisation
        gaussian_point_cloud_rasterisation = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig())
        num_points = 100000

        for idx in tqdm(range(100)):
            point_cloud = torch.rand(
                size=(num_points, 3), dtype=torch.float32, device=torch.device("cuda:0"), requires_grad=True)
            point_cloud_features = torch.rand(
                size=(num_points, 56), dtype=torch.float32, device=torch.device("cuda:0"), requires_grad=True)
            camera_info = CameraInfo(
                camera_height=1088,
                camera_width=1920,
                camera_id=0,
                camera_intrinsics=torch.tensor([[500, 0, 960], [0, 500, 540], [
                    0, 0, 1]], dtype=torch.float32, device=torch.device("cuda:0")),
            )
            T_pointcloud_to_camera = torch.eye(
                4, dtype=torch.float32, device=torch.device("cuda:0"))
            T_pointcloud_to_camera[2, 3] = -0.5
            input_data = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=point_cloud,
                point_cloud_features=point_cloud_features,
                camera_info=camera_info,
                T_pointcloud_camera=T_pointcloud_to_camera)
            image = gaussian_point_cloud_rasterisation(input_data)
            loss = image.sum()
            loss.backward()

    def test_backward_coverage(self):
        """ use a fake image, and test if gradient descent can have a good coverage
        """
        image_size = (32, 32)
        num_points = 10000
        fake_image = torch.zeros(
            size=(image_size[0], image_size[1], 3), dtype=torch.float32, device=torch.device("cuda:0"))
        fake_image[:5, :2, 0] = 1.0
        fake_image[:5, :2, 1] = 0.7
        fake_image[8:24, 8:24, 0] = 0.5
        fake_image[8:24, 8:24, 1] = 0.2
        fake_image[8:24, 8:24, 1] = 0.7
        fake_image[20:28, 20:28, 0] = 0.8
        fake_image[20:28, 20:28, 1] = 0.1
        fake_image[20:28, 20:28, 1] = 0.1
        point_cloud = torch.nn.Parameter((torch.rand(size=(num_points, 3), dtype=torch.float32, device=torch.device(
            "cuda:0")) - 0.5) * 3)
        tmp = torch.rand(size=(
            num_points, 56), dtype=torch.float32, device=torch.device("cuda:0"))
        tmp[:, 4:7] = 0.01
        tmp[:, 7] = 0.5
        point_cloud_features = torch.nn.Parameter(tmp)
        camera_info = CameraInfo(
            camera_height=image_size[0],
            camera_width=image_size[1],
            camera_id=0,
            camera_intrinsics=torch.tensor([[32, 0, 16], [0, 32, 16], [
                                           0, 0, 1]], dtype=torch.float32, device=torch.device("cuda:0")),
        )
        T_camera_world = torch.eye(
            4, dtype=torch.float32, device=torch.device("cuda:0"))
        T_camera_world[2, 3] = -2
        gaussian_point_cloud_rasterisation = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=1.,
                far_plane=10.
            ))
        optimizer = torch.optim.Adam(
            [point_cloud, point_cloud_features], lr=0.001)
        intital_loss = None
        latest_loss = None
        for idx in tqdm(range(10000)):
            optimizer.zero_grad()
            input_data = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=point_cloud,
                point_cloud_features=point_cloud_features,
                camera_info=camera_info,
                T_pointcloud_camera=T_camera_world,
                color_l0_only=True)
            pred_image = gaussian_point_cloud_rasterisation(input_data)
            loss = ((pred_image - fake_image)**2).sum()
            loss.backward()
            optimizer.step()
            print(f"loss: {loss.item()}")
            if idx == 0:
                initial_loss = loss.item()
            latest_loss = loss.item()
        self.assertTrue(latest_loss < initial_loss)

    def test_single_point(self):
        T_camera_pointcloud = torch.tensor([[1., 0., 0., 0.],
                                            [0., 1., 0., 0.],
                                            [0., 0., 1., 2.],
                                            [0., 0., 0., 1.]])
        camera_intrinsics = torch.tensor([[32.,  0., 16.],
                                          [0., 32., 16.],
                                          [0.,  0.,  1.]])
        direction = torch.tensor([-0.3906, -0.3906,  5.0000])
        origin = torch.tensor([0., 0., -2.])
        xyz = torch.tensor([-0.4325, -0.7224, -0.4733],
                           dtype=torch.float32, requires_grad=True)
        features = torch.tensor([0.0115,  0.5507,  0.6920,  0.4666,  0.6306,  0.0871, -0.0112,  1.7667,
                                2.2963,  0.1560,  0.8710,  0.3418,  0.3658,  0.1913,  0.8727,  0.3608,
                                0.6874,  0.7516,  0.9281,  0.5649,  0.9469,  0.9090,  0.7356,  0.5436,
                                1.7886,  0.7542,  0.9568,  0.2868,  0.3552,  0.3872,  0.0827,  0.4101,
                                0.7783,  0.6266,  0.9601,  0.8252,  0.7846,  0.0183,  0.6635,  0.4688,
                                -1.4012,  0.1584,  0.3252,  0.5403,  0.4992,  0.2780,  0.7412,  0.5056,
                                0.8236,  0.9722,  0.5467,  0.6644,  0.2583,  0.0953,  0.3986,  0.2265],
                                dtype=torch.float32, requires_grad=True)
        q = features[:4]
        s = features[4:7]
        point_alpha = features[7]
        r_sh = features[8:24]
        g_sh = features[24:40]
        b_sh = features[40:56]
        pixel_uv = torch.tensor([3, 3])

        torch_alpha = torch_single_point_alpha_forward(
            point_xyz=xyz,
            point_q=q,
            point_s=s,
            T_camera_pointcloud=T_camera_pointcloud,
            camera_intrinsics=camera_intrinsics,
            point_alpha=point_alpha,
            pixel_uv=pixel_uv
        )
        torch_alpha.backward()
        xyz_grad = xyz.grad
        xyz_feature_grad = features.grad

        @ti.kernel
        def single_point_alpha_forward(
            pixel_v: ti.i32,
            pixel_u: ti.i32,
            camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
            T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
            pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (1, 3,)
            pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (1, 56,)
            # (H, W, 3)
        ) -> ti.f32:
            point_id = 0
            camera_intrinsics_mat = ti.Matrix(
                [[camera_intrinsics[row, col] for col in range(3)] for row in range(3)])
            T_camera_pointcloud_mat = ti.Matrix(
                [[T_camera_pointcloud[row, col] for col in range(4)] for row in range(4)])
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
            gaussian_alpha = get_point_probability_density_from_2d_gaussian(
                xy=ti.math.vec2([pixel_u + 0.5, pixel_v + 0.5]),
                gaussian_mean=uv,
                gaussian_covariance=uv_cov,
            )
            alpha = gaussian_alpha * gaussian_point_3d.alpha
            return alpha
        ti_alpha = single_point_alpha_forward(
            pixel_u=pixel_uv[0].item(),
            pixel_v=pixel_uv[1].item(),
            camera_intrinsics=camera_intrinsics,
            T_camera_pointcloud=T_camera_pointcloud,
            pointcloud=xyz.reshape(1, 3),
            pointcloud_features=features.reshape(1, 56)
        )
        print(f"ti_alpha: {ti_alpha}")
        print(f"torch_alpha: {torch_alpha.item()}")
        self.assertTrue(np.allclose(torch_alpha.item(), ti_alpha, atol=1e-4))

        @ti.kernel
        def single_point_alpha_backward(
            alpha_grad: ti.f32,
            pixel_v: ti.i32,
            pixel_u: ti.i32,
            camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2),  # (3, 3)
            T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
            pointcloud: ti.types.ndarray(ti.f32, ndim=2),  # (1, 3,)
            pointcloud_features: ti.types.ndarray(ti.f32, ndim=2),  # (1, 56,)
            pointcloud_grad: ti.types.ndarray(ti.f32, ndim=2),  # (1, 3,)
            # (1, 56,)
            pointcloud_features_grad: ti.types.ndarray(ti.f32, ndim=2),
        ):
            point_id = 0
            camera_intrinsics_mat = ti.Matrix(
                [[camera_intrinsics[row, col] for col in range(3)] for row in range(3)])
            T_camera_pointcloud_mat = ti.Matrix(
                [[T_camera_pointcloud[row, col] for col in range(4)] for row in range(4)])
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
            prod_alpha = gaussian_alpha * gaussian_point_3d.alpha
            ggaussian_point_3d_alpha_grad = alpha_grad * gaussian_alpha
            gaussian_alpha_grad = alpha_grad * gaussian_point_3d.alpha
            # gaussian_alpha_grad is dp
            uv_grad = gaussian_alpha_grad * \
                d_p_d_mean
            print(f"taichi uv_grad: {uv_grad}")
            K = camera_intrinsics_mat
            t = translation_camera
            d_uv_d_translation_camera = mat2x3f([
                [K[0, 0] / t.z, K[0, 1] / t.z,
                    (-K[0, 0] * t.x - K[0, 1] * t.y) / (t.z * t.z)],
                [K[1, 0] / t.z, K[1, 1] / t.z, (-K[1, 0] * t.x - K[1, 1] * t.y) / (t.z * t.z)]])
            translation_grad = gaussian_alpha_grad * \
                d_p_d_mean @ d_uv_d_translation
            print(f"taichi translation_grad: {translation_grad}")
            # cov is Sigma
            gaussian_q_grad = gaussian_alpha_grad * \
                d_p_d_cov_flat @ d_Sigma_prime_d_q
            gaussian_s_grad = gaussian_alpha_grad * \
                d_p_d_cov_flat @ d_Sigma_prime_d_s
            pointcloud_grad[0, 0] = translation_grad[0]
            pointcloud_grad[0, 1] = translation_grad[1]
            pointcloud_grad[0, 2] = translation_grad[2]
            pointcloud_features_grad[0, 0] = gaussian_q_grad[0]
            pointcloud_features_grad[0, 1] = gaussian_q_grad[1]
            pointcloud_features_grad[0, 2] = gaussian_q_grad[2]
            pointcloud_features_grad[0, 3] = gaussian_q_grad[3]
            pointcloud_features_grad[0, 4] = gaussian_s_grad[0]
            pointcloud_features_grad[0, 5] = gaussian_s_grad[1]
            pointcloud_features_grad[0, 6] = gaussian_s_grad[2]
            pointcloud_features_grad[0, 7] = ggaussian_point_3d_alpha_grad

        pointcloud_grad = torch.zeros(1, 3, dtype=torch.float32)
        pointcloud_features_grad = torch.zeros(1, 56, dtype=torch.float32)
        single_point_alpha_backward(
            alpha_grad=1.0,
            pixel_u=pixel_uv[0].item(),
            pixel_v=pixel_uv[1].item(),
            camera_intrinsics=camera_intrinsics,
            T_camera_pointcloud=T_camera_pointcloud,
            pointcloud=xyz.reshape(1, 3),
            pointcloud_features=features.reshape(1, 56),
            pointcloud_grad=pointcloud_grad,
            pointcloud_features_grad=pointcloud_features_grad,
        )
        pointcloud_grad = pointcloud_grad.reshape(3)
        pointcloud_features_grad = pointcloud_features_grad.reshape(56)
        self.assertTrue(np.allclose(
            pointcloud_grad.detach().cpu().numpy(), xyz_grad.detach().cpu().numpy(), atol=1e-4),
            msg=f"Expected: {pointcloud_grad.detach().cpu().numpy()}, Actual: {xyz_grad.detach().cpu().numpy()}")
