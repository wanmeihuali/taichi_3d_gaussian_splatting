import numpy as np
import unittest
import taichi as ti
import torch
from GaussianPointCloudRasterisation import (
    find_tile_start_and_end, load_point_cloud_row_into_gaussian_point_3d, GaussianPointCloudRasterisation)
from GaussianPoint3D import GaussianPoint3D
from SphericalHarmonics import SphericalHarmonics
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
            T_pointcloud_to_camera[2, 3] = -2
            input_data = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=point_cloud,
                point_cloud_features=point_cloud_features,
                camera_info=camera_info,
                T_pointcloud_camera=T_pointcloud_to_camera)
            image = gaussian_point_cloud_rasterisation(input_data)
            loss = image.sum()
            loss.backward()
