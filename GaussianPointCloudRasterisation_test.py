import unittest
import taichi as ti
import torch
from GaussianPointCloudRasterisation import find_tile_start_and_end


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
