import unittest
from taichi_3d_gaussian_splatting.GaussianPointAdaptiveController import GaussianPointAdaptiveController
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from tqdm import tqdm
import torch
import taichi as ti


class GaussianPointAdaptiveControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        ti.init(arch=ti.cuda)

    def test_basic(self):
        """
        ensure the code can run.
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
        point_invalid_mask = torch.zeros((num_points,), dtype=torch.int8, device=torch.device(
            "cuda:0"))
        point_invalid_mask[1000:] = 1
        tmp = torch.rand(size=(
            num_points, 56), dtype=torch.float32, device=torch.device("cuda:0"))
        tmp[:, 4:7] = -4.60517018599
        tmp[:, 7] = 0.5
        point_cloud_features = torch.nn.Parameter(tmp)
        point_object_id = torch.zeros((num_points,), dtype=torch.int32, device=torch.device("cuda:0"))
        camera_info = CameraInfo(
            camera_height=image_size[0],
            camera_width=image_size[1],
            camera_id=0,
            camera_intrinsics=torch.tensor([[32, 0, 16], [0, 32, 16], [
                                           0, 0, 1]], dtype=torch.float32, device=torch.device("cuda:0")),
        )
        """
        T_camera_world = torch.eye(
            4, dtype=torch.float32, device=torch.device("cuda:0"))
        T_camera_world[2, 3] = -2
        """
        q_camera_world = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0)
        t_camera_world = torch.tensor([0, 0, -2], dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0)
        gaussian_point_adaptive_controller = GaussianPointAdaptiveController(
            config=GaussianPointAdaptiveController.GaussianPointAdaptiveControllerConfig(),
            maintained_parameters=GaussianPointAdaptiveController.GaussianPointAdaptiveControllerMaintainedParameters(
                pointcloud=point_cloud,
                pointcloud_features=point_cloud_features,
                point_invalid_mask=point_invalid_mask,
                point_object_id=point_object_id,
            ))
        gaussian_point_cloud_rasterisation = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=1.,
                far_plane=10.
            ),
            backward_valid_point_hook=gaussian_point_adaptive_controller.update)
        optimizer = torch.optim.Adam(
            [point_cloud, point_cloud_features], lr=0.001)
        intital_loss = None
        latest_loss = None
        for idx in tqdm(range(10000)):
            optimizer.zero_grad()
            input_data = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=point_cloud,
                point_cloud_features=point_cloud_features,
                point_object_id=point_object_id,
                point_invalid_mask=point_invalid_mask,
                camera_info=camera_info,
                q_pointcloud_camera=q_camera_world,
                t_pointcloud_camera=t_camera_world,
                color_max_sh_band=idx // 1000)
            pred_image, _, _ = gaussian_point_cloud_rasterisation(input_data)
            loss = ((pred_image - fake_image)**2).sum()
            loss.backward()
            optimizer.step()
            gaussian_point_adaptive_controller.refinement()
            if idx % 100 == 0 or idx % 100 == 99:
                print(f"loss: {loss.item()}")
            if idx == 0:
                initial_loss = loss.item()
            latest_loss = loss.item()
        self.assertTrue(latest_loss < initial_loss)
