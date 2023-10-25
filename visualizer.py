# %%
import argparse
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import torch2ti, SE3_to_quaternion_and_translation_torch, quaternion_rotate_torch, quaternion_multiply_torch, quaternion_conjugate_torch
from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
# %%


@ti.kernel
def torchImage2tiImage(field: ti.template(), data: ti.types.ndarray()):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        field[col, data.shape[0] - row -
              1] = ti.math.vec3(data[row, col, 0], data[row, col, 1], data[row, col, 2])


class GaussianPointVisualizer:
    @dataclass
    class GaussianPointVisualizerConfig:
        device: str = "cuda"
        image_height: int = 546
        image_width: int = 980
        camera_intrinsics: torch.Tensor = torch.tensor(
            [[581.743, 0.0, 490.0], [0.0, 581.743, 273.0], [0.0, 0.0, 1.0]],
            device="cuda")
        initial_T_pointcloud_camera: torch.Tensor = torch.tensor(
            [[0.9992602094, -0.0041446825, 0.0382342376, 0.8111615373], [0.0047891027, 0.9998477637, -0.0167783848,
                                                                         0.4972433596], [-0.0381588759, 0.0169490798, 0.999127935, -3.8378280443], [0.0, 0.0, 0.0, 1.0]],
            device="cuda")
        parquet_path_list: List[str] = None
        step_size: float = 0.1
        mouse_sensitivity: float = 3

    @dataclass
    class GaussianPointVisualizerState:
        next_t_pointcloud_camera: torch.Tensor
        next_q_pointcloud_camera: torch.Tensor
        selected_scene: int = 0
        last_mouse_pos: Tuple[float, float] = None

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, config) -> None:
        self.config = config
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16

        scene_list = []
        for parquet_path in self.config.parquet_path_list:
            print(f"Loading {parquet_path}")
            scene = GaussianPointCloudScene.from_parquet(
                parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
            scene_list.append(scene)
        print("Merging scenes")
        self.scene = self._merge_scenes(scene_list)
        print("Done merging scenes")
        self.scene = self.scene.to(self.config.device)

        initial_T_pointcloud_camera = self.config.initial_T_pointcloud_camera.to(
            self.config.device)
        initial_T_pointcloud_camera = initial_T_pointcloud_camera.unsqueeze(
            0).repeat(len(scene_list), 1, 1)
        initial_q_pointcloud_camera, initial_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
            initial_T_pointcloud_camera)

        self.state = self.GaussianPointVisualizerState(
            next_q_pointcloud_camera=initial_q_pointcloud_camera,
            next_t_pointcloud_camera=initial_t_pointcloud_camera,
            selected_scene=0,
            last_mouse_pos=None,
        )

        self.gui = ti.GUI(
            "Gaussian Point Visualizer",
            (self.config.image_width, self.config.image_height),
            fast_gui=True)

        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(
                self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_id=0,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.8,
                far_plane=1000.,
                depth_to_sort_key_scale=100.))

        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(
            self.config.image_width, self.config.image_height))

    def start(self):
        while self.gui.running:
            events = self.gui.get_events(self.gui.PRESS)
            start_offset = 0
            end_offset = self.scene.point_cloud.shape[0]
            selected_objects = torch.arange(
                len(self.extra_scene_info_dict), device=self.config.device)
            object_selected = self.state.selected_scene != 0
            move_factor = -1 if object_selected else 1
            if object_selected:
                start_offset = self.extra_scene_info_dict[self.state.selected_scene - 1].start_offset
                end_offset = self.extra_scene_info_dict[self.state.selected_scene - 1].end_offset
                selected_objects = self.state.selected_scene - 1

            for event in events:
                if event.key >= "0" and event.key <= "9":
                    scene_index = int(event.key)
                    if scene_index <= len(self.extra_scene_info_dict):
                        self.state.selected_scene = scene_index
                elif event.key == "w":
                    delta = torch.zeros_like(
                        self.state.next_t_pointcloud_camera)
                    delta[selected_objects,
                          2] = self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(
                        v=delta, q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "s":
                    delta = torch.zeros_like(
                        self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 2] = - \
                        self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(
                        v=delta, q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "a":
                    delta = torch.zeros_like(
                        self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 0] = - \
                        self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(
                        v=delta, q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "d":
                    delta = torch.zeros_like(
                        self.state.next_t_pointcloud_camera)
                    delta[selected_objects,
                          0] = self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(
                        v=delta, q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "-":
                    delta = torch.zeros_like(
                        self.state.next_t_pointcloud_camera)
                    delta[selected_objects,
                          1] = self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(
                        v=delta, q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "=":
                    delta = torch.zeros_like(
                        self.state.next_t_pointcloud_camera)
                    delta[selected_objects, 1] = - \
                        self.config.step_size * move_factor
                    delta = quaternion_rotate_torch(
                        v=delta, q=self.state.next_q_pointcloud_camera)
                    self.state.next_t_pointcloud_camera += delta
                elif event.key == "q":
                    delta_q = torch.zeros_like(
                        self.state.next_q_pointcloud_camera)
                    delta_q[..., 3] = 1.
                    delta_q[selected_objects,
                            3] = np.cos(-self.config.step_size / 2 * move_factor)
                    delta_q[selected_objects,
                            1] = np.sin(-self.config.step_size / 2 * move_factor)
                    delta_q = delta_q / \
                        torch.norm(delta_q, dim=-1, keepdim=True)
                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q)
                    self.state.next_q_pointcloud_camera = self.state.next_q_pointcloud_camera / \
                        torch.norm(self.state.next_q_pointcloud_camera,
                                   dim=-1, keepdim=True)
                elif event.key == "e":
                    delta_q = torch.zeros_like(
                        self.state.next_q_pointcloud_camera)
                    delta_q[..., 3] = 1.
                    delta_q[selected_objects, 3] = np.cos(
                        self.config.step_size / 2 * move_factor)
                    delta_q[selected_objects, 1] = np.sin(
                        self.config.step_size / 2 * move_factor)
                    delta_q = delta_q / \
                        torch.norm(delta_q, dim=-1, keepdim=True)
                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q)
                    self.state.next_q_pointcloud_camera = self.state.next_q_pointcloud_camera / \
                        torch.norm(self.state.next_q_pointcloud_camera,
                                   dim=-1, keepdim=True)
                elif event.key == "h":
                    self.scene.point_invalid_mask[start_offset:end_offset] = 1
                elif event.key == "p":
                    self.scene.point_invalid_mask[start_offset:end_offset] = 0

            mouse_pos = self.gui.get_cursor_pos()
            if self.gui.is_pressed(self.gui.LMB):
                if self.state.last_mouse_pos is None:
                    self.state.last_mouse_pos = mouse_pos
                else:
                    dy, dx = mouse_pos[0] - self.state.last_mouse_pos[0], mouse_pos[1] - \
                        self.state.last_mouse_pos[1]
                    angle_x = dx * self.config.mouse_sensitivity
                    angle_y = dy * self.config.mouse_sensitivity
                    if self.state.selected_scene != 0:
                        pointcloud_object_center = self.extra_scene_info_dict[self.state.selected_scene - 1].center.unsqueeze(
                            0)
                        pointcloud_object_center = pointcloud_object_center.to(
                            self.state.next_t_pointcloud_camera.device)
                        pointcloud_camera_to_center = pointcloud_object_center - \
                            self.state.next_t_pointcloud_camera[selected_objects]
                        camera_camera_to_center = quaternion_rotate_torch(
                            q=quaternion_conjugate_torch(
                                self.state.next_q_pointcloud_camera[selected_objects]),
                            v=pointcloud_camera_to_center)

                    delta_q_y = torch.zeros_like(
                        self.state.next_q_pointcloud_camera)
                    delta_q_y[..., 3] = 1.
                    delta_q_y[selected_objects, 3] = np.cos(angle_y / 2)
                    delta_q_y[selected_objects, 1] = np.sin(angle_y / 2)
                    delta_q_y = delta_q_y / \
                        torch.norm(delta_q_y, dim=-1, keepdim=True)

                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q_y)
                    self.state.next_q_pointcloud_camera = self.state.next_q_pointcloud_camera / \
                        torch.norm(self.state.next_q_pointcloud_camera,
                                   dim=-1, keepdim=True)

                    delta_q_x = torch.zeros_like(
                        self.state.next_q_pointcloud_camera)
                    delta_q_x[..., 3] = 1.
                    delta_q_x[selected_objects, 3] = np.cos(angle_x / 2)
                    delta_q_x[selected_objects, 0] = np.sin(angle_x / 2)
                    delta_q_x = delta_q_x / \
                        torch.norm(delta_q_x, dim=-1, keepdim=True)

                    self.state.next_q_pointcloud_camera = quaternion_multiply_torch(
                        self.state.next_q_pointcloud_camera, delta_q_x)
                    self.state.next_q_pointcloud_camera = self.state.next_q_pointcloud_camera / \
                        torch.norm(self.state.next_q_pointcloud_camera,
                                   dim=-1, keepdim=True)

                    if object_selected:
                        pointcloud_object_center = self.extra_scene_info_dict[self.state.selected_scene - 1].center.unsqueeze(
                            0)
                        pointcloud_object_center = pointcloud_object_center.to(
                            self.state.next_t_pointcloud_camera.device)
                        object_center_new = quaternion_rotate_torch(
                            q=self.state.next_q_pointcloud_camera[selected_objects],
                            v=camera_camera_to_center)
                        self.state.next_t_pointcloud_camera[selected_objects] = pointcloud_object_center - object_center_new

                    self.state.last_mouse_pos = mouse_pos
            else:
                self.state.last_mouse_pos = None

            with torch.no_grad():
                image, _, _ = self.rasteriser(
                    GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        point_object_id=self.scene.point_object_id,
                        camera_info=self.camera_info,
                        q_pointcloud_camera=self.state.next_q_pointcloud_camera,
                        t_pointcloud_camera=self.state.next_t_pointcloud_camera,
                        color_max_sh_band=3,
                    )
                )
                # ti.profiler.print_kernel_profiler_info("count")
                # ti.profiler.clear_kernel_profiler_info()
            torchImage2tiImage(self.image_buffer, image)
            self.gui.set_image(self.image_buffer)
            self.gui.show()
        self.gui.close()

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


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path_list", type=str,
                        nargs="+", required=True)
    args = parser.parse_args()
    parquet_path_list = args.parquet_path_list
    ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
    visualizer = GaussianPointVisualizer(config=GaussianPointVisualizer.GaussianPointVisualizerConfig(
        parquet_path_list=parquet_path_list,
    ))
    visualizer.start()
