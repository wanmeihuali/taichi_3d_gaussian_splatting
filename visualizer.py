# %%
import argparse
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import torch2ti
from dataclasses import dataclass
from typing import Tuple
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
# %%
@ti.kernel
def torchImage2tiImage(field: ti.template(), data: ti.types.ndarray()):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        field[col, data.shape[0] - row - 1] = ti.math.vec3(data[row, col, 0], data[row, col, 1], data[row, col, 2])

class GaussianPointVisualizer:
    @dataclass
    class GaussianPointVisualizerConfig:
        device: str = "cuda"
        image_height: int = 546
        image_width: int = 980
        camera_intrinsics: torch.Tensor = torch.tensor(
            [[581.743,0.0,490.0],[0.0,581.743,273.0],[0.0,0.0,1.0]],
            device="cuda")
        initial_T_pointcloud_camera: torch.Tensor = torch.tensor(
            [[0.9992602094,-0.0041446825,0.0382342376,0.8111615373],[0.0047891027,0.9998477637,-0.0167783848,0.4972433596],[-0.0381588759,0.0169490798,0.999127935,-3.8378280443],[0.0,0.0,0.0,1.0]],
            device="cuda")
        parquet_path_list: list[str] = None
        step_size: float = 0.1
        mouse_sensitivity: float = 1

    @dataclass
    class GaussianPointVisualizerState:
        next_T_pointcloud_camera: torch.Tensor
        selected_scene: int = 0
        extra_translation: torch.Tensor = None
        extra_rotation: torch.Tensor = None
        extra_scale: torch.Tensor = None
        last_mouse_pos: Tuple[float, float] = None

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        extra_rotation: torch.Tensor
        extra_translation: torch.Tensor
        extra_scale: torch.Tensor
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
        self.state = self.GaussianPointVisualizerState(
            next_T_pointcloud_camera=self.config.initial_T_pointcloud_camera.to(self.config.device),
            selected_scene=0,
            extra_translation=torch.zeros_like(self.scene.point_cloud),
            extra_rotation=torch.zeros((self.scene.point_cloud.shape[0], 4), device=self.config.device),
            extra_scale=torch.ones_like(self.scene.point_cloud),
            last_mouse_pos=None,
        )
        self.state.extra_rotation[:, 3] = 1.

        self.gui = ti.GUI(
            "Gaussian Point Visualizer", 
            (self.config.image_width, self.config.image_height),
            fast_gui=True)

        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_id=0,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.8,
                far_plane=1000.,
                depth_to_sort_key_scale=100.))
        
        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(self.config.image_width, self.config.image_height))
    
    def start(self):
        while self.gui.running:
            events = self.gui.get_events(self.gui.PRESS)
            start_offset = self.extra_scene_info_dict[self.state.selected_scene].start_offset
            end_offset = self.extra_scene_info_dict[self.state.selected_scene].end_offset
            for event in events:
                if event.key == "w":
                    # self.state.next_T_pointcloud_camera[2, 3] += self.config.step_size
                    delta = torch.eye(4, device=self.config.device)
                    delta[2, 3] = self.config.step_size
                    self.state.next_T_pointcloud_camera = self.state.next_T_pointcloud_camera @ delta
                elif event.key == "s":
                    delta = torch.eye(4, device=self.config.device)
                    delta[2, 3] = -self.config.step_size
                    self.state.next_T_pointcloud_camera = self.state.next_T_pointcloud_camera @ delta
                elif event.key == "a":
                    delta = torch.eye(4, device=self.config.device)
                    delta[0, 3] = -self.config.step_size
                    self.state.next_T_pointcloud_camera = self.state.next_T_pointcloud_camera @ delta
                elif event.key == "d":
                    delta = torch.eye(4, device=self.config.device)
                    delta[0, 3] = self.config.step_size
                    self.state.next_T_pointcloud_camera = self.state.next_T_pointcloud_camera @ delta
                elif event.key == "q":
                    # rotate
                    self.state.next_T_pointcloud_camera[:3, :3] = torch.matmul(
                        torch.tensor([[0.9998477637, 0.0, 0.0167783848],
                                      [0.0, 1.0, 0.0],
                                      [-0.0167783848, 0.0, 0.9998477637]],
                                     device=self.config.device),
                        self.state.next_T_pointcloud_camera[:3, :3])
                elif event.key == "e":
                    self.state.next_T_pointcloud_camera[:3, :3] = torch.matmul(
                        torch.tensor([[0.9998477637, 0.0, -0.0167783848],
                                      [0.0, 1.0, 0.0],
                                      [0.0167783848, 0.0, 0.9998477637]],
                                     device=self.config.device),
                        self.state.next_T_pointcloud_camera[:3, :3])
                elif event.key == "=":
                    delta = torch.eye(4, device=self.config.device)
                    delta[1, 3] = -self.config.step_size
                    self.state.next_T_pointcloud_camera = self.state.next_T_pointcloud_camera @ delta
                elif event.key == "-":
                    delta = torch.eye(4, device=self.config.device)
                    delta[1, 3] = self.config.step_size
                    self.state.next_T_pointcloud_camera = self.state.next_T_pointcloud_camera @ delta
                elif event.key >= "0" and event.key <= "9":
                    scene_index = int(event.key)
                    if scene_index < len(self.extra_scene_info_dict):
                        self.state.selected_scene = scene_index
                elif event.key == "h":
                    self.scene.point_invalid_mask[start_offset:end_offset] = 1
                elif event.key == "p":
                    self.scene.point_invalid_mask[start_offset:end_offset] = 0
                elif event.key == self.gui.UP:
                    self.state.extra_translation[start_offset:end_offset, 1] += self.config.step_size
                elif event.key == self.gui.DOWN:
                    self.state.extra_translation[start_offset:end_offset, 1] -= self.config.step_size
                elif event.key == self.gui.LEFT:
                    self.state.extra_translation[start_offset:end_offset, 0] -= self.config.step_size
                elif event.key == self.gui.RIGHT:
                    self.state.extra_translation[start_offset:end_offset, 0] += self.config.step_size
                elif event.key == ",":
                    self.state.extra_scale[start_offset:end_offset] *= 0.9
                elif event.key == ".":
                    self.state.extra_scale[start_offset:end_offset] *= 1.1
            
            mouse_pos = self.gui.get_cursor_pos()
            if self.gui.is_pressed(self.gui.LMB):
                if self.state.last_mouse_pos is None:
                    self.state.last_mouse_pos = mouse_pos
                else:
                    dy, dx = mouse_pos[0] - self.state.last_mouse_pos[0], mouse_pos[1] - self.state.last_mouse_pos[1]
                    angle_x = dx * self.config.mouse_sensitivity
                    angle_z = dy * self.config.mouse_sensitivity
                    qx = [np.sin(angle_x / 2), 0, 0, np.cos(angle_x / 2)]
                    # qy = [0, np.sin(angle_y / 2), 0, np.cos(angle_y / 2)]
                    qz = [0, 0, np.sin(angle_z / 2), np.cos(angle_z / 2)]
                    
                    current_rot = self.extra_scene_info_dict[self.state.selected_scene].extra_rotation.detach().cpu().numpy()
                    next_rot = self._quaternion_multiply(self._quaternion_multiply(qx, current_rot), qz)
                    self.extra_scene_info_dict[self.state.selected_scene].extra_rotation = torch.tensor(next_rot, device=self.config.device, dtype=torch.float32)
                    self.state.extra_rotation[start_offset:end_offset] = self.extra_scene_info_dict[self.state.selected_scene].extra_rotation
                    self.state.last_mouse_pos = mouse_pos
            else:
                self.state.last_mouse_pos = None
                


            with torch.no_grad():
                image, _, _ = self.rasteriser(
                    GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=self.camera_info,
                        T_pointcloud_camera=self.state.next_T_pointcloud_camera,
                        color_max_sh_band=3,
                        point_extra_translation=self.state.extra_translation,
                        point_extra_rotation=self.state.extra_rotation,
                        point_extra_scale=self.state.extra_scale, 
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
        num_of_points_list = [scene.point_cloud.shape[0] for scene in scene_list]
        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
        end_offset_list = np.cumsum(num_of_points_list).tolist()
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                extra_rotation=torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.config.device),
                extra_translation=torch.zeros(3, dtype=torch.float32, device=self.config.device),
                extra_scale=torch.ones(3, dtype=torch.float32, device=self.config.device),
                visible=True
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            config=GaussianPointCloudScene.PointCloudSceneConfig(
                max_num_points_ratio=None
            ))
        return merged_scene

    
    @staticmethod
    def _quaternion_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x =  w1*x2 + x1*w2 + y1*z2 - z1*y2
        y =  w1*y2 - x1*z2 + y1*w2 + z1*x2
        z =  w1*z2 + x1*y2 - y1*x2 + z1*w2
        w =  w1*w2 - x1*x2 - y1*y2 - z1*z2
        return [x, y, z, w]

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path_list", type=str, nargs="+", required=True)
    args = parser.parse_args()
    parquet_path_list = args.parquet_path_list
    ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
    visualizer = GaussianPointVisualizer(config=GaussianPointVisualizer.GaussianPointVisualizerConfig(
        parquet_path_list=parquet_path_list,
    ))
    visualizer.start()