# %%
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import torch2ti
from dataclasses import dataclass
import torch
import numpy as np
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

    @dataclass
    class GaussianPointVisualizerState:
        next_T_pointcloud_camera: torch.Tensor

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
        self.state = self.GaussianPointVisualizerState(
            next_T_pointcloud_camera=self.config.initial_T_pointcloud_camera.to(self.config.device)
        )

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
            for event in events:
                if event.key == "W":
                    pass

            with torch.no_grad():
                image, _, _ = self.rasteriser(
                    GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=self.camera_info,
                        T_pointcloud_camera=self.state.next_T_pointcloud_camera,
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
    




# %%
ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
parquet_path_0 = "/home/kuangyuan/hdd/Development/taichi_3d_gaussian_splatting/logs/tat_truck_every_8_experiment/scene_7000.parquet"
# parquet_path_1 = "/home/kuangyuan/hdd/Development/taichi_3d_gaussian_splatting/logs/boots_super_sparse_view_space_control_remove_floater_weak/scene_7000.parquet"
visualizer = GaussianPointVisualizer(config=GaussianPointVisualizer.GaussianPointVisualizerConfig(
    parquet_path_list=[parquet_path_0]
))
visualizer.start()
# %%
