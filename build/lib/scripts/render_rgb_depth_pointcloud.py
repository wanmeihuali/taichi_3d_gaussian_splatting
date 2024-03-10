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
import os
import matplotlib.image as mpimg
from matplotlib import transforms
import json
from scipy import ndimage
from plyfile import PlyData, PlyElement
from PIL import Image
import open3d as o3d

render_rgb = True
render_depth = True
render_point_cloud = True
use_ply = False

def generate_pointcloud(depth_map: np.array, rotation_matrix: np.array, translation_vector: np.array, intrinsics_matrix: np.array):

    # Unpack camera parameters
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    #print("Camera Intrinsics (fx, fy, cx, cy):", fx, fy, cx, cy)

    depth_map = np.squeeze(depth_map, axis=2)
    # Get image shape
    height, width = depth_map.shape #or should it be width, height?
    #print("Width: ", width, " Height: ", height)

    # Generate pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    depth_map = depth_map
    # Compute 3D coordinates
    X = (u - width/2) * depth_map / fx
    X = np.expand_dims(X, axis=2)
    Y = (v - height/2) * depth_map / fy
    Y = np.expand_dims(Y, axis=2)
    Z = depth_map
    Z = np.expand_dims(Z, axis=2)

    total_number = Z.size # total number of points in point cloud
    X = np.reshape(X, (total_number, 1))
    Y = np.reshape(Y, (total_number, 1))
    Z = np.reshape(Z, (total_number, 1))

    # Apply extrinsics
    #points_homogeneous = np.vstack([X, Y, Z, np.ones_like(X)])

    points_homogeneous = np.stack([X, Y, Z, np.ones_like(X)], axis = 1)
    points_homogeneous = np.reshape(points_homogeneous, (total_number, 4))

    # rotation_matrix = np.reshape(rotation_matrix, (3, 3))
    rotation_matrix = np.squeeze(rotation_matrix, axis = 0)
    translation_vector = np.reshape(translation_vector, (3,1))

    #transformation_matrix = np.vstack([np.hstack([np.transpose(rotation_matrix), -np.transpose(rotation_matrix).dot(translation_vector)]),\
    transformation_matrix = np.vstack([np.hstack([rotation_matrix, translation_vector]),\
                                       np.array([0,0,0,1])])

    points_homogeneous = np.transpose(points_homogeneous)
    points_transformed = np.dot(transformation_matrix, points_homogeneous)

    # Convert homogeneous coordinates to 3D points
    point_cloud = np.transpose(points_transformed[:3, :])

    return point_cloud

@ti.kernel
def torchImage2tiImage(field: ti.template(), data: ti.types.ndarray()):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        field[col, data.shape[0] - row -
              1] = ti.math.vec3(data[row, col, 0], data[row, col, 1], data[row, col, 2])


class GaussianPointVisualizer:
    output_path = ""
    trajectory_path = ""
    @dataclass
    class GaussianPointVisualizerConfig:
        device: str = "cuda"
        image_height: int = 270
        image_width: int = 480
        camera_intrinsics: torch.Tensor = torch.tensor(
            [[266.666666666, 0.0, 240.0], [0.0, 266.666666666, 135.0], [0.0, 0.0, 1.0]],
            device="cuda")
        initial_T_pointcloud_camera: torch.Tensor = torch.tensor(
            [[-0.9558463891,-0.1729765611,-0.2375642857,0.7597502332],
             [-0.1518054114,0.9828351684,-0.104833911,-0.2140170738],
              [0.2516203441,-0.0641415712,-0.9656982247,1.3730826485],
               [0.0,0.0,0.0,1.0]],
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

    def __init__(self, trajectory_path, output_path, config) -> None:
        self.config = config
        self.trajectory_path = trajectory_path
        self.output_path = output_path
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16

        scene_list = []

        if use_ply:

          # Extra stuff to read ply file

          PLY_PATH = self.config.parquet_path_list[0]
          plydata = PlyData.read(PLY_PATH)

          xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                          np.asarray(plydata.elements[0]["y"]),
                          np.asarray(plydata.elements[0]["z"])),  axis=1)
          opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

          features_dc = np.zeros((xyz.shape[0], 3, 1))
          features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
          features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
          features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

          extra_f_names = [
              p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
          max_sh_degree = 3
          assert len(extra_f_names) == 3*(max_sh_degree + 1) ** 2 - 3
          features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
          for idx, attr_name in enumerate(extra_f_names):
              features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
          # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
          features_extra = features_extra.reshape(
              (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

          scale_names = [
              p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
          scales = np.zeros((xyz.shape[0], len(scale_names)))
          for idx, attr_name in enumerate(scale_names):
              scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

          rot_names = [
              p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
          rots = np.zeros((xyz.shape[0], len(rot_names)))
          for idx, attr_name in enumerate(rot_names):
              rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

          print("rots.shape", rots.shape)
          # wxyz -> xyzw
          rots = np.roll(rots, shift=-1, axis=1)
          # normalize quaternion
          rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)

        for parquet_path in self.config.parquet_path_list:
          if not use_ply:
            print(f"Loading {parquet_path}")
            scene = GaussianPointCloudScene.from_parquet(parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
          else:
            point_cloud = torch.from_numpy(xyz).float()
            # rot, scale, opacities, features_dc[:, 0, 0], features_extra[:, 0], features_dc[:, 1, 0], features_extra[:, 1], features_dc[:, 2, 0], features_extra[:, 2]
            r = features_dc[:, 0, 0].reshape(-1, 1)
            g = features_dc[:, 1, 0].reshape(-1, 1)
            b = features_dc[:, 2, 0].reshape(-1, 1)
            r_extra = features_extra[:, 0, :].reshape(-1, (max_sh_degree + 1) ** 2 - 1)
            g_extra = features_extra[:, 1, :].reshape(-1, (max_sh_degree + 1) ** 2 - 1)
            b_extra = features_extra[:, 2, :].reshape(-1, (max_sh_degree + 1) ** 2 - 1)
            point_cloud_features_np = np.concatenate([rots, scales, opacities, r, r_extra, g, g_extra, b, b_extra], axis=1)
            point_cloud_features = torch.from_numpy(point_cloud_features_np).float()
            scene = GaussianPointCloudScene(
                        point_cloud=point_cloud,
                        config=GaussianPointCloudScene.PointCloudSceneConfig(
                            max_num_points_ratio=None),
                        point_cloud_features=point_cloud_features
                    )
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

        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(
                self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_id=0,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.001,
                far_plane=1000.,
                depth_to_sort_key_scale=100.))

        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(
            self.config.image_width, self.config.image_height))

    def start(self):
      # For image in json....
      count = 0
      
      with open(self.trajectory_path) as f:
        d = json.load(f)
        for view in d:
          original_image_path = view["image_path"]

          temp_initial_T_pointcloud_camera = torch.tensor(
            view["T_pointcloud_camera"],
            device="cuda")

          temp_initial_T_pointcloud_camera = temp_initial_T_pointcloud_camera.unsqueeze(
            0)
          temp_initial_q_pointcloud_camera, temp_initial_t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(
              temp_initial_T_pointcloud_camera)


          self.state = self.GaussianPointVisualizerState(
              next_q_pointcloud_camera=temp_initial_q_pointcloud_camera,
              next_t_pointcloud_camera=temp_initial_t_pointcloud_camera,
              selected_scene=0,
              last_mouse_pos=None,
          )
          with torch.no_grad():
                  image, depth, _, _= self.rasteriser(
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

          torchImage2tiImage(self.image_buffer, image)
          # Assuming field is a 2D ti.field
          # You may need to convert the ti.field to a numpy array before displaying
          image_np = self.image_buffer.to_numpy()
          image_np_90 = ndimage.rotate(image_np, 90, reshape=False)
          image_np_90 = image_np
          original_image = mpimg.imread(original_image_path)
          im = Image.fromarray((original_image.reshape((original_image.shape[0], original_image.shape[1],3))).astype(np.uint8))
          if not os.path.exists(os.path.join(self.output_path,f'groundtruth/')):
            os.makedirs(os.path.join(self.output_path,'groundtruth/'))
          im.save(os.path.join(self.output_path,f'groundtruth/groundtruth_{count}.png'))

          im = Image.fromarray((image_np_90.reshape((image_np_90.shape[0], image_np_90.shape[1],3))*255).astype(np.uint8))
          im = im.rotate(90, expand=True)
          if not os.path.exists(os.path.join(self.output_path,f'rgb_render/')):
            os.makedirs(os.path.join(self.output_path,'rgb_render/'))
          im.save(os.path.join(self.output_path,f'rgb_render/frame_{count}.png'))

          # DEPTH IMAGE
          depth_image = depth.unsqueeze(2)
          depth_image_max = torch.max(depth_image)
          depth_image = torch.div(depth, depth_image_max)
          depth_image_np = depth_image.cpu().numpy()
          im = Image.fromarray((depth_image_np.reshape((depth_image_np.shape[0], depth_image_np.shape[1])) * 255).astype(np.uint8))
          if not os.path.exists(os.path.join(self.output_path,f'depth_render')):
            os.makedirs(os.path.join(self.output_path,'depth_render'))
          im.save(os.path.join(self.output_path,f'depth_render/depth_{count}.png'))        
          
          # POINTCLOUD 
          image = depth.unsqueeze(2)

          next_q_pointcloud_camera_numpy = self.state.next_q_pointcloud_camera.cpu().numpy()
          r_camera_pointcloud = R.from_quat(next_q_pointcloud_camera_numpy).as_matrix()
          t_camera_pointcloud = self.state.next_t_pointcloud_camera.cpu().numpy()
          pointcloud = generate_pointcloud(image.cpu().numpy(),\
                                           r_camera_pointcloud, t_camera_pointcloud, self.camera_info.camera_intrinsics.cpu().numpy())
          pcd = o3d.geometry.PointCloud()
          pcd.points = o3d.utility.Vector3dVector(pointcloud)
          if not os.path.exists(os.path.join(self.output_path,f'point_clouds')):
            os.makedirs(os.path.join(self.output_path,'point_clouds'))
          o3d.io.write_point_cloud(os.path.join(self.output_path,f"point_clouds/pointcloud_{count}.pcd"), pcd)
          count+=1
          

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

parser = argparse.ArgumentParser(description='Parquet file path')
parser.add_argument('--parquet-path', type=str, help='Parquet file path')
parser.add_argument('--trajectory-path', type=str, help='Json trajectory file path')
parser.add_argument('--output-path', type=str, help='Output folder path')

args = parser.parse_args()

if os.path.splitext(args.parquet_path)[-1] == '.ply':
    use_ply = True
print("Opening parquet file ", args.parquet_path)
parquet_path_list=[args.parquet_path]
ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
visualizer = GaussianPointVisualizer(args.trajectory_path, args.output_path, config=GaussianPointVisualizer.GaussianPointVisualizerConfig(
    parquet_path_list=parquet_path_list,
))
visualizer.start()
