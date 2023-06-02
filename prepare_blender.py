# %%
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import trimesh
import numpy as np
import json
# %%
stl_path = "/home/kuangyuan/hdd/datasets/blender/scene/Boots.stl"
dataset_path = "/home/kuangyuan/hdd/datasets/nerf_gen/test_1/dataset_d3/dataset_d3_train"
camera_info_path = "/home/kuangyuan/hdd/datasets/nerf_gen/test_1/dataset_d3/dataset_d3_train/transforms_train.json"
output_dir = "/home/kuangyuan/hdd/datasets/nerf_gen/test_1/dataset_d3/3d_gaussian"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
mesh = trimesh.load(stl_path)

# %%
# Sample points on the surface
point_cloud, index = trimesh.sample.sample_surface(mesh, count=50000)
point_cloud = point_cloud.T
print(point_cloud.shape)
# %%
camera_info = json.load(open(camera_info_path))
# %%
camera_intrinsics = np.array([
    [camera_info["fl_x"], 0, camera_info["cx"]],
    [0, camera_info["fl_y"], camera_info["cy"]],
    [0, 0, 1]
])
camera_width = camera_info["w"]
camera_height = camera_info["h"]

# %%
print(camera_intrinsics)
print(camera_width, camera_height)

# %%


class CameraView:
    def __init__(self, T_pointcloud_camera, label, sensor_id, path):
        self.T_pointcloud_camera = T_pointcloud_camera
        if isinstance(T_pointcloud_camera, np.ndarray):
            self.T_pointcloud_camera = torch.from_numpy(
                T_pointcloud_camera)
        self.label = label
        self.sensor_id = sensor_id
        self.path = path


class CameraInfo:
    def __init__(self, camera_intrinsic, camera_width, camera_height):
        self.camera_intrinsics = camera_intrinsic
        if isinstance(camera_intrinsic, np.ndarray):
            self.camera_intrinsics = torch.from_numpy(camera_intrinsic)
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)


camera_view_list = []
camera_info_dict = {0: CameraInfo(
    camera_intrinsics, camera_width, camera_height)}
for idx, frame in enumerate(camera_info["frames"]):
    image_path = frame["file_path"]
    T_pointcloud_camera_blender = np.array(
        frame["transform_matrix"]).reshape(4, 4)
    T_pointcloud_camera = T_pointcloud_camera_blender
    flip_x = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    T_pointcloud_camera = T_pointcloud_camera_blender @ flip_x
    image_full_path = os.path.join(dataset_path, image_path)
    camera_view_list.append(CameraView(
        T_pointcloud_camera, idx, 0, image_full_path))

    """
    image = plt.imread(image_full_path)

    point_cloud_camera = np.linalg.inv(T_pointcloud_camera) @ \
        np.concatenate([point_cloud, np.ones_like(point_cloud[:1, :])], axis=0)
    point_cloud_camera = point_cloud_camera[:3, :]
    point_cloud_uv1 = (camera_intrinsics @ point_cloud_camera) / \
        point_cloud_camera[2, :]
    point_cloud_uv = point_cloud_uv1[:2, :].T
    print(point_cloud_camera[:2].mean())
    plt.imshow(image)
    plt.scatter(point_cloud_uv[:, 0], point_cloud_uv[:, 1], s=1)
    plt.xlim([0, camera_width])
    plt.ylim([camera_height, 0])
    plt.show()
    break
    """


# %%
point_cloud_df = pd.DataFrame(point_cloud.T, columns=["x", "y", "z"])
# %%

point_cloud_df.to_parquet(os.path.join(
    output_dir, "point_cloud.parquet"))

data = {
    "image_path": [camera_view.path for camera_view in camera_view_list],
    "T_pointcloud_camera": [camera_view.T_pointcloud_camera.numpy() for camera_view in camera_view_list],
    "camera_intrinsics": [camera_info_dict[camera_view.sensor_id].camera_intrinsics.numpy() for camera_view in camera_view_list],
    "camera_height": [camera_info_dict[camera_view.sensor_id].camera_height for camera_view in camera_view_list],
    "camera_width": [camera_info_dict[camera_view.sensor_id].camera_width for camera_view in camera_view_list],
    "camera_id": [camera_view.sensor_id for camera_view in camera_view_list],
}
df = pd.DataFrame(data)
# select training data and validation data, have a val every 3 frames
df["is_train"] = df.index % 3 != 0
train_df = df[df["is_train"]].copy()
val_df = df[~df["is_train"]].copy()
train_df.drop(columns=["is_train"], inplace=True)
val_df.drop(columns=["is_train"], inplace=True)
# df.to_json(os.path.join(output_dir, "kitti.json"), orient="records")
train_df.to_json(os.path.join(
    output_dir, "boots_train.json"), orient="records")
val_df.to_json(os.path.join(output_dir, "boots_val.json"), orient="records")


# %%

# %%
