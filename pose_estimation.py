# %%
import sys
sys.path.append("../..")
# %%
import argparse
import taichi as ti
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.CameraPoses import CameraPoses
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.ImagePoseDataset import ImagePoseDataset
from taichi_3d_gaussian_splatting.LossFunction import LossFunction
from taichi_3d_gaussian_splatting.utils import torch2ti, SE3_to_quaternion_and_translation_torch, quaternion_rotate_torch, quaternion_multiply_torch, quaternion_conjugate_torch, inverse_SE3_qt_torch
from dataclasses import dataclass
from typing import List, Tuple
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt
# %%
DELTA_T_RANGE = 0.2
DELTA_ANGLE_RANGE = 0.3


def add_delta_to_se3(se3_matrix: np.ndarray):
    np.random.seed(0)
    delta_t = np.random.uniform(-DELTA_T_RANGE, DELTA_T_RANGE, size=(3,))
    delta_angle = np.random.uniform(-DELTA_ANGLE_RANGE,
                                    DELTA_ANGLE_RANGE, size=(3,))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(delta_angle[0]), -np.sin(delta_angle[0])],
                   [0, np.sin(delta_angle[0]), np.cos(delta_angle[0])]])
    RY = np.array([[np.cos(delta_angle[1]), 0, np.sin(delta_angle[1])],
                   [0, 1, 0],
                   [-np.sin(delta_angle[1]), 0, np.cos(delta_angle[1])]])
    Rz = np.array([[np.cos(delta_angle[2]), -np.sin(delta_angle[2]), 0],
                   [np.sin(delta_angle[2]), np.cos(delta_angle[2]), 0],
                   [0, 0, 1]])
    delta_rotation = Rz @ RY @ Rx
    se3_matrix[:3, :3] = se3_matrix[:3, :3] @ delta_rotation
    # se3_matrix[:3, 3] += delta_t
    return se3_matrix


# %%
ti.init(ti.cuda)
trained_parquet_path = "/home/kuangyuan/hdd/Development/taichi_3d_gaussian_splatting/logs/tat_truck_every_8_experiment/scene_29000.parquet"
dataset_json_path = "data/tat_truck_every_8_test/train.json"

rasterisation = GaussianPointCloudRasterisation(
    config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
        enable_grad_camera_pose=True,
        near_plane=0.8,
        far_plane=1000.,
        depth_to_sort_key_scale=100.))
scene = GaussianPointCloudScene.from_parquet(
    trained_parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
scene = scene.cuda()
train_dataset = ImagePoseDataset(
    dataset_json_path=dataset_json_path)

loss_function = LossFunction(
    config=LossFunction.LossFunctionConfig(
        enable_regularization=False))

df = pd.read_json(dataset_json_path, orient="records")
df["T_pointcloud_camera_original"] = df["T_pointcloud_camera"].apply(
    lambda x: np.array(x).reshape(4, 4))
df["T_pointcloud_camera"] = df["T_pointcloud_camera_original"].apply(
    lambda x: add_delta_to_se3(x))

# save df to a temp json file
df.to_json("/tmp/temp.json", orient="records")
with_noise_dataset_json_path = "/tmp/temp.json"

camera_poses = CameraPoses(dataset_json_path=with_noise_dataset_json_path)
camera_poses = camera_poses.cuda()
camera_pose_optimizer = torch.optim.AdamW(
    camera_poses.parameters(), lr=1e-3, betas=(0.9, 0.999))
"""
camera_pose_optimizer = torch.optim.AdamW(
    [camera_poses.t_camera_pointcloud_table], lr=1e-3, betas=(0.9, 0.999))
"""
"""
camera_pose_optimizer = torch.optim.AdamW(
    [camera_poses.q_camera_pointcloud_table], lr=1e-3, betas=(0.9, 0.999))
"""

distance_list = []
angle_list = []
loss_list = []

for i in range(1000):
    # decay learning rate by 0.5 every 50 iterations
    if i % 50 == 0:
        for param_group in camera_pose_optimizer.param_groups:
            param_group['lr'] *= 0.9
    camera_pose_optimizer.zero_grad()
    image_gt, input_q_pointcloud_camera, input_t_pointcloud_camera, camera_pose_indices, camera_info = train_dataset[
        200]
    input_q_camera_pointcloud, input_t_camera_pointcloud = inverse_SE3_qt_torch(
        q=input_q_pointcloud_camera, t=input_t_pointcloud_camera)
    trained_q_camera_pointcloud, trained_t_camera_pointcloud = camera_poses(
        camera_pose_indices)
    print(
        f"trained_q_camera_pointcloud: {trained_q_camera_pointcloud.detach().cpu().numpy()}")
    print(
        f"input_q_camera_pointcloud: {input_q_camera_pointcloud.detach().cpu().numpy()}")
    print(
        f"trained_t_camera_pointcloud: {trained_t_camera_pointcloud.detach().cpu().numpy()}")
    print(
        f"input_t_camera_pointcloud: {input_t_camera_pointcloud.detach().cpu().numpy()}")

    image_gt = image_gt.cuda()
    input_q_camera_pointcloud = input_q_camera_pointcloud.cuda()
    input_t_camera_pointcloud = input_t_camera_pointcloud.cuda()
    trained_q_camera_pointcloud = trained_q_camera_pointcloud.cuda()
    trained_t_camera_pointcloud = trained_t_camera_pointcloud.cuda()
    camera_info.camera_intrinsics = camera_info.camera_intrinsics.cuda()

    delta_t = input_t_camera_pointcloud - trained_t_camera_pointcloud
    distance = torch.norm(delta_t, dim=-1).item()
    distance_list.append(distance)
    delta_angle_cos = (input_q_camera_pointcloud *
                       trained_q_camera_pointcloud).sum(dim=-1)
    delta_angle = torch.acos(delta_angle_cos).item() * 180 / np.pi
    angle_list.append(delta_angle)
    camera_info.camera_width = int(camera_info.camera_width)
    camera_info.camera_height = int(camera_info.camera_height)
    gaussian_point_cloud_rasterisation_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
        point_cloud=scene.point_cloud.contiguous(),
        point_cloud_features=scene.point_cloud_features.contiguous(),
        point_object_id=scene.point_object_id.contiguous(),
        point_invalid_mask=scene.point_invalid_mask.contiguous(),
        camera_info=camera_info,
        q_camera_pointcloud=trained_q_camera_pointcloud.contiguous(),
        t_camera_pointcloud=trained_t_camera_pointcloud.contiguous(),
        color_max_sh_band=3,
    )
    image_pred, image_depth, pixel_valid_point_count = rasterisation(
        gaussian_point_cloud_rasterisation_input)
    # clip to [0, 1]
    image_pred = torch.clamp(image_pred, min=0, max=1)
    # hxwx3->3xhxw
    image_pred = image_pred.permute(2, 0, 1)
    loss, l1_loss, ssim_loss = loss_function(
        image_pred,
        image_gt,
        point_invalid_mask=scene.point_invalid_mask,
        pointcloud_features=scene.point_cloud_features)
    loss.backward()
    camera_pose_optimizer.step()
    camera_poses.normalize_quaternion()
    loss_list.append(loss.item())

iteration = np.arange(len(distance_list))
ax, fig = plt.subplots(1, 3, figsize=(15, 5))
fig[0].plot(iteration, distance_list, label="distance")
fig[0].set_title("distance")
fig[1].plot(iteration, angle_list, label="angle")
fig[1].set_title("angle")
fig[2].plot(iteration, loss_list, label="loss")
fig[2].set_title("loss")
plt.show()

# %%
