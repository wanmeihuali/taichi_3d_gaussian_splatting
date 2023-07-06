import torch
import torch.nn as nn

class CameraPoses(nn.Module):
    def __init__(self, q_pointcloud_camera_table, t_pointcloud_camera_table):
        super().__init__()
        self.q_pointcloud_camera_table = q_pointcloud_camera_table
        self.t_pointcloud_camera_table = t_pointcloud_camera_table

    def forward(self, camera_pose_indices):
        q_pointcloud_camera = self.q_pointcloud_camera_table[camera_pose_indices].contiguous()
        t_pointcloud_camera = self.t_pointcloud_camera_table[camera_pose_indices].contiguous()
        return q_pointcloud_camera, t_pointcloud_camera