import torch
import torch.nn as nn

class CameraPoses(nn.Module):
    def __init__(self, q_pointcloud_camera_table, t_pointcloud_camera_table):
        super().__init__()
        self.q_pointcloud_camera_table = nn.Parameter(q_pointcloud_camera_table)
        self.t_pointcloud_camera_table = nn.Parameter(t_pointcloud_camera_table)

    def forward(self, camera_pose_indices):
        q_pointcloud_camera = self.q_pointcloud_camera_table[camera_pose_indices].contiguous()
        t_pointcloud_camera = self.t_pointcloud_camera_table[camera_pose_indices].contiguous()
        return q_pointcloud_camera, t_pointcloud_camera

    def normalize_quaternion(self):
        with torch.no_grad():
            self.q_pointcloud_camera_table /= torch.norm(self.q_pointcloud_camera_table, dim=-1, keepdim=True)