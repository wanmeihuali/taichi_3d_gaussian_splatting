import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .utils import se3_to_quaternion_and_translation_torch
from typing import Any


class CameraPoses(nn.Module):
    def __init__(self, dataset_json_path: str):
        super().__init__()
        required_columns = ["T_pointcloud_camera"]
        self.df = pd.read_json(dataset_json_path, orient="records")
        for column in required_columns:
            assert column in self.df.columns, f"column {column} is not in the dataset"
        
        q_pointcloud_camera_table, t_pointcloud_camera_table, self.num_objects = self._get_all_camera_poses()
        self.input_q_pointcloud_camera_table = q_pointcloud_camera_table
        self.input_t_pointcloud_camera_table = t_pointcloud_camera_table
        self.q_pointcloud_camera_table = nn.Parameter(q_pointcloud_camera_table)
        self.t_pointcloud_camera_table = nn.Parameter(t_pointcloud_camera_table)
        

    def forward(self, camera_pose_indices):
        q_pointcloud_camera = self.q_pointcloud_camera_table[camera_pose_indices].contiguous()
        t_pointcloud_camera = self.t_pointcloud_camera_table[camera_pose_indices].contiguous()
        return q_pointcloud_camera, t_pointcloud_camera

    def normalize_quaternion(self):
        with torch.no_grad():
            self.q_pointcloud_camera_table /= torch.norm(self.q_pointcloud_camera_table, dim=-1, keepdim=True)

    def to_parquet(self, path):
        with torch.no_grad():
            df = pd.DataFrame({
                "trained_q_pointcloud_camera_table": self._tensor_to_list(self.q_pointcloud_camera_table), # (N, 4)
                "trained_t_pointcloud_camera_table": self._tensor_to_list(self.t_pointcloud_camera_table), # (N, 3)
                "input_q_pointcloud_camera_table": self._tensor_to_list(self.input_q_pointcloud_camera_table), # (N, 4)
                "input_t_pointcloud_camera_table": self._tensor_to_list(self.input_t_pointcloud_camera_table), # (N, 3)
            })
            df.to_parquet(path)

    def _tensor_to_list(self, input_tensor):
        # input_tensor is of shape (N, 4)
        # output is a list of length N, each element is an array of shape (4,)
        np_array = input_tensor.cpu().numpy()
        return [np_array[i] for i in range(np_array.shape[0])]

    def _get_all_camera_poses(self):
        q_pointcloud_camera_list = []
        t_pointcloud_camera_list = []
        num_objects = None
        for idx, row in self.df.iterrows():
            T_pointcloud_camera = self._pandas_field_to_tensor(
                row["T_pointcloud_camera"])
            if len(T_pointcloud_camera.shape) == 2:
                T_pointcloud_camera = T_pointcloud_camera.unsqueeze(0)
            
            # both q_pointcloud_camera and t_pointcloud_camera are of shape (K, 4), K is num of objects
            q_pointcloud_camera, t_pointcloud_camera = se3_to_quaternion_and_translation_torch(
                T_pointcloud_camera)
            num_objects = q_pointcloud_camera.shape[0]
            q_pointcloud_camera_list.append(q_pointcloud_camera)
            t_pointcloud_camera_list.append(t_pointcloud_camera)
        q_pointcloud_camera = torch.cat(q_pointcloud_camera_list, dim=0)
        t_pointcloud_camera = torch.cat(t_pointcloud_camera_list, dim=0)
        return q_pointcloud_camera, t_pointcloud_camera, num_objects

    def _pandas_field_to_tensor(self, field: Any) -> torch.Tensor:
        if isinstance(field, np.ndarray):
            return torch.from_numpy(field)
        elif isinstance(field, list):
            return torch.tensor(field)
        elif isinstance(field, torch.Tensor):
            return field