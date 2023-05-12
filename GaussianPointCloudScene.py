import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


class GaussianPointCloudScene(torch.nn.modules):

    @dataclass
    class PointCloudSceneConfig:
        num_of_features: int = 56

    def __init__(
        self,
        point_cloud: np.ndarray,
        config: PointCloudSceneConfig,
    ):
        super().__init__()
        assert len(point_cloud.shape) == 2, "point_cloud must be a 2D array"
        assert point_cloud.shape[1] == 3, "point_cloud must have 3 columns(x,y,z)"
        self.point_cloud = torch.from_numpy(point_cloud, dtype=torch.float32)
        self.config = config
        self.register_buffer("point_cloud", self.point_cloud)
        self.point_cloud_features = nn.Parameter(
            torch.randn(self.point_cloud.shape[0], self.config.num_of_features)
        )

    def forward(self):
        return self.point_cloud, self.point_cloud_features
