import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from scipy.spatial.ckdtree import cKDTree
from typing import Optional


class GaussianPointCloudScene(torch.nn.modules):

    @dataclass
    class PointCloudSceneConfig:
        num_of_features: int = 56
        max_num_points_ratio: Optional[float] = None,

    def __init__(
        self,
        point_cloud: np.ndarray,
        config: PointCloudSceneConfig,
    ):
        super().__init__()
        assert len(point_cloud.shape) == 2, "point_cloud must be a 2D array"
        assert point_cloud.shape[1] == 3, "point_cloud must have 3 columns(x,y,z)"
        if config.max_num_points_ratio is not None:
            num_points = point_cloud.shape[0]
            max_num_points = int(num_points * config.max_num_points_ratio)
            assert max_num_points > num_points, "max_num_points_ratio should be greater than 1.0"
            point_cloud = np.concatenate(
                [point_cloud, np.zeros((max_num_points - num_points, 3))], axis=0)
        self.point_cloud = nn.Parameter(
            torch.from_numpy(point_cloud, dtype=torch.float32))
        self.config = config
        self.point_cloud_features = nn.Parameter(
            torch.zeros(self.point_cloud.shape[0], self.config.num_of_features)
        )
        self.register_buffer(
            "point_invalid_mask",
            torch.zeros(self.point_cloud.shape[0], dtype=torch.int8)
        )
        if config.max_num_points_ratio is not None:
            self.point_invalid_mask[num_points:] = 1

    def forward(self):
        return self.point_cloud, self.point_cloud_features

    def initialize(self):
        """
        estimate the initial covariance matrix as an isotropic Gaussian
        with axes equal to the mean of the distance to the closest three points.
        """
        valid_point_cloud_np = self.point_cloud[~(self.point_invalid_mask)].detach(
        ).cpu().numpy()  # shape: [num_points, 3]
        nearest_neighbor_tree = cKDTree(valid_point_cloud_np)
        nearest_three_neighbor_distance, _ = nearest_neighbor_tree.query(
            valid_point_cloud_np, k=3)
        initial_covariance = np.mean(nearest_three_neighbor_distance, axis=1)
        # s is log of the covariance, so we take log of the initial covariance
        self.point_cloud_features[~(self.point_invalid_mask), 4:7] = torch.from_numpy(
            np.log(initial_covariance), dtype=torch.float32)

        # for rotation quaternion(x,y,z,w), we set it to identity
        self.point_cloud_features[:, 3] = 1.0
        self.point_cloud_features[:, 0:3] = 0.0
        # for alpha before sigmoid, we set it to 0.0, so sigmoid(alpha) is 0.5
        self.point_cloud_features[:, 7] = 0.0
        # for color spherical harmonics factors, we set them to 0.5
        self.point_cloud_features[:, 8:56] = 0.5

    def to_parquet(self, path: str):
        point_cloud_df = pd.DataFrame(
            self.point_cloud.detach().cpu().numpy(), columns=["x", "y", "z"])
        feature_columns = [f"cov_q{i}" for i in range(4)] + \
            [f"cov_s{i}" for i in range(3)] + \
            [f"alpha{i}" for i in range(1)] + \
            [f"r_sh{i}" for i in range(16)] + \
            [f"g_sh{i}" for i in range(16)] + \
            [f"b_sh{i}" for i in range(16)]
        point_cloud_features_df = pd.DataFrame(
            self.point_cloud_features.detach().cpu().numpy(), columns=feature_columns)
        scene_df = pd.concat([point_cloud_df, point_cloud_features_df], axis=1)
        scene_df.to_parquet(path)

    @staticmethod
    def from_parquet(path: str, config=PointCloudSceneConfig()):
        scene_df = pd.read_parquet(path)
        point_cloud = scene_df[["x", "y", "z"]].to_numpy()
        feature_columns = [f"cov_q{i}" for i in range(4)] + \
            [f"cov_s{i}" for i in range(3)] + \
            [f"alpha{i}" for i in range(1)] + \
            [f"r_sh{i}" for i in range(16)] + \
            [f"g_sh{i}" for i in range(16)] + \
            [f"b_sh{i}" for i in range(16)]
        scene = GaussianPointCloudScene(
            point_cloud, config)
        if not set(feature_columns).issubset(set(scene_df.columns)):
            scene.initialize()
        else:
            valid_point_cloud_features = torch.from_numpy(
                scene_df[feature_columns].to_numpy())
            scene.point_cloud_features[~(
                scene.point_invalid_mask)] = valid_point_cloud_features
        return scene
