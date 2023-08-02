import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from scipy.spatial.ckdtree import cKDTree
from typing import Optional, Union
from dataclass_wizard import YAMLWizard


class GaussianPointCloudScene(torch.nn.Module):

    @dataclass
    class PointCloudSceneConfig(YAMLWizard):
        num_of_features: int = 56
        max_num_points_ratio: Optional[float] = None
        add_sphere: bool = False
        sphere_radius_factor: float = 4.0
        num_points_sphere: int = 10000
        max_initial_covariance: Optional[float] = None
        initial_alpha: float = -2.0
        initial_covariance_ratio: float = 1.0

    def __init__(
        self,
        point_cloud: Union[np.ndarray, torch.Tensor],
        config: PointCloudSceneConfig,
        point_cloud_features: Optional[torch.Tensor] = None,
        point_object_id: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert len(point_cloud.shape) == 2, "point_cloud must be a 2D array"
        assert point_cloud.shape[1] == 3, "point_cloud must have 3 columns(x,y,z)"
        # convert point_cloud to float32
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

        if config.max_num_points_ratio is not None:
            num_points = point_cloud.shape[0]
            max_num_points = int(num_points * config.max_num_points_ratio)
            assert max_num_points > num_points, "max_num_points_ratio should be greater than 1.0"
            point_cloud = torch.cat(
                [point_cloud, torch.zeros((max_num_points - num_points, 3))], dim=0)
            if point_cloud_features is not None:
                point_cloud_features = torch.cat(
                    [point_cloud_features, torch.zeros((max_num_points - num_points, config.num_of_features))], dim=0)
        self.point_cloud = nn.Parameter(point_cloud)
        self.config = config
        if point_cloud_features is not None:
            self.point_cloud_features = nn.Parameter(point_cloud_features)
        else:
            self.point_cloud_features = nn.Parameter(
                torch.zeros(
                    self.point_cloud.shape[0], self.config.num_of_features)
            )
        self.register_buffer(
            "point_invalid_mask",
            torch.zeros(self.point_cloud.shape[0], dtype=torch.int8)
        )
        if point_object_id is None:
            point_object_id = torch.zeros(
                self.point_cloud.shape[0], dtype=torch.int32)
        self.register_buffer(
            "point_object_id",
            point_object_id
        )
        if config.max_num_points_ratio is not None:
            self.point_invalid_mask[num_points:] = 1

    def forward(self):
        return self.point_cloud, self.point_cloud_features

    def initialize(self, point_cloud_rgb: Optional[torch.Tensor] = None):
        with torch.no_grad():
            """
            estimate the initial covariance matrix as an isotropic Gaussian
            with axes equal to the mean of the distance to the closest three points.
            """
            valid_point_cloud_np = self.point_cloud[self.point_invalid_mask == 0].detach(
            ).cpu().numpy()  # shape: [num_points, 3]
            nearest_neighbor_tree = cKDTree(valid_point_cloud_np)
            nearest_three_neighbor_distance, _ = nearest_neighbor_tree.query(
                valid_point_cloud_np, k=3 + 1)
            initial_covariance = np.mean(nearest_three_neighbor_distance[:, 1:], axis=1) * \
                self.config.initial_covariance_ratio
            # clip the initial covariance to [1e-6, inf]
            initial_covariance = np.clip(
                initial_covariance, 1e-6, self.config.max_initial_covariance)
            # s is log of the covariance, so we take log of the initial covariance
            self.point_cloud_features[(self.point_invalid_mask == 0), 4:7] = torch.tensor(
                np.log(initial_covariance), dtype=torch.float32).unsqueeze(1)

            """
            # for rotation quaternion(x,y,z,w), we set it to identity
            self.point_cloud_features[:, 3] = 1.0
            self.point_cloud_features[:, 0:3] = 0.0
            """
            # for rotation quaternion(x,y,z,w), we set it to random normalized value
            self.point_cloud_features[:, 0:4] = torch.rand_like(
                self.point_cloud_features[:, 0:4])
            self.point_cloud_features[:, 0:4] = self.point_cloud_features[:, 0:4] / \
                torch.norm(
                    self.point_cloud_features[:, 0:4], dim=1, keepdim=True)

            # for alpha before sigmoid, we set it to 0.0, so sigmoid(alpha) is 0.5
            # self.point_cloud_features[:, 7] = 0.0
            self.point_cloud_features[:, 7] = self.config.initial_alpha
            # for color spherical harmonics factors, we set them to 0.5
            self.point_cloud_features[:, 8] = 1.0
            self.point_cloud_features[:, 9:24] = 0.0
            self.point_cloud_features[:, 24] = 1.0
            self.point_cloud_features[:, 25:40] = 0.0
            self.point_cloud_features[:, 40] = 1.0
            self.point_cloud_features[:, 41:56] = 0.0
            if point_cloud_rgb is not None:
                point_cloud_rgb = torch.tensor(
                    point_cloud_rgb, dtype=torch.float32, requires_grad=False, device=self.point_cloud_features.device)
                point_cloud_rgb = point_cloud_rgb / 255.0
                point_cloud_rgb = torch.clamp(point_cloud_rgb, 0.0, 0.99)
                c0 = 0.28209479177387814
                self.point_cloud_features[(self.point_invalid_mask == 0), 8] = \
                    self._logit(point_cloud_rgb[:, 0]) / c0
                self.point_cloud_features[(self.point_invalid_mask == 0), 24] = \
                    self._logit(point_cloud_rgb[:, 1]) / c0
                self.point_cloud_features[(self.point_invalid_mask == 0), 40] = \
                    self._logit(point_cloud_rgb[:, 2]) / c0

    def _logit(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1.0 - x))

    def to_parquet(self, path: str):
        valid_point_cloud = self.point_cloud[self.point_invalid_mask == 0]
        valid_point_cloud_features = self.point_cloud_features[self.point_invalid_mask == 0]
        point_cloud_df = pd.DataFrame(
            valid_point_cloud.detach().cpu().numpy(), columns=["x", "y", "z"])
        feature_columns = [f"cov_q{i}" for i in range(4)] + \
            [f"cov_s{i}" for i in range(3)] + \
            [f"alpha{i}" for i in range(1)] + \
            [f"r_sh{i}" for i in range(16)] + \
            [f"g_sh{i}" for i in range(16)] + \
            [f"b_sh{i}" for i in range(16)]
        point_cloud_features_df = pd.DataFrame(
            valid_point_cloud_features.detach().cpu().numpy(), columns=feature_columns)
        scene_df = pd.concat([point_cloud_df, point_cloud_features_df], axis=1)
        scene_df.to_parquet(path)

    @staticmethod
    def from_parquet(path: str, config=PointCloudSceneConfig()):
        scene_df = pd.read_parquet(path)
        feature_columns = [f"cov_q{i}" for i in range(4)] + \
            [f"cov_s{i}" for i in range(3)] + \
            [f"alpha{i}" for i in range(1)] + \
            [f"r_sh{i}" for i in range(16)] + \
            [f"g_sh{i}" for i in range(16)] + \
            [f"b_sh{i}" for i in range(16)]
        if config.add_sphere:
            scene_df = GaussianPointCloudScene._add_sphere(
                scene_df, config.sphere_radius_factor, config.num_points_sphere)

        df_has_color = "r" in scene_df.columns and "g" in scene_df.columns and "b" in scene_df.columns
        point_cloud = scene_df[["x", "y", "z"]].to_numpy()

        if not set(feature_columns).issubset(set(scene_df.columns)):
            scene = GaussianPointCloudScene(
                point_cloud, config)

            point_cloud_rgb = scene_df[["r", "g", "b"]
                                       ].to_numpy() if df_has_color else None
            scene.initialize(point_cloud_rgb=point_cloud_rgb)
        else:
            valid_point_cloud_features = torch.from_numpy(
                scene_df[feature_columns].to_numpy())
            scene = GaussianPointCloudScene(
                point_cloud, config, point_cloud_features=valid_point_cloud_features)
        return scene

    @staticmethod
    def _add_sphere(scene_df: pd.DataFrame, radius_factor: float, num_points: int):
        """ add a sphere to the scene, with radius equal to center to the farthest point * radius_factor

        Args:
            scene_df (pd.DataFrame): requires columns: x, y, z
            radius_factor (float): the radius of the sphere is equal to center to the farthest point * radius_factor
        """
        df_has_color = "r" in scene_df.columns and "g" in scene_df.columns and "b" in scene_df.columns
        x_min, x_max = scene_df["x"].min(), scene_df["x"].max()
        y_min, y_max = scene_df["y"].min(), scene_df["y"].max()
        z_min, z_max = scene_df["z"].min(), scene_df["z"].max()
        far_distance = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
        radius = far_distance * radius_factor
        # sample points on the sphere
        phi = 2.0 * np.pi * np.random.rand(num_points)
        theta = np.arccos(2.0 * np.random.rand(num_points) - 1.0)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        points = np.stack([x, y, z], axis=1)
        columns = ["x", "y", "z"]
        if df_has_color:
            rgb = np.ones((num_points, 3)) * (255 // 2)
            points = np.concatenate([points, rgb], axis=1)
            columns += ["r", "g", "b"]
        scene_df = pd.concat([scene_df, pd.DataFrame(points, columns=columns)])
        return scene_df
