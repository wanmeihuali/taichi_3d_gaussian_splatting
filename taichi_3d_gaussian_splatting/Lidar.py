import torch
from typing import Optional, Union
import numpy as np
import math
import open3d as o3d
from .utils import quaternion_to_rotation_matrix_torch, inverse_SE3

class Lidar:
    
    def __init__(
            self,
            point_cloud: Union[np.ndarray, torch.Tensor],
            transform_camera_lidar: Union[np.ndarray, torch.Tensor], 
        ):
        self.point_cloud = point_cloud
        self.transform_camera_lidar = transform_camera_lidar
        
    def lidar_points_to_camera(
        self,
        lidar_point_cloud, # lidar points in camera coordinates
        T_camera_world,
        projective_transform,
        image_size: tuple
    ):        

        if projective_transform.dtype != lidar_point_cloud.dtype:
            # Convert T_camera_world to the data type of lidar_point_cloud_homogeneous
            projective_transform = projective_transform.to(lidar_point_cloud.dtype)        
        uv1 = torch.matmul(projective_transform, torch.transpose(lidar_point_cloud,0,1)) / \
            lidar_point_cloud[:, 2]
        uv = uv1[:2, :]
        
        u = torch.floor(uv[0, :]).long()
        v = torch.floor(uv[1, :]).long()
    

        depth_map = torch.full((image_size[1], image_size[0]), -1.0, dtype=lidar_point_cloud.dtype,device="cuda")
        depth_map[v[:], u[:]] = lidar_point_cloud[:, 2]
        # depth_map = torch.flip(depth_map, [1])

        return depth_map
    
    def lidar_points_visible(self,
                             lidar_point_cloud, # in world frame
                            T_world_camera, 
                            projective_transform, 
                            image_size: tuple):
        # Transform 3D point to camera coordinates
        lidar_point_cloud_homogeneous = torch.hstack((
                lidar_point_cloud,torch.ones((lidar_point_cloud.shape[0], 1)).cuda()
            ))
        T_camera_world = inverse_SE3(T_world_camera)
        if T_camera_world.dtype != lidar_point_cloud_homogeneous.dtype:
            # Convert T_camera_world to the data type of lidar_point_cloud_homogeneous
            T_camera_world = T_camera_world.to(lidar_point_cloud_homogeneous.dtype)
        transformed_points = torch.matmul(T_camera_world.cuda(), torch.transpose(lidar_point_cloud_homogeneous,0,1))
                            
        transformed_points = transformed_points[:3, :]
        # Apply perspective projection
        if projective_transform.dtype != transformed_points.dtype:
            projective_transform = projective_transform.to(transformed_points.dtype)
        #normalized_point = transformed_points / transformed_points[2]
        normalized_point = torch.matmul(projective_transform, transformed_points) / transformed_points[2]
        
        # Check if the point is within image boundaries
        image_size = list(image_size)

        is_visible_x = (0 <= normalized_point[0]) & (normalized_point[0] < image_size[0]) & (transformed_points[2] > 0.001) & (transformed_points[2] < 1000.)
        is_visible_y = (0 <= normalized_point[1]) & (normalized_point[1] < image_size[1])  & (transformed_points[2] > 0.001) & (transformed_points[2] < 1000.)
        is_visible = is_visible_x & is_visible_y
        
        transformed_points = torch.transpose(transformed_points, 0, 1)
        visible_points = transformed_points[is_visible]
        
        # lidar = o3d.geometry.PointCloud()
        # lidar.points = o3d.utility.Vector3dVector(transformed_points.detach().cpu().numpy())
        # o3d.io.write_point_cloud("debug_camera_frame.ply", lidar)
        
        # lidar = o3d.geometry.PointCloud()
        # lidar.points = o3d.utility.Vector3dVector(visible_points.detach().cpu().numpy())
        # o3d.io.write_point_cloud("debug_visible_points.ply", lidar)
        
        return visible_points