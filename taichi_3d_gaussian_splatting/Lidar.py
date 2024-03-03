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
        fx = 400 # projective_transform[0, 0]
        fy = 400 #projective_transform[1, 1]
        cx = 360.0 #projective_transform[0, 2]
        cy = 202.5  #projective_transform[1, 2]

        print("T_camera_world")
        print(T_camera_world)
        print("Projective transform")
        print(projective_transform)
        print("Image size")
        print(image_size)
        image_size = list(image_size)
        # image_size[0] = int(cx*2)
        # image_size[1] = int(cy*2)
        # depth_map = torch.zeros((image_size[1],image_size[0]))
        # for i in range(lidar_point_cloud.shape[0]):
        #     v = (lidar_point_cloud[i,0]/ (-lidar_point_cloud[i,2])) * fx + cx
        #     u = (lidar_point_cloud[i,1]/ (-lidar_point_cloud[i,2])) * fy + cy
        #     if u>= 0 and u <image_size[0] and v>= 0 and v <image_size[1]:
        #         depth_map[math.floor(v), math.floor(u)] = -lidar_point_cloud[i,2]
        if projective_transform.dtype != lidar_point_cloud.dtype:
            # Convert T_camera_world to the data type of lidar_point_cloud_homogeneous
            projective_transform = projective_transform.to(lidar_point_cloud.dtype)        
        uv1 = torch.matmul(projective_transform, torch.transpose(lidar_point_cloud,0,1)) / \
            lidar_point_cloud[:, 2]
        uv = uv1[:2, :]
        print(uv)
        print(torch.max(uv[0,:]))
        print(torch.max(uv[1,:]))
        print(torch.min(uv[0,:]))
        print(torch.min(uv[1,:]))
        depth_map = torch.zeros((image_size[0],image_size[1]))
        for i in range(uv.shape[1]):
            u = (uv[1, i]) 
            v = (uv[0, i]) 
            # if u >= 0 and u < image_size[0] and v >= 0 and v < image_size[1]:
            #     depth_map[math.floor(v), math.floor(u)] = -lidar_point_cloud[i,2]
            depth_map[math.floor(v), math.floor(u)] = -lidar_point_cloud[i,2]
        print(torch.max(depth_map))
        print(torch.min(depth_map))
        depth_map = depth_map / torch.max(depth_map)
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
        print("image_size")
        print(image_size)
        # image_size[0] = 720
        # image_size[1] = 405
        is_visible_x = (0 <= normalized_point[0]) & (normalized_point[0] < image_size[0]) & (transformed_points[2] < 0.)
        is_visible_y = (0 <= normalized_point[1]) & (normalized_point[1] < image_size[1])  & (transformed_points[2] < 0.)
        is_visible = is_visible_x & is_visible_y
        
        transformed_points = torch.transpose(transformed_points, 0, 1)
        visible_points = transformed_points[is_visible]
        
        lidar = o3d.geometry.PointCloud()
        lidar.points = o3d.utility.Vector3dVector(transformed_points.detach().cpu().numpy())
        o3d.io.write_point_cloud("debug_camera_frame.ply", lidar)
        
        lidar = o3d.geometry.PointCloud()
        lidar.points = o3d.utility.Vector3dVector(visible_points.detach().cpu().numpy())
        o3d.io.write_point_cloud("debug_visible_points.ply", lidar)
        
        return visible_points