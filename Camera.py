import torch
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CameraInfo:
    camera_intrinsics: torch.Tensor  # 3x3 matrix
    camera_height: int  # height of the image
    camera_width: int  # width of the image
    camera_id: int  # camera id


@dataclass
class CameraView:
    camera_view_id: int  # camera view id
    # 4x4 SE(3) matrix, transforms points from the camera frame to the pointcloud frame
    T_pointcloud_camera: torch.Tensor
    camera_id: int  # camera id of the camera that took this view
    image_id: int  # image id of the image that was taken by this camera
    # timestamp of the image that was taken by this camera, if available, otherwise None. Unit: microseconds
    timestamp: Optional[int] = None


class CameraDatabase:
    def __init__(self):
        self.camera_info_dict = {}
        self.camera_view_dict = {}

    def add_camera_info(self, camera_info: CameraInfo):
        self.camera_info_dict[camera_info.camera_id] = camera_info

    def get_camera_info(self, camera_id: int) -> CameraInfo:
        return self.camera_info_dict[camera_id]

    def add_camera_view(self, camera_view: CameraView):
        self.camera_view_dict[camera_view.camera_view_id] = camera_view

    def get_camera_view_and_info(self, camera_view_id: int) -> CameraView:
        return self.camera_view_dict[camera_view_id], self.camera_info_dict[self.camera_view_dict[camera_view_id].camera_id]
