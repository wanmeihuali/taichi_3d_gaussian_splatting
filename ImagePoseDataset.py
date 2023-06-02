import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from Camera import CameraInfo
from typing import Any


class ImagePoseDataset(torch.utils.data.Dataset):
    """
    A dataset that contains images and poses, and camera intrinsics.
    """

    def __init__(self, dataset_json_path: str):
        super().__init__()
        required_columns = ["image_path", "T_pointcloud_camera",
                            "camera_intrinsics", "camera_height", "camera_width", "camera_id"]
        self.df = pd.read_json(dataset_json_path, orient="records")
        for column in required_columns:
            assert column in self.df.columns, f"column {column} is not in the dataset"

    def __len__(self):
        # return 1 # for debugging
        return len(self.df)

    def _pandas_field_to_tensor(self, field: Any) -> torch.Tensor:
        if isinstance(field, np.ndarray):
            return torch.from_numpy(field)
        elif isinstance(field, list):
            return torch.tensor(field)
        elif isinstance(field, torch.Tensor):
            return field

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["image_path"]
        T_pointcloud_camera = self._pandas_field_to_tensor(
            self.df.iloc[idx]["T_pointcloud_camera"])
        camera_intrinsics = self._pandas_field_to_tensor(
            self.df.iloc[idx]["camera_intrinsics"])
        camera_height = self.df.iloc[idx]["camera_height"]
        camera_width = self.df.iloc[idx]["camera_width"]
        camera_id = self.df.iloc[idx]["camera_id"]
        image = PIL.Image.open(image_path)
        image = torchvision.transforms.functional.to_tensor(image)
        # we want image width and height to be always divisible by 16
        # so we crop the image
        camera_width = camera_width - camera_width % 16
        camera_height = camera_height - camera_height % 16
        image = image[:3, :camera_height, :camera_width].contiguous()
        camera_info = CameraInfo(
            camera_intrinsics=camera_intrinsics,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_id=camera_id,
        )
        return image, T_pointcloud_camera, camera_info
