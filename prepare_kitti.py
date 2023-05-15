# %%
import pandas as pd
import os
import json
import argparse
import torch
# import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET


class KittiPreprocessor:
    class CameraView:
        def __init__(self, T_pointcloud_camera, label, sensor_id, path):
            self.T_pointcloud_camera = T_pointcloud_camera
            if isinstance(T_pointcloud_camera, np.ndarray):
                self.T_pointcloud_camera = torch.from_numpy(
                    T_pointcloud_camera)
            self.label = label
            self.sensor_id = sensor_id
            self.path = path

    class CameraInfo:
        def __init__(self, camera_intrinsic, camera_width, camera_height):
            self.camera_intrinsics = camera_intrinsic
            if isinstance(camera_intrinsic, np.ndarray):
                self.camera_intrinsics = torch.from_numpy(camera_intrinsic)
            self.camera_height = int(camera_height)
            self.camera_width = int(camera_width)

    def __init__(self, camera_view_path, camera_intrinsics_path, point_cloud_path, image_dir, use_cuda=True, output_dir=""):
        """
        The whole read process can be divided into two parts:
         - Render the point cloud into the camera view: the result is a depth map and a point cloud index map
         with the same size as the camera view.
         - Train a CNN network to get the encoding on each point in the point cloud, so that the CNN network can
         generate a image by the depth map and the point cloud index map.

         The CNN network can be trained using patch on the image;
         The downsampling of the camera view during rendering can be achieved by using a depth based pooling on the
         point cloud index map during the training process.

         Overall, it is totally not necessary to do the point cloud rendering during the training process. So we introduce
         this preprocessor to do the rendering and save the result to the disk. The CNN network can directly load the
         rendered result from the disk.

        :param camera_view_path:
        :param camera_intrinsics_path:
        :param point_cloud_path:
        """
        self.camera_view_list = self.extrinsics_from_xml(
            camera_view_path, image_dir)

        self.camera_info_dict = self.intrinsics_from_xml(
            camera_intrinsics_path)

        self.point_cloud = self.load_point_cloud(point_cloud_path)
        # also save the point cloud to the disk
        point_cloud_df = pd.DataFrame(
            self.point_cloud.numpy(), columns=["x", "y", "z"])
        point_cloud_df.to_parquet(os.path.join(
            output_dir, "point_cloud.parquet"))
        data = {
            "image_path": [camera_view.path for camera_view in self.camera_view_list],
            "T_pointcloud_camera": [camera_view.T_pointcloud_camera.numpy() for camera_view in self.camera_view_list],
            "camera_intrinsics": [self.camera_info_dict[camera_view.sensor_id].camera_intrinsics.numpy() for camera_view in self.camera_view_list],
            "camera_height": [self.camera_info_dict[camera_view.sensor_id].camera_height for camera_view in self.camera_view_list],
            "camera_width": [self.camera_info_dict[camera_view.sensor_id].camera_width for camera_view in self.camera_view_list],
            "camera_id": [camera_view.sensor_id for camera_view in self.camera_view_list],
        }
        df = pd.DataFrame(data)
        df.to_json(os.path.join(output_dir, "kitti.json"), orient="records")

    @staticmethod
    def extrinsics_from_xml(xml_file, image_dir, verbose=False):
        root = ET.parse(xml_file).getroot()
        camera_view_list = []
        for e in root.findall('chunk/cameras')[0].findall("camera"):
            label = e.get('label')
            sensor_id = e.get('sensor_id')
            try:
                T_pointcloud_camera_text = e.find("transform").text
                # remove /n in the text
                T_pointcloud_camera_text = T_pointcloud_camera_text.replace(
                    '\n', '')
                T_pointcloud_camera = np.array(
                    [float(x) for x in T_pointcloud_camera_text.split(' ') if len(x) > 0], dtype=np.float32).reshape(4, 4)
                image_name = f"{label}.png"
                image_path = os.path.join(image_dir, image_name)
                # find absolute path
                image_path = os.path.abspath(image_path)
                camera_view = KittiPreprocessor.CameraView(
                    T_pointcloud_camera, label, sensor_id, image_path)
                camera_view_list.append(camera_view)
            except Exception as e:
                print("Error parsing camera extrinsics for camera {}".format(label))
                print(e)
                continue
        camera_view_list.sort(key=lambda x: x.label)
        return camera_view_list

    @staticmethod
    def intrinsics_from_xml(xml_file):
        root = ET.parse(xml_file).getroot()
        sensors = root.findall('chunk/sensors/sensor')
        intrinsics_dict = {}
        for sensor in sensors:
            sensor_id = sensor.get('id')
            calibration = sensor.find('calibration')
            resolution = calibration.find('resolution')
            width = float(resolution.get('width'))
            height = float(resolution.get('height'))
            f = float(calibration.find('f').text)
            cx = width / 2
            cy = height / 2
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            intrinsics_dict[sensor_id] = KittiPreprocessor.CameraInfo(
                K, width, height)
        return intrinsics_dict

    @staticmethod
    def load_point_cloud(point_cloud_path):
        point_cloud = PlyData.read(point_cloud_path)
        point_cloud = np.vstack(
            (point_cloud['vertex']['x'], point_cloud['vertex']['y'], point_cloud['vertex']['z'])).T
        point_cloud = torch.from_numpy(point_cloud).float()
        return point_cloud


# %%
camera_view_path = "data/kitti/kitti6_368_total/camera.xml"
camera_intrinsics_path = "data/kitti/kitti6_368_total/camera.xml"
point_cloud_path = "data/kitti/kitti6_368_total/pointcloud.ply"
image_path = "data/kitti/image_kitti6_all_368/"
kitti_preprocessor = KittiPreprocessor(
    camera_view_path, camera_intrinsics_path, point_cloud_path, image_path)

# %%
