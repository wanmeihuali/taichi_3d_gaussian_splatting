# %%
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import trimesh
import numpy as np
import json
# %%
def convert_json(input_json, image_path_prefix):
    camera_intrinsics = np.array([
        [input_json["fl_x"], 0, input_json["cx"]],
        [0, input_json["fl_y"], input_json["cy"]],
        [0, 0, 1]])
    camera_width = input_json["w"]
    camera_height = input_json["h"]
    data_list = []
    for idx, frame in enumerate(input_json["frames"]):
        image_path = frame["file_path"]
        T_pointcloud_camera_blender = np.array(
            frame["transform_matrix"]).reshape(4, 4)
        T_pointcloud_camera = T_pointcloud_camera_blender
        flip_x = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        T_pointcloud_camera = T_pointcloud_camera_blender @ flip_x
        image_full_path = os.path.join(image_path_prefix, image_path)
        data = {
            "image_path": image_full_path,
            "T_pointcloud_camera": T_pointcloud_camera.tolist(),
            "camera_intrinsics": camera_intrinsics.tolist(),
            "camera_height": int(camera_height),
            "camera_width": int(camera_width),
            "camera_id": 0,
        }
        data_list.append(data)
    return data_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms_train", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--mesh_sample_points", type=int, default=500)
    parser.add_argument("--transforms_test", type=str, default=None, help="If not specified, sample from train set")
    parser.add_argument("--val_sample", type=int, default=8)
    parser.add_argument("--image_path_prefix", type=str, default="")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    input_json = json.load(open(args.transforms_train))
    data_list = convert_json(input_json, args.image_path_prefix)
    train_data_list = None
    val_data_list = None
    if args.transforms_test is not None:
        input_json = json.load(open(args.transforms_test))
        val_data_list = convert_json(input_json, args.image_path_prefix)
        train_data_list = data_list
    else:
        train_data_list = [data_list[i] for i in range(len(data_list)) if i % args.val_sample != 0]
        val_data_list = [data_list[i] for i in range(len(data_list)) if i % args.val_sample == 0]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, "train.json"), "w") as f:
        json.dump(train_data_list, f, indent=4)
    with open(os.path.join(args.output_path, "val.json"), "w") as f:
        json.dump(val_data_list, f, indent=4)
    mesh = trimesh.load(args.mesh_path)
    point_cloud, _ = trimesh.sample.sample_surface(mesh, count=args.mesh_sample_points)
    point_cloud_df = pd.DataFrame(point_cloud, columns=["x", "y", "z"])

    point_cloud_df.to_parquet(os.path.join(
        args.output_path, "point_cloud.parquet"))



    

        

