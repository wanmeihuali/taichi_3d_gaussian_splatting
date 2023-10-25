import argparse
import pandas as pd
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene

def save_ply(pointcloud):
    print(pointcloud.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True)
    args = parser.parse_args()
    scene = GaussianPointCloudScene.from_parquet(
        args.parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
    scene.to_ply(args.ply_path)