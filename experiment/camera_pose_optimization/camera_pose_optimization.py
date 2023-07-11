import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer

DELTA_T_RANGE = 0.1
DELTA_ANGLE_RANGE = 0.01

def add_delta_to_se3(se3_matrix: np.ndarray):
    delta_t = np.random.uniform(-DELTA_T_RANGE, DELTA_T_RANGE, size=(3,))
    delta_angle = np.random.uniform(-DELTA_ANGLE_RANGE, DELTA_ANGLE_RANGE, size=(3,))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(delta_angle[0]), -np.sin(delta_angle[0])],
                   [0, np.sin(delta_angle[0]), np.cos(delta_angle[0])]])
    RY = np.array([[np.cos(delta_angle[1]), 0, np.sin(delta_angle[1])],
                     [0, 1, 0],
                     [-np.sin(delta_angle[1]), 0, np.cos(delta_angle[1])]])
    Rz = np.array([[np.cos(delta_angle[2]), -np.sin(delta_angle[2]), 0],
                   [np.sin(delta_angle[2]), np.cos(delta_angle[2]), 0],
                   [0, 0, 1]])
    delta_rotation = Rz @ RY @ Rx
    se3_matrix[:3, :3] = se3_matrix[:3, :3] @ delta_rotation
    se3_matrix[:3, 3] += delta_t
    return se3_matrix
    

if __name__ == "__main__":
    plt.switch_backend("agg")
    parser = argparse.ArgumentParser("Train a Gaussian Point Cloud Scene")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--gen_template_only",
                        action="store_true", default=False)
    args = parser.parse_args()
    if args.gen_template_only:
        config = GaussianPointCloudTrainer.TrainConfig()
        # convert config to yaml
        config.to_yaml_file(args.train_config)
        exit(0)
    config = GaussianPointCloudTrainer.TrainConfig.from_yaml_file(
        args.train_config)

    original_train_dataset_json_path = config.train_dataset_json_path

    df = pd.read_json(original_train_dataset_json_path, orient="records")
    df["T_pointcloud_camera"] = df["T_pointcloud_camera"].apply(lambda x: np.array(x).reshape(4, 4))
    df["T_pointcloud_camera_with_noise"] = df["T_pointcloud_camera"].apply(lambda x: add_delta_to_se3(x))
    # sample in row, select 20% of the data to add noise
    df["T_pointcloud_camera"] = df.apply(lambda x: x["T_pointcloud_camera_with_noise"] if np.random.rand() < 0.2 else x["T_pointcloud_camera"], axis=1)


    # save df to a temp json file
    df.to_json("/tmp/temp.json", orient="records")
    config.train_dataset_json_path = "/tmp/temp.json"
    
    trainer = GaussianPointCloudTrainer(config)
    trainer.train()
