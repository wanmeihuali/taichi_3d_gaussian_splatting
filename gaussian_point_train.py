import matplotlib.pyplot as plt
import argparse
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer
import os
import shutil

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

    # Create log dir and paste config file
    os.makedirs(config.summary_writer_log_dir, exist_ok=True)
    file_name = os.path.basename(args.train_config)
    shutil.copy2(args.train_config, os.path.join(config.summary_writer_log_dir, file_name))
    
    trainer = GaussianPointCloudTrainer(config)
    trainer.train()
