import matplotlib.pyplot as plt
import argparse
from taichi_3d_gaussian_splatting.GaussianPointTrainer import GaussianPointCloudTrainer
import os
import shutil
import numpy as np
import yaml

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
    
    # # Parameter sweep
    # lambda_depth_values = np.array([0, 0.05, 0.1, 0.15, 0.2]) 
    # lambda_smooth_values= np.array([0, 0.0005, 0.001, 0.003, 0.005])
    
    # summary_writer_log_dir_base = config.summary_writer_log_dir
    # count = 6
    # for i in range(1, len(lambda_depth_values)):
    #     for j in range(len(lambda_smooth_values)):
    #         config.output_model_dir = summary_writer_log_dir_base + f"_{count}"
    #         config.summary_writer_log_dir = summary_writer_log_dir_base + f"_{count}"
    #         config.loss_function_config.lambda_depth_value = float(lambda_depth_values[i])
    #         config.loss_function_config.lambda_smooth_value = float(lambda_smooth_values[j])
    #         output_dir = f"/media/scratch1/logs/replica/{os.path.basename(config.summary_writer_log_dir)}"
            
    #         # Save config as yaml file
    #         os.makedirs(config.summary_writer_log_dir, exist_ok=True)
    #         file=open(os.path.join(config.summary_writer_log_dir, "replica_room_1_high_quality_500_frames.yaml"),"w")
    #         yaml.dump(config,file)
    #         file.close()
            
    #         print("########################################################################")
    #         print("lambda depth: ", config.loss_function_config.lambda_depth_value)
    #         print("lambda smooth: ", config.loss_function_config.lambda_smooth_value)
    #         print("Output dir: ", output_dir)
    #         print("########################################################################")
            
    #         os.path.join(config.summary_writer_log_dir, "depth parameters.txt")
    #         trainer = GaussianPointCloudTrainer(config)
    #         trainer.train()
            
    #         print("###########################################################")
    #         print("Finished training")
    #         count+=1
    #         # memory issues: move to /media/scratch1            
    #         shutil.move(config.summary_writer_log_dir, output_dir)     
            
