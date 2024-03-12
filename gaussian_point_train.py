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

    # # Create log dir and paste config file
    # os.makedirs(config.summary_writer_log_dir, exist_ok=True)
    # file_name = os.path.basename(args.train_config)
    # shutil.copy2(args.train_config, os.path.join(config.summary_writer_log_dir, file_name))
    
    # trainer = GaussianPointCloudTrainer(config)
    # trainer.train()
    
    # Noise Parameter sweep
    std_noise_q_values = np.array([0.0, 0.05, 0.1, 0.15]) 
    std_noise_t_values= np.array([0.0,  0.05, 0.1, 0.2, 0.3])
    
    q_t_noise_values = (np.array([0.0, 0.3]),
                        np.array([0.05, 0.0]), 
                        np.array([0.1, 0.0]),
                        np.array([0.15, 0.0]),
                        np.array([0.05, 0.05]), 
                        np.array([0.1, 0.2]))
    
    summary_writer_log_dir_base = config.summary_writer_log_dir

    for q_t_noise_std in q_t_noise_values:
        config.noise_std_q = float(q_t_noise_std[0])
        config.noise_std_t = float(q_t_noise_std[1])
        config.summary_writer_log_dir = summary_writer_log_dir_base + f"_q_{config.noise_std_q}_t_{config.noise_std_t}"
        config.output_model_dir = summary_writer_log_dir_base + f"_q_{config.noise_std_q}_t_{config.noise_std_t}"
        output_dir = f"/media/scratch1/mroncoroni/git/taichi_3d_gaussian_splatting/logs/replica_colmap/{os.path.basename(config.summary_writer_log_dir)}"
        
        # Save config as yaml file
        os.makedirs(config.summary_writer_log_dir, exist_ok=True)
        file=open(os.path.join(config.summary_writer_log_dir, "replica_room_1_high_quality_500_frames.yaml"),"w")
        yaml.dump(config,file)
        file.close()
        
        print("########################################################################")
        print("config.std_noise_q: ", config.noise_std_q)
        print("config.std_noise_t: ", config.noise_std_t)
        print("Output dir: ", config.summary_writer_log_dir)
        print("########################################################################")
        
        os.path.join(config.summary_writer_log_dir, "depth parameters.txt")
        trainer = GaussianPointCloudTrainer(config)
        trainer.train()
        
        print("###########################################################")
        print("Finished training")

        # memory issues: move to /media/scratch1            
        # shutil.move(config.summary_writer_log_dir, output_dir)     
        
