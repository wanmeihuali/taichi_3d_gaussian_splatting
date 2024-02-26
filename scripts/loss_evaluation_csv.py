
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import re
import csv
import ruamel.yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

"""
    Generate .csv file containing the average losses for a selected number of tf events, for a set of experiments all saved in /media/scratch1/logs/replica/
"""
def main():
    root_dir = "/media/scratch1/logs/replica/"
    loss_tags = ["train/depth loss", "train/smooth_loss", "train/l1 loss", "train/ssim loss", "train/loss"]
    
    train_dirs = os.listdir(root_dir)
    pattern = re.compile(r'room_1_high_quality_500_frames_(\d+)')
    numbers = [int(match.group(1)) for element in train_dirs for match in [pattern.match(element)] if match]

    numbers = [i for i in range(1, 26) if i != 5]
    train_dirs = [f"room_1_high_quality_500_frames_{number}" for number in numbers]

    data = []
    for dir in train_dirs:
        print(os.path.join(root_dir, dir))
        event_file = [filename for filename in os.listdir(os.path.join(root_dir, dir)) if filename.startswith("events")][0]
        event_path = os.path.join(root_dir, dir, event_file)

        acc = EventAccumulator(event_path)
        acc.Reload()
        acc.Tags()

        selected_steps = list(range(25000, 30001))
        row = {"path": event_path}
        for tag in loss_tags:
            scalars = acc.Scalars(tag)         
            selected_scalars = [scalar for scalar in scalars if scalar.step in selected_steps]
            
            # Convert selected scalars to a DataFrame
            df = pd.DataFrame(selected_scalars)
            mean_value = df['value'].mean()
            row[tag] = mean_value 
        
        yaml = ruamel.yaml.YAML()
        with open(os.path.join(root_dir, dir, "replica_room_1_high_quality_500_frames.yaml"), 'r') as stream:
            train_config = yaml.load(stream)
            lambda_depth = train_config.get("loss_function_config", {}).get("lambda_depth_value")
            lambda_smooth = train_config.get("loss_function_config", {}).get("lambda_smooth_value")

            # Update your row dictionary with the obtained value
            row["lambda_depth"] = lambda_depth
            row["lambda_smooth"] = lambda_smooth
        
        data.append(row)
    # Save the data to a CSV file
    csv_file_path = "/home/mroncoroni/report.csv"
    with open(csv_file_path, 'w', newline='') as csv_file:
        fieldnames = ["path"] + loss_tags + ["lambda_depth", "lambda_smooth"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
def visualize_losses():
    df = pd.read_csv("/home/mroncoroni/report.csv")

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(14, 8))

    # Create a ScalarMappable to map normalized values to colors
    norm = Normalize(vmin=0, vmax=25)
    sm = ScalarMappable(cmap='tab20b', norm=norm)
    
    # Plot train/smooth_loss, train/depth loss
    for iteration, row in df.iterrows():  
        if iteration == 0:
            continue
        
        if iteration >= 5:
            iteration += 1
        color = sm.to_rgba(iteration)
        label = f'Smooth: {row["lambda_smooth"]} - Depth: {row["lambda_depth"]} - Iteration: {iteration}'   
        
        if iteration % 3 == 0:
            marker = "o"
        if iteration % 3 == 1:
            marker = "x"    
        if iteration % 3 == 2:
            marker = "v"
        
        axs[0].scatter(row['train/depth loss'], row['train/smooth_loss'], color=color, marker=marker, label=label)
        axs[0].set_title('Train/depth loss and Train/Smooth Loss')
        axs[0].legend(bbox_to_anchor=(1, 1.1))
        axs[0].set_xlabel("train/depth loss")
        axs[0].set_ylabel("train/smooth_loss")
        # Plot train/l1 loss
        axs[1].scatter(row['train/ssim loss'], row['train/l1 loss'], color=color, marker=marker, label=label)
        axs[1].set_title('Train/SSIM Loss and Train/L1 Loss')
        axs[1].set_xlabel("train/ ssim loss")
        axs[1].set_ylabel("train/l1 loss")
    # Adjust layout
    
    plt.tight_layout()

    # Show the plot
    plt.savefig('/home/mroncoroni/report.png')
    
if __name__ == '__main__':
    # main()
    visualize_losses()