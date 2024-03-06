import argparse
import json
import os
import re
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def add_lidar(json_path:str, lidar_folder_path:str, lidar_frequency:int, noisy_lidar_measurements:bool):
    lidar_files = os.listdir(lidar_folder_path)
    lidar_files = sorted(lidar_files, key=lambda x: x[0])
    
    noisy_lidar_files = [s for s in lidar_files if "noise" in s]
    noisy_lidar_files = sorted(noisy_lidar_files, key=lambda x: x[0])
    lidar_files = [s for s in lidar_files if "noise" not in s]
    lidar_files = sorted(lidar_files, key=lambda x: x[0])
    print(lidar_files)
    print(noisy_lidar_files)
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    for entry in data:
        image_path = entry.get("image_path", "")
        # Use regular expression to find the frame number
        match = re.search(r'frame(\d+).jpg$', image_path)
        if match:
            frame_number = int(match.group(1))
        lidar_path = None
        # if frame_number%lidar_frequency == 0:
        if not noisy_lidar_measurements:
            try:
                lidar_file = [s for s in lidar_files if f"_{frame_number+1}." in s][0]
            except:
                lidar_file = None
        else:
            try:
                lidar_file = [s for s in noisy_lidar_files if f"_{frame_number+1}_" in s][0]
            except:
                lidar_file = None
        
        if lidar_file:
            lidar_path =  os.path.join(lidar_folder_path, lidar_file)
            

        if lidar_path:
            lidar_to_camera_transform = np.eye(4)
            
            entry["lidar_path"] = lidar_path
            entry["T_camera_lidar"] = lidar_to_camera_transform.tolist()
        else:
            entry["lidar_path"] = None
            entry["T_camera_lidar"] = None
    
    json_object = json.dumps(data, indent=4)
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_train_path", type=str, required=True)
    parser.add_argument("--json_val_path", type=str, required=True)
    parser.add_argument("--json_test_path", type=str, required=False)
    parser.add_argument("--lidar_folder_path", type=str, required=True)
    parser.add_argument("--noisy_lidar_measurements", nargs='?', type=str2bool,
                        const='True', default='False',
                        help="Use noisy lidar measurements if true")
    parser.add_argument("--lidar_frequency", type=int,
                        required=True)
    
    args = parser.parse_args()
    
    add_lidar(args.json_train_path, args.lidar_folder_path, args.lidar_frequency, args.noisy_lidar_measurements)
    add_lidar(args.json_val_path, args.lidar_folder_path, args.lidar_frequency, args.noisy_lidar_measurements)