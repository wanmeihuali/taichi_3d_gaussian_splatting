import argparse
import json
import os
import re
import numpy as np

def add_groundtruth_camera_trajectory(json_path:str, groundtruth_json_path: str):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    with open(groundtruth_json_path, 'r') as json_file:
        groundtruth_data = json.load(json_file)

    groundtruth_dictionaries = [entry for entry in groundtruth_data["frames"]]

    for entry in data:
        image_path = entry.get("image_path", "")
        match = re.search(r'frame(\d+).jpg$', image_path)
        
        #gt_match = re.search(r'train/(\d+).png$', groundtruth_file_paths)
        if match:
            frame_number = int(match.group(1))
        index = next((i for i, d in enumerate(groundtruth_dictionaries) if d["file_path"] == f"train/{frame_number+1:04d}.png"), None)
        groundtruth_transform = np.array(groundtruth_dictionaries[index]["transform_matrix"])
        entry["T_pointcloud_camera"] = groundtruth_transform.tolist()
        
    json_object = json.dumps(data, indent=4)
    print("output")
    print(os.path.join(os.path.dirname(json_path),os.path.splitext(os.path.basename(json_path))[0])+"_groundtruth.json")
    with open(os.path.join(os.path.dirname(json_path),os.path.splitext(os.path.basename(json_path))[0])+"_groundtruth.json", "w") as outfile:
        outfile.write(json_object)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_train_path", type=str, required=True)
    parser.add_argument("--json_val_path", type=str, required=True)
    parser.add_argument("--json_test_path", type=str, required=False)
    parser.add_argument("--groundtruth_trajectory_path", type=str, required=True)
    
    args = parser.parse_args()
    
    add_groundtruth_camera_trajectory(args.json_train_path, args.groundtruth_trajectory_path)
    add_groundtruth_camera_trajectory(args.json_val_path, args.groundtruth_trajectory_path)