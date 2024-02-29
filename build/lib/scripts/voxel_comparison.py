import torch
import argparse
import open3d as o3d
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
    
def load_voxel_grid(file_path):
    # Load voxel grid from file using Open3D
    voxel_grid = o3d.io.read_voxel_grid(file_path)
    return voxel_grid

def process_gt_center(gt_center, reconstruction_centers, bound):
    within_bounds_mask = (
        (gt_center[0] - bound <= reconstruction_centers[:, 0]) & (reconstruction_centers[:, 0] <= gt_center[0] + bound) &
        (gt_center[1] - bound <= reconstruction_centers[:, 1]) & (reconstruction_centers[:, 1] <= gt_center[1] + bound) &
        (gt_center[2] - bound <= reconstruction_centers[:, 2]) & (reconstruction_centers[:, 2] <= gt_center[2] + bound)
    )

    # Process the centers within bounds
    reconstruction_centers_within_bounds = reconstruction_centers[within_bounds_mask].tolist()

    return reconstruction_centers_within_bounds

def compute_overlapping_voxels(voxel_grid1, voxel_grid2):
    voxel_size = voxel_grid1.voxel_size
    print(f"Voxel size: {voxel_size}")
    bound = voxel_size / 2
    # Extract centers from Open3D voxel grids
    gt_centers = torch.tensor([voxel_grid1.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxel_grid1.get_voxels()], dtype=torch.float32, device='cuda')
    reconstruction_centers = torch.tensor([voxel_grid2.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxel_grid2.get_voxels()], dtype=torch.float32, device='cuda')
    
    print("Fitting nearest neighbors")
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(reconstruction_centers.cpu())
    print("Evaluating neighbors")
    _, indices = nbrs.kneighbors(gt_centers.cpu())
    
    print("Computing overlapping centers")
    overlap_centers = reconstruction_centers[indices]
    overlap_centers = torch.tensor(overlap_centers, dtype=torch.float32, device='cuda')

    overlap_centers = torch.squeeze(overlap_centers)
    
    within_bounds_mask = (
        (gt_centers[:, 0] - bound <= overlap_centers[:, 0]) & (overlap_centers[:, 0] <= gt_centers[:, 0] + bound) &
        (gt_centers[:, 1] - bound <= overlap_centers[:, 1]) & (overlap_centers[:, 1] <= gt_centers[:, 1] + bound) &
        (gt_centers[:, 2] - bound <= overlap_centers[:, 2]) & (overlap_centers[:, 2] <= gt_centers[:, 2] + bound)
    )

    within_bounds_mask = within_bounds_mask.cpu().numpy()
    
    overlap_centers = overlap_centers.cpu().numpy()
    overlap_centers = overlap_centers[within_bounds_mask, :]
    
    intersection_point_cloud = o3d.geometry.PointCloud()
    intersection_point_cloud.points = o3d.utility.Vector3dVector(overlap_centers)
    output_directory = "/home/mroncoroni/git/evaluation_scripts/output/voxelization/voxelized_taichi3dgs_reconstruction"
    if not os.path.exists(output_directory):
        print("Creating output directory.....")
        os.makedirs(output_directory)
    o3d.io.write_point_cloud(os.path.join(output_directory, "overlap_room_1_high_quality_500_frames_max_smoothing.ply"), intersection_point_cloud)
    print(f"Writing overlapping pointcloud to {os.path.join(output_directory, 'overlap_room_1_high_quality_500_frames_max_smoothing.ply')}")
    print(overlap_centers)
    # Number of overlapping voxels
    num_overlapping_voxels = len(overlap_centers)

    return num_overlapping_voxels


def main(groundtruth_voxel_grid_path, reconstructed_voxel_grid_path, output_directory_path):
    print("Evaluating overlapping voxels")
    print(f"Ground truth voxel file path: {groundtruth_voxel_grid_path}")
    print(f"Reconstructed voxel file path: {reconstructed_voxel_grid_path}")
    
    # Load voxel grids
    groundtruth_voxel_grid = load_voxel_grid(groundtruth_voxel_grid_path)
    reconstructed_voxel_grid = load_voxel_grid(reconstructed_voxel_grid_path)
    
    # Compute the number of overlapping voxels
    num_overlapping_voxels = compute_overlapping_voxels(groundtruth_voxel_grid, reconstructed_voxel_grid)
    print(f"Number of overlapping voxels: {num_overlapping_voxels}")
    output_directory = output_directory_path
   
    with open(os.path.join(output_directory,"readme_voxel.txt"), 'w') as f:
        f.write(f"Number overlapping voxels (rendered voxels in GT): {num_overlapping_voxels}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth_voxel_grid_path", type=str, required=True)
    parser.add_argument("--reconstructed_voxel_grid_path", type=str, required=True)
    parser.add_argument("--output_directory_path", type=str, required=True)
    
    args = parser.parse_args()
    main(args.groundtruth_voxel_grid_path, args.reconstructed_voxel_grid_path, args.output_directory_path)