import os
import numpy as np
import pandas as pd
import open3d as o3d
import argparse
from matplotlib import pyplot as plt
from groundtruth_centroids_evaluation import point_cloud_groundtruth_comparison, get_pointcloud_dimension


def apply_transformation(pointcloud_reconstruction_path: String, trasnform: np.array):
    global completePointSet
    global count
    
    # High quality reconstruction, 500 frames
    
    # Obtained manually
    room_1_translation =(-1.58048,0.396771,-0.227569)
    room_1_rotation = [-0.405482, 0.511259, 0.601951, -0.460276]
    room_1_scale =  0.498308
    
    # Load   
    room_1_transform = np.array([[-0.07453163,  0.11889557, -0.4673894,  -1.59774687],
                            [ 0.48178008,  0.03977281 ,-0.06670892,  0.4059509 ],
                            [ 0.02184015 ,-0.47162058, -0.12345461, -0.21347023],
                            [ 0.    ,     0.  ,        0.  ,        1.        ]])

    pc_reconstruction = o3d.io.read_point_cloud(pointcloud_reconstruction_path)   
    
    pc_reconstruction.transform(room_1_transform)
    
    #Add transformed reconstruction pointcloud to completePointSet
    if count==0:
        completePointSet= np.asarray(pc_reconstruction.points)
        print("completePointSet len :", len(completePointSet))
        count+=1
    else:
        pc_reconstruction_load = np.asarray(pc_reconstruction.points)
        completePointSet = np.concatenate((completePointSet,pc_reconstruction_load), axis=0)
        print("completePointSet len :", len(completePointSet))
        
    return pc_reconstruction

def main(groundtruth_voxel_grid_path, reconstructed_voxel_grid_path):
    # Convert ply to voxel
    mesh_gt =  o3d.io.read_point_cloud("/home/martina/master-thesis/groundtruth/replica/room_1/mesh_upsampled.ply")
    mesh_reconstructed =  o3d.io.read_point_cloud("/home/martina/master-thesis/taichi_3dgs_output/replica_colmap/room_1_high_quality_500_frames_max_smoothing/point_clouds/completePointSet_aligned_icp.ply")

    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mesh_gt,
                                                                voxel_size=0.01)
    voxel_grid_reconstruction = o3d.geometry.VoxelGrid.create_from_point_cloud(mesh_reconstructed,
                                                                voxel_size=0.01)

    o3d.io.write_voxel_grid("/home/martina/master-thesis/scripts_output/voxelization/voxelized_gt.ply", voxel_grid)
    o3d.io.write_voxel_grid("/home/martina/master-thesis/scripts_output/voxelization/voxelized_room_1_high_quality_500_frames_max_smoothing.ply", voxel_grid_reconstruction)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth_voxel_grid_path", type=str, required=True)
    parser.add_argument("--reconstructed_voxel_grid_path", type=str, required=True)
    
    args = parser.parse_args()
    main(args.groundtruth_voxel_grid_path, args.reconstructed_voxel_grid_path)