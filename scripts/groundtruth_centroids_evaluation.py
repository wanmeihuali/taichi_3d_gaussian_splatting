'''
Script to evaluate difference metrics between ground truth mesh and the Gaussian centroids
'''

from decimal import MIN_EMIN
from tokenize import String
import numpy as np 
import open3d as o3d
import os
import pandas as pd
from matplotlib import pyplot as plt

def point_cloud_groundtruth_comparison(pc_groundtruth, pc_reconstruction, show_pointclouds=False, show_plots=False):
    pc_groundtruth.paint_uniform_color([0,0,1])
    pc_reconstruction.paint_uniform_color([0.5,0.5,0])
    
    print(pc_reconstruction)
    
    # visualization
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    if show_pointclouds:
        downpcd = pc_groundtruth.voxel_down_sample(voxel_size=0.05)
        o3d.visualization.draw_geometries([downpcd, pc_reconstruction, axis, ]) #pc_selected
    # Calculate distances of pc_1 to pc_2. 
    dist_pc1_pc2 = pc_groundtruth.compute_point_cloud_distance(pc_reconstruction)

    # dist_pc1_pc2 is an Open3d object, we need to convert it to a numpy array to 
    # acess the data
    dist_pc1_pc2 = np.asarray(dist_pc1_pc2)

    if show_plots:
        # Boxplot, histogram and serie to visualize distances. 
        df = pd.DataFrame({"distances": dist_pc1_pc2}) # transform to a dataframe
        ax1 = df.boxplot(return_type="axes") # BOXPLOT
        ax2 = df.plot(kind="hist", alpha=0.5, bins = 1000) # HISTOGRAM
        ax3 = df.plot(kind="line") # SERIE
        plt.show()
    
    # Compute chamfer distance
    dist_pc2_pc1 = pc_reconstruction.compute_point_cloud_distance(pc_groundtruth)
    dist_pc2_pc1 = np.asarray(dist_pc2_pc1)
    print("len(pc_groundtruth.points): ", len(pc_groundtruth.points))
    print("len(pc_reconstruction.points): ", len(pc_reconstruction.points))
    chamfer = np.sum(dist_pc1_pc2)/len(pc_groundtruth.points)+ np.sum(dist_pc2_pc1)/len(pc_reconstruction.points)
    print("Chamfer distance: ", chamfer)
    
    return dist_pc1_pc2, chamfer,

def get_pointcloud_dimension(pointcloud_file_path):
    pc = o3d.io.read_point_cloud(pointcloud_file_path)
    bounding_box = pc.get_axis_aligned_bounding_box()
    extent = bounding_box.get_extent()  
    return np.max(extent)


def main():
    # Replica room_0 dataset
    show_pointclouds = True
    show_plots = True
    
    groundtruth_file_path = "output/replica/room_0/point_clouds/mesh.pcd"
    pointcloud_reconstruction_path = 'output/replica/room_0/gaussian_point_clouds/parquet.pcd'
    
    pc_groundtruth = o3d.io.read_point_cloud(groundtruth_file_path)
    pc_reconstruction = o3d.io.read_point_cloud(pointcloud_reconstruction_path)
    
    distances = point_cloud_groundtruth_comparison(pc_groundtruth, pc_reconstruction, show_pointclouds, show_plots)
    
    max_extent = get_pointcloud_dimension(groundtruth_file_path)
    print("Max extent: ", max_extent)
    
    distances = np.array(distances).flatten()
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    print("Average distance:", avg_distance)
    print("Median distance:", median_distance)
    
    df = pd.DataFrame({"distances": distances}) # transform to a dataframe
    # Some graphs
    ax1 = df.boxplot(return_type="axes") # BOXPLOT
    ax1.set_title("Boxplot")
    ax2 = df.plot(kind="hist", alpha=0.5, bins = 1000) # HISTOGRAM
    ax3 = df.plot(kind="line") # SERIE
    plt.show()
    
    # Boots dataset
    show_pointclouds = True
    show_plots = True
    groundtruth_file_path = "output/boots_super_sparse/point_clouds/mesh.pcd"
    gaussian_file_path = 'output/boots_super_sparse/gaussian_point_clouds/parquet.pcd'
    
    distances = point_cloud_groundtruth_comparison(groundtruth_file_path, gaussian_file_path, show_pointclouds, show_plots)
    
    max_extent = get_pointcloud_dimension(groundtruth_file_path)
    print("Max extent: ", max_extent)
    
    distances = np.array(distances).flatten()
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    print("Average distance:", avg_distance)
    print("Median distance:", median_distance)


if __name__ == '__main__':
    main()