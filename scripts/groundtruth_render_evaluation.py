from decimal import MIN_EMIN
from tokenize import String
import numpy as np 
import open3d as o3d
import os
import pandas as pd
from matplotlib import pyplot as plt
from groundtruth_centroids_evaluation import point_cloud_groundtruth_comparison, get_pointcloud_dimension
import argparse
import string



def apply_transformation(groundtruth_file_path, pointcloud_reconstruction_path, transform_path, show_pointclouds=False, show_plots=False):
    
    # High quality reconstruction, 500 frames
    
    # Obtained aligning trajectory
    room_1_transform = np.array([[-0.07453163,  0.11889557, -0.4673894,  -1.59774687],
                            [ 0.48178008,  0.03977281 ,-0.06670892,  0.4059509 ],
                            [ 0.02184015 ,-0.47162058, -0.12345461, -0.21347023],
                            [ 0.    ,     0.  ,        0.  ,        1.        ]])
    room_1_transform = np.loadtxt(transform_path, delimiter=',', usecols=range(4))

    pc_reconstruction = o3d.io.read_point_cloud(pointcloud_reconstruction_path)   
    
    pc_reconstruction.transform(room_1_transform)
        
    return pc_reconstruction

def align(target, source):
    """
    target: glued mesh
    source: mesh to align on target
    """
    threshold = 1
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0], 
                            [0.0, 0.0, 0.0, 1.0]])

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.000001,
                                        relative_rmse=0.000001,
                                        max_iteration=50000)
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp( source, target, threshold, trans_init, \
        o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    
    # Apply transformation
    source.transform(reg_p2p.transformation)
    return source

def main(transform_path: string, output_folder_path: string, groundtruth_path: string, reconstruction_folder_path: string):
    show_pointclouds = False
    show_plots = False
    groundtruth_file_path = groundtruth_path
    
    # Compare all reconstructed pointclouds in the folder
    path = reconstruction_folder_path
    fileList = os.listdir(path)
    
    completePointSet = None
    count = 0
    for pointcloud_reconstruction_path in fileList:
        if pointcloud_reconstruction_path.startswith('pointcloud_'):
            print(pointcloud_reconstruction_path)
            reconstruction_pcd_aligned = apply_transformation(groundtruth_file_path, os.path.join(path+pointcloud_reconstruction_path), transform_path, show_pointclouds, show_plots) 
            #Add transformed reconstruction pointcloud to completePointSet
            if count==0:
                completePointSet= np.asarray(reconstruction_pcd_aligned.points)
                count+=1
            else:
                pc_reconstruction_load = np.asarray(reconstruction_pcd_aligned.points)
                completePointSet = np.concatenate((completePointSet,pc_reconstruction_load), axis=0)
            print("completePointSet len :", len(completePointSet))
                
    pc_groundtruth = o3d.io.read_point_cloud(groundtruth_file_path)
    #Evaluate comprehensive pointcloud
    completePointSet_pcd = o3d.geometry.PointCloud()
    print(len(completePointSet_pcd.points))
    completePointSet_pcd.points =  o3d.utility.Vector3dVector(completePointSet)
    completePointSet_aligned = align(pc_groundtruth, completePointSet_pcd)
    # completePointSet_aligned = completePointSet_pcd
    
    # Visualize pointcloud ensemble
    pc_groundtruth.paint_uniform_color([0,0,1])
    completePointSet_pcd.paint_uniform_color([0.5,0.5,0])
    print("Compolete point set visual")
    o3d.visualization.draw_geometries([completePointSet_pcd, pc_groundtruth, ])
    
    # Save .ply after it's been aligned with ICP
    o3d.io.write_point_cloud(os.path.join(path, "completePointSet_aligned_icp.ply"), completePointSet_aligned)
    distances, chamfer_dist = point_cloud_groundtruth_comparison(pc_groundtruth, completePointSet_aligned, show_pointclouds, show_plots)
    
    max_extent = get_pointcloud_dimension(groundtruth_file_path)
    print("Max extent: ", max_extent)
    
    distances = np.array(distances).flatten()
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    print("Average distance:", avg_distance)
    print("Median distance:", median_distance)
    
    df = pd.DataFrame({"distances": distances}) # transform to a dataframe
    
    # Plot and save figures
    output_directory = output_folder_path
    print(output_directory)
    if not os.path.exists(output_directory):
        print("Creating output directory.....")
        os.makedirs(output_directory)
        
    ax1 = df.boxplot(return_type="axes") # BOXPLOT
    ax1.set_title("Boxplot")
    plt.savefig(os.path.join(output_directory, 'boxplot.png'))
    
    ax2 = df.plot(kind="hist", alpha=0.5, bins = 1000) # HISTOGRAM
    ax2.set_title("Histogram")
    plt.savefig(os.path.join(output_directory, 'histogram.png'))
    
    ax3 = df.plot(kind="line") # SERIE
    ax3.set_title("Line Plot")
    plt.savefig(os.path.join(output_directory, 'line_plot.png'))    
        
    with open(output_directory+'/readme.txt', 'w') as f:
        f.write("Reconstruction file directory "+path+"\n")
        f.write("Reconstruction file name completePointSet_aligned \n")
        f.write("Groundtruth file path "+groundtruth_file_path+"\n")
        f.write(f"Points in groundtruth file: {len(pc_groundtruth.points)}"+"\n")
        f.write(f"Points in reconstruction file: {len(completePointSet_pcd.points)}"+"\n")
        f.write(f"Chamfer distance: {chamfer_dist}"+"\n")
        f.write(f"Average distance: {avg_distance}\n")
        f.write(f"Median distance:{median_distance}\n")
        f.write(f"Max extent: {max_extent}\n")
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='File path')
    parser.add_argument('--transform_file_path', type=str, help='Path to .txt file containing transformation from rendered mesh to groundtruth mesh, in gt frame')
    parser.add_argument('--output_path', type=str, help='Path to output folder')
    parser.add_argument('--groundtruth_file_path', type=str, help='Path to groundtruth .pcd file')
    parser.add_argument('--reconstruction_folder_path', type=str, help='Path to directory containing reconstruction .pcd files')
    args = parser.parse_args()
    main(args.transform_file_path, args.output_path, args.groundtruth_file_path, args.reconstruction_folder_path)