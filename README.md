# taichi_3d_gaussian_splatting
An unofficial implementation of paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by taichi lang. 

## Current status
Working but not reaching the metric from paper. Now the repo can generate result for datasets such as tank and temple, and shows pretty good performance for small object dataset. However, the performance metric is still not a bit worse than the paper.

| Dataset | PSNR from paper | PSNR from this repo | SSIM from paper | SSIM from this repo | training time(RTX 3090) |  training time(T4) | #points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Truck(7k) | 23.51 | 22.84 | 0.840 | 0.798 | 10min50s | - | 350k |
| Truck(30k) | 25.187 | 24.25 | 0.879 | 0.8357 | 1h14min | - |682k |
| Truck(30k) less point | 25.187 | 24.15 | 0.879 | 0.824 | - | 1h35min |313k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train(30k) less point | 21.8 | 20.097 | 0.802 | 0.758 | - | 1h55min | 445k |

 The Rasterization part working well. For the Adaptive controller part, I'm pretty sure the implementation has some difference with the paper. The paper does not provide enough details about the Adaptive controller part. e.g. The view-space position gradient threshold is 0.0002 from the paper, but the current implementation only works with a much smaller value(4e-6). I also notice that the current threshold led to more points than expected(300k to 500k at 30k iteration). So if the controller can densify points more correctly, we shall reach the training speed claimed in the paper. I'm still trying to figure out the details. The current implementation is based on my understanding of the paper.
 
 As a personal project, the parameters are not tuned well. And the code is not well organized yet. I will try to improve the code quality and performance in the future. Feel free to open an issue if you have any questions, and PRs are welcome, especially for any performance improvement.

## example result
top left: [result from this repo(30k iteration)](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/blob/cf7c1428e8d26495a236726adf9546e4f2a9adb7/config/tat_truck_every_8_test.yaml), top right: ground truth, bottom left: normalized depth, bottom right: normalized num of points per pixel
![image](images/tat_truck_image5_val.png)
![image](images/tat_truck_image7_val.png)
![image](images/tat_truck_image14_val.png)

## Installation
Right now a install script/docker image is still not ready. User needs to install dependencies manually. The dependencies are:
```
taichi>=1.5.0
pytorch==2.0.0  # earlier version may also work
torchvision
numpy
pytorch_msssim
dataclass-wizard
pillow
pyyaml
pandas[parquet]>=2.0.0
scipy
argparse
tensorboard
```
All dependencies can be installed by pip. pytorch/tochvision can be installed by conda. The code is tested on Ubuntu 20.04.2 LTS with python 3.10.10. The hardware is RTX 3090 and CUDA 12.1. The code is not tested on other platforms, but it should work on other platforms with minor modifications.

## Prepare dataset
The algorithm requires point cloud for whole scene, camera parameters, and ground truth image. The point cloud is stored in parquet format. The camera parameters and ground truth image are stored in json format. The running config is stored in yaml format. A script to build dataset from colmap output is provided. It is also possible to build dataset from raw data.

### Build dataset from colmap
- Reconstruct using colmap: See https://colmap.github.io/tutorial.html. The image should be undistorted. Sparse reconstruction is usually enough.
- save as txt: the standard colmap txt output contains three files, cameras.txt, images.txt, points3D.txt
- transform the txt into json and parquet: see [this file](prepare_colmap.py) about how to prepare it.
- prepare config yaml: see [this file](config/tat_train.yaml) as an example
- run with the config.

### Build dataset from raw data
#### Point cloud
The input point cloud is stored in parquet format. The parquet file should have the following columns:
```
x: float32
y: float32
z: float32
```
The unit does not matter, but the unit should be consistent with camera parameters, and training parameters need to be adjusted accordingly. The point cloud center also does not matter. The parquet can be easily generated from numpy by:
```python
point_cloud_df = pd.DataFrame(point_cloud.T, columns=["x", "y", "z"])

point_cloud_df.to_parquet(os.path.join(
    output_dir, "point_cloud.parquet"))
```

#### Camera parameters
Two json file(for train and validation) is required.
```json
[
    {
        "image_path": "\/home\/kuangyuan\/hdd\/datasets\/nerf_gen\/test_1\/dataset_d3\/dataset_d3_train\/images_train\/COS_Camera.png",
        "T_pointcloud_camera": [
            [
                -0.7146853805,
                -0.5808342099,
                0.3896875978,
                -1.1690626144
            ],
            [
                -0.6994460821,
                0.5934892297,
                -0.3981780708,
                1.1945340633
            ],
            [
                0.000000052,
                -0.5571374893,
                -0.8304202557,
                2.4912610054
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ], # 4x4 matrix, the transformation matrix from camera coordinate to point cloud coordinate
        "camera_intrinsics": [
            [
                2666.6666666667,
                0.0,
                960.0
            ],
            [
                0.0,
                2666.6666666667,
                540.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ], # 3x3 matrix, the camera intrinsics matrix K
        "camera_height": 1080, # image height, in pixel
        "camera_width": 1920, # image width, in pixel
        "camera_id": 0 # camera id, not used
    },
    ...
]
```
The projection is done by the following formula:
```math
\begin{bmatrix}x'\\y'\\z'\\1\end{bmatrix} = T_{pointcloud\_camera}^{-1} \cdot \begin{bmatrix}x\\y\\z\\1\end{bmatrix}
```

```math
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix} = K \cdot \begin{bmatrix}
x' / z'\\
y' / z'\\
1
\end{bmatrix}
```
in which $u$ is the column index, $v$ is the row index, $x, y, z$ is the point directly from point cloud, $x', y', z'$ is the point after transformation, $K$ is the camera intrinsics matrix from the json, $T_{pointcloud\_camera}$ is the transformation matrix from camera coordinate to point cloud coordinate from the json.

So the camera system in the json is with x-axis pointing right, y-axis pointing down, z-axis pointing forward. The image coordinate system is the standard pytorch image coordinate system, with origin at top left corner, x-axis pointing right, y-axis pointing down.

#### Ground truth image
The ground truth image is stored in png format. The image should be in RGB format. The image should be the same size as the camera height and width in the json file.

### Running config
See [here](config/boots_super_sparse_config.yaml) for example. The config is in yaml format. Please update the path to point cloud and camera parameters in the config file.

## Run
```bash
python gaussian_point_train.py --train_config {path to config file}
```
The result is visualized in tensorboard. The tensorboard log is stored in the output directory specified in the config file.


## TODO
### Algorithm part
- [ ] Fix the adaptive controller part, something is wrong with the densify process, and the description in the paper is very vague. Further experiments are needed to figure out the correct/better implementation.
    - figure if the densify shall apply to all points, or only points in current frame.
    - figure what "average magnitude of view-space position gradients" means, is it average across frames, or average across pixel? 
    - ~figure the correct split policy. Where shall the location of new point be? Currently the location is the location before optimization. Will it be better to put it at foci of the original ellipsoid?~ use sampling of pdf for over-reconstruct, use position before optimization for under-reconstruct.
- [ ] Add result score/image in README.md
    - try same dataset in the paper.
    - fix issue in current blender plugin, and also make the plugin open source.

### Engineering part
- [ ] fix bug: crash when there's no point in camrea.
- [ ] Add a inference only framework to support adding/moving objects in the scene, scene merging, scene editing, etc.
- [ ] Add a install script/docker image
- [ ] Support batch training. Currently the code only supports single image training, and only uses small part of the GPU memory.
