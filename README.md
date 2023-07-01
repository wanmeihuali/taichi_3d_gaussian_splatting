# taichi_3d_gaussian_splatting
An unofficial implementation of paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by taichi lang. 

## What does 3D Gaussian Splatting do?

### Training:
The algorithm takes image from multiple views, a sparse point cloud, and camera pose as input, use a differentiable rasterizer to train the point cloud, and output a dense point cloud with extra features(covariance, color information, etc.).

<img src="images/image_from_multi_views.png" alt="drawing" width="200"/>\
If we view the training process as module, it can be described as:
```mermaid
graph LR
    A[ImageFromMultiViews] --> B((Training))
    C[sparsePointCloud] --> B
    D[CameraPose] --> B
    B --> E[DensePointCloudWithExtraFeatures]
```

### Inference:
The algorithm takes the dense point cloud with extra features and any camera pose as input, use the same rasterizer to render the image from the camera pose.
```mermaid
graph LR
    C[DensePointCloudWithExtraFeatures] --> B((Inference))
    D[NewCameraPose] --> B
    B --> E[Image]
```
An example of inference result:

https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/assets/18469933/cc760693-636b-4157-ae85-33813f3da54d

Because the nice property of point cloud, the algorithm easily handles scene/object merging compared to other NeRF-like algorithms.

https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/assets/18469933/bc38a103-e435-4d35-9239-940e605b4552



<details><summary>other example result</summary>
<p>

top left: [result from this repo(30k iteration)](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/blob/cf7c1428e8d26495a236726adf9546e4f2a9adb7/config/tat_truck_every_8_test.yaml), top right: ground truth, bottom left: normalized depth, bottom right: normalized num of points per pixel
![image](images/tat_truck_image5_val.png)
![image](images/tat_truck_image7_val.png)
![image](images/tat_truck_image14_val.png)

</p>
</details>

## Why taichi?
- Taichi is a language for high-performance computing. It is designed to close the gap between the productivity-focused Python language and the performance- and parallelism-focused C++/CUDA languages. By using Taichi, the repo is pure Python, and achieves the same or even better performance compared to CUDA implementation. Also, the code is much easier to read and maintain.
- Taichi provides various backends, including CUDA, OpenGL, Metal, etc. We do plan to change the backend to support various platforms, but currently, the repo only supports CUDA backend.
- Taichi provides automatic differentiation, although the repo does not use it currently, it is a nice feature for future development. 

## Current status
Working but not reaching the metric from paper. Now the repo can generate result for datasets such as tank and temple, and shows pretty good performance for small object dataset. However, the performance metric is still a bit worse than the paper.

| Dataset | PSNR from paper | PSNR from this repo | SSIM from paper | SSIM from this repo | training time(RTX 3090) |  training time(T4) | #points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Truck(7k) | 23.51 | 22.84 | 0.840 | 0.798 | 10min50s | - | 350k |
| Truck(30k) | 25.187 | 24.25 | 0.879 | 0.8357 | 1h14min | - |682k |
| [Truck(30k) less point](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/pull/36#issuecomment-1603107339) | 25.187 | 24.15 | 0.879 | 0.824 | - | 1h40min |313k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train(30k) less point | 21.8 | 20.097 | 0.802 | 0.758 | - | 1h55min | 445k |

[garden(1927x840)](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/pull/49#issuecomment-1605892361)
| train:ssim | val:psnr | train:psnr | val:5kpsnr | val:7kssim | val:7kpsnr | train:5kpsnr | val:5kssim | train:5kssim | train:7kssim | val:ssim | train:7kpsnr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8359270095825195 | 26.33028221130371 | 26.563976287841797 | 24.334638595581055 | 0.7433000802993774 | 25.014371871948242 | 23.131235122680664 | 0.7052685618400574 | 0.6888041496276855 | 0.7371317744255066 | 0.8041130304336548 | 25.55982208251953 |


[Truck(30k)(recent best result)](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/pull/49#issuecomment-1605699569):
| val:5kpsnr | val:7kssim | train:7kpsnr | train:7kssim | val:7kpsnr | val:ssim | train:psnr | val:psnr | train:5kssim | train:ssim | train:5kpsnr | val:5kssim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22.275985717773438 | 0.8001110553741455 | 24.976150512695312 | 0.8154618144035339 | 22.946680068969727 | 0.8391811847686768 | 25.5905704498291 | 24.461441040039062 | 0.7955425977706909 | 0.8418866395950317 | 23.247669219970703 | 0.7754911780357361 |

## Installation
1. Prepare an environment contains pytorch and torchvision
2. clone the repo and cd into the directory.
3. run the following command
```
pip install -r requirements.txt
pip install -e .
```

All dependencies can be installed by pip. pytorch/tochvision can be installed by conda. The code is tested on Ubuntu 20.04.2 LTS with python 3.10.10. The hardware is RTX 3090 and CUDA 12.1. The code is not tested on other platforms, but it should work on other platforms with minor modifications.

## Dataset
The algorithm requires point cloud for whole scene, camera parameters, and ground truth image. The point cloud is stored in parquet format. The camera parameters and ground truth image are stored in json format. The running config is stored in yaml format. A script to build dataset from colmap output is provided. It is also possible to build dataset from raw data.
### Train on Tank and temple Truck scene
<details><summary>CLICK ME</summary>
<p>
**Disclaimer**: users are required to get permission from the original dataset provider. Any usage of the data must obey the license of the dataset owner.

The truck scene in [tank and temple](https://www.tanksandtemples.org/download/) dataset is the major dataset used to develop this repo. We use a downsampled version of images in most experiments. The camera poses and the sparse point cloud can be easily generated by colmap. The preprocessed image, pregenerated camera pose and point cloud for truck scene can be downloaded from this [link](https://drive.google.com/drive/folders/1ZhMSkm3YGfhtywII5Hik5YDdMzD3lZjX?usp=sharing
).

Please download the images into a folder named `image` and put it under the root directory of this repo. The camera poses and sparse point cloud should be put under `data/tat_truck_every_8_test`. The folder structure should be like this:
```
├── data
│   ├── tat_truck_every_8_test
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── point_cloud.parquet
├── image
│   ├── 000000.png
│   ├── 000001.png
```
the config file [config/tat_truck_every_8_test.yaml](config/tat_truck_every_8_test.yaml) is provided. The config file is used to specify the dataset path, the training parameters, and the network parameters. The config file is self-explanatory. The training can be started by running
```bash
python gaussian_point_train.py --train_config config/tat_truck_every_8_test.yaml
```
</p>
</details>


### Train on Example Object(boot)

<details><summary>CLICK ME</summary>
<p>
    
It is actually one random free mesh from [Internet](https://www.turbosquid.com/3d-models/3d-tactical-boots-1948918), I believe it is free to use. An modified version of an Open Source Blender Plugin is used to generate the camera pose, point cloud, and ground truth image. The plugin is not ready for public release yet. The preprocessed image, pregenerated camera pose and point cloud for boot scene can be downloaded from this [link](https://drive.google.com/drive/folders/1d14l9ewnyI7zCA6BxuQUWseQbIKyo3Jh?usp=sharing). Please download the images into a folder named `image` and put it under the root directory of this repo. The camera poses and sparse point cloud should be put under `data/boots_super_sparse`. The folder structure should be like this:
```
├── data
│   ├── boots_super_sparse
│   │   ├── boots_train.json
│   │   ├── boots_val.json
│   │   ├── point_cloud.parquet
├── image
│   ├── images_train
│   │   ├── COS_Camera.001.png
│   │   ├── COS_Camera.002.png
|   |   ├── ...
```
Note that because the image in this dataset has a higher resolution(1920x1080), training on it is actually slower than training on the truck scene.

</p>
</details>


### Train on dataset generated by colmap
<details><summary>CLICK ME</summary>
<p>
    
- Reconstruct using colmap: See https://colmap.github.io/tutorial.html. The image should be undistorted. Sparse reconstruction is usually enough.
- save as txt: the standard colmap txt output contains three files, cameras.txt, images.txt, points3D.txt
- transform the txt into json and parquet: see [this file](tools/prepare_colmap.py) about how to prepare it.
- prepare config yaml: see [this file](config/tat_train.yaml) as an example
- run with the config.

</p>
</details>

### Train on dataset generated by other methods
<details><summary>CLICK ME</summary>
<p>

see [this file](docs/RawDataFormat.md) about how to prepare the dataset.

</p>
</details>



 
## Run
```bash
python gaussian_point_train.py --train_config {path to config file}
```

The training process works in the following way:
```mermaid
stateDiagram-v2
    state WeightToTrain {
        sparsePointCloud
        pointCloudExtraFeatures
    }
    WeightToTrain --> Rasterizer: input
    cameraPose --> Rasterizer: input
    Rasterizer --> Loss: rasterized image
    ImageFromMultiViews --> Loss
    Loss --> Rasterizer: gradient
    Rasterizer --> WeightToTrain: gradient
```

The result is visualized in tensorboard. The tensorboard log is stored in the output directory specified in the config file. The trained point cloud with feature is also stored as parquet and the output directory is specified in the config file.

## Visualization
A simple visualizer is provided. The visualizer is implemented by Taichi GUI which limited the FPS to 60(If anyone knows how to change this limitation please ping me). The visualizer takes one or multiple parquet results. Example parquets can be downloaded [here](https://drive.google.com/file/d/12-kZZay8RFlDk7hJQysG_Cr4-oxDp37l/view?usp=sharing).
```bash
python3 visualizer --parquet_path_list <parquet_path_0> <parquet_path_1> ...
```
The visualizer merges multiple point clouds and displays them in the same scene.
- Press 0 to select all point clouds(default state).
- Press 1 to 9 to select one of the point clouds.
- When all point clouds are selected, use "WASD=-" to move the camera, and use "QE" to rotate by the y-axis, or drag the mouse to do free rotation.
- When only one of the point clouds is selected, use "WASD=-" to move the object/scene, and use "QE" to rotate the object/scene by the y-axis, or r drag the mouse to do free rotation by the center of the object.

## How to contribute/Use CI to train on cloud

I've enabled CI and cloud-based training now. The function is not very stable yet. It enables anyone to contribute to this repo even if you don't have a GPU.
Generally, the workflow is:
1. For any algorithm improvement, please create a new branch and make a pull request.
2. Please @wanmeihuali in the pull request, and I will check the code and add a label `need_experiment` or `need_experiment_garden` or `need_experiment_tat_truck` to the pull request.
3. The CI will automatically build the docker image and upload it to AWS ECR. Then the cloud-based training will be triggered. The training result will be uploaded to the pull request as a comment, e.g. [this PR](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting/pull/38). The dataset is generated by the default config of colmap. The training is on g4dn.xlarge Spot Instance(NVIDIA T4, a weaker GPU than 3090/A6000), the training usually takes 2-3 hours.
4. Now the best training result in README.md is manually updated. I will try to automate this process in the future.

The current implementation is based on my understanding of the paper, and it will have some difference from the paper/official implementation(they plan to release the code in the July). As a personal project, the parameters are not tuned well. I will try to improve performance in the future. Feel free to open an issue if you have any questions, and PRs are welcome, especially for any performance improvement.


## TODO
### Algorithm part
- [ ] Fix the adaptive controller part, something is wrong with the densify process, and the description in the paper is very vague. Further experiments are needed to figure out the correct/better implementation.
    - figure if the densify shall apply to all points, or only points in current frame.
    - figure what "average magnitude of view-space position gradients" means, is it average across frames, or average across pixel? 
    - ~figure the correct split policy. Where shall the location of new point be? Currently the location is the location before optimization. Will it be better to put it at foci of the original ellipsoid?~ use sampling of pdf for over-reconstruct, use position before optimization for under-reconstruct.
- [x] Add result score/image in README.md
    - try same dataset in the paper.
    - fix issue in current blender plugin, and also make the plugin open source.
- [ ] camera pose optimization: get the gradient of the camera pose, and optimize it during training.
- [ ] Dynamic Rigid Object support. The current implementation already supports multiple camera poses in one scene, so the movement of rigid objects shall be able to transform into the movement of the camera. Need to find some sfm solution that can provide an estimation of 6 DOF pose for different objects, and modify the dataset code to do the test.

### Engineering part
- [x] fix bug: crash when there's no point in camrea.
- [x] Add a inference only framework to support adding/moving objects in the scene, scene merging, scene editing, etc.
- [ ] Add a install script/docker image
- [ ] Support batch training. Currently the code only supports single image training, and only uses small part of the GPU memory.
- [ ] Implement radix sort/cumsum by Taichi instead of torch, torch-taichi tensor cast seems only available on CUDA device. If we want to switch to other device, we need to get rid of torch.
- [ ] Implement a Taichi only inference rasterizer which only use taichi field, and migrate to MacOS/Android/IOS.
