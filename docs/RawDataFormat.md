## Point cloud
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

## Camera parameters
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

## Ground truth image
The ground truth image is stored in png format. The image should be in RGB format. The image should be the same size as the camera height and width in the json file.

## Running config
See [here](../config/boots_super_sparse_config.yaml) for example. The config is in yaml format. Please update the path to point cloud and camera parameters in the config file.

