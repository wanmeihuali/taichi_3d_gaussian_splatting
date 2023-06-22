# %%
import os
import pandas as pd
import json
import numpy as np
import argparse

# %%
parser = argparse.ArgumentParser("Prepare dataset for 3D Gaussian Splatting from COLMAP text output")
parser.add_argument("--base_path", type=str, required=True, help="Path to the COLMAP output folder, containing cameras.txt, images.txt, points3D.txt")
parser.add_argument("--image_path", type=str, required=True, help="Path to the COLMAP Image folder")
parser.add_argument("--test_image_list_path", type=str, default=None, help="Path to the test image list")
parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
args = parser.parse_args()
base_path = args.base_path
image_path = args.image_path
output_dir = args.output_dir
test_image_list_path = args.test_image_list_path
# %%
def read_images_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    # Skip the header lines
    lines = lines[4:]
    images = {}
    for i in range(0, len(lines), 2):
        fields = lines[i].split()
        image_id = int(fields[0])
        qvec = list(map(float, fields[1:5]))
        tvec = list(map(float, fields[5:8]))
        camera_id = int(fields[8])
        name = " ".join(fields[9:])  # 这里处理文件名中可能包含空格的情况
        images[name] = {'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id}
    return images

def parse_parameters_dict(row):
    params = row['params']
    model = row['model']
    if model == 'SIMPLE_RADIAL':
        return {'f': params[0], 'cx': params[1], 'cy': params[2], 'k1': params[3]}
    elif model == 'RADIAL':
        return {'f': params[0], 'cx': params[1], 'cy': params[2], 'k1': params[3], 'k2': params[4]}
    elif model == 'PINHOLE':
        return {'fx': params[0], 'fy': params[1], 'cx': params[2], 'cy': params[3]}
    else:
        return {'params': params}

def get_intrinsic_matrix(params):
    if 'f' in params:  # For SIMPLE_RADIAL and RADIAL models
        f = params['f']
        cx = params['cx']
        cy = params['cy']
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif 'fx' in params:  # For PINHOLE model
        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        return None

def read_cameras_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    # Skip the header lines
    lines = lines[3:]

    data = {}
    for line in lines:
        fields = line.split()
        camera_id = int(fields[0])
        model = fields[1]
        width = int(fields[2])
        height = int(fields[3])
        params = [float(x) for x in fields[4:]]

        data[camera_id] = {'model': model, 'width': width, 'height': height, 'params': params}

    df = pd.DataFrame.from_dict(data, orient='index')
    df['params_dict'] = df.apply(parse_parameters_dict, axis=1)
    df['K'] = df['params_dict'].apply(get_intrinsic_matrix)
    return df


def read_points3D_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    # Skip the header lines
    lines = lines[3:]

    data = {}
    for line in lines:
        fields = line.split()
        point3d_id = int(fields[0])
        x, y, z = map(float, fields[1:4])
        r, g, b = map(int, fields[4:7])
        error = float(fields[7])
        track = list(zip(map(int, fields[8::2]), map(int, fields[9::2])))

        data[point3d_id] = {'x': x, 'y': y, 'z': z, 'r': r, 'g': g, 'b': b, 'error': error, 'track': track}

    return pd.DataFrame.from_dict(data, orient='index')

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

images = read_images_txt(os.path.join(base_path, 'images.txt'))
cameras = read_cameras_txt(os.path.join(base_path, 'cameras.txt'))
points = read_points3D_txt(os.path.join(base_path, 'points3D.txt'))

point_cloud = points[['x', 'y', 'z']].values
point_cloud = point_cloud.T
point_cloud_color = points[['r', 'g', 'b']].values
point_cloud_color = point_cloud_color.T
print(point_cloud.shape)
print(point_cloud_color.shape)

data = []
idx = 0
for name, image in images.items():
    idx += 1
    camera = cameras.loc[int(image['camera_id'])]
    # Extract quaternion and translation vector
    qvec = np.array(image['qvec'])
    tvec = np.array(image['tvec'])
    # Convert quaternion to rotation matrix
    R = np.zeros((4, 4))
    R[:3, :3] = quaternion_to_rotation_matrix(qvec)
    R[:3, 3] = tvec
    R[3, 3] = 1.0
    T_pointcloud_camera = np.linalg.inv(R)
    K = camera['K']
    # Construct the JSON data
    image_full_path = os.path.join(image_path, name)
    data.append({
        'image_path': image_full_path,
        'T_pointcloud_camera': T_pointcloud_camera.tolist(),
        'camera_intrinsics': camera['K'].tolist(),
        'camera_height': camera['height'],
        'camera_width': camera['width'],
        'camera_id': camera.name,
    })
    """
    print(data[-1])
    import matplotlib.pyplot as plt
    image = plt.imread(image_full_path)

    point_cloud_camera = np.linalg.inv(T_pointcloud_camera) @ \
        np.concatenate([point_cloud, np.ones_like(point_cloud[:1, :])], axis=0)
    point_cloud_camera = point_cloud_camera[:3, :]
    point_cloud_depth = point_cloud_camera[2, :]
    # print(point_cloud_depth.mean())
    point_cloud_uv1 = (camera["K"] @ point_cloud_camera) / \
        point_cloud_camera[2, :]
    point_cloud_uv = point_cloud_uv1[:2, :].T
    plt.imshow(image)
    plt.scatter(point_cloud_uv[:, 0], point_cloud_uv[:, 1], s=1)
    # scatter with color
    # plt.scatter(point_cloud_uv[:, 0], point_cloud_uv[:, 1], s=1, c=point_cloud_color.T/255)
    plt.xlim([0, camera["width"]])
    plt.ylim([camera["height"], 0])
    plt.show()
    break
    """

df = pd.DataFrame(data)
if test_image_list_path is not None:
    with open(test_image_list_path, "r") as f:
        test_images = f.readlines()
        test_images = [x.strip() for x in test_images]

    df["is_train"] = df["image_path"].apply(lambda x: os.path.basename(x) not in test_images)
else:
    # taking every 8th photo for test,
    df["is_train"] = df.index % 8 != 0
    
# test_images = [f"00{idx}.png" for idx in range(175, 250)]
# select training data and validation data, have a val every 3 frames
train_df = df[df["is_train"]].copy()
val_df = df[~df["is_train"]].copy()
print(train_df.shape)
print(val_df.shape)
# %%
train_df.drop(columns=["is_train"], inplace=True)
val_df.drop(columns=["is_train"], inplace=True)
# df.to_json(os.path.join(output_dir, "kitti.json"), orient="records")
train_df.to_json(os.path.join(
    output_dir, "train.json"), orient="records")
val_df.to_json(os.path.join(output_dir, "val.json"), orient="records")
points.to_parquet(os.path.join(
    output_dir, "point_cloud.parquet"))





# %%
