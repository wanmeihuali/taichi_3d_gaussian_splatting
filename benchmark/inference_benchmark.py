# %%
import numpy as np
from plyfile import PlyData
import taichi as ti
import torch
from torchvision.utils import save_image
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.ImagePoseDataset import ImagePoseDataset
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import torch2ti, SE3_to_quaternion_and_translation_torch, quaternion_rotate_torch, quaternion_multiply_torch, quaternion_conjugate_torch
# %%
ITERATIONS = 100
WARMUP_ITERATIONS = 1000
DEVICE = torch.device("cuda:0")
DATASET_JSON_PATH = "/home/kuangyuan/hdd/Development/taichi_3d_gaussian_splatting/data/tat_truck_every_8_original/train.json"
PARQUET_PATH = "/home/kuangyuan/hdd/Development/taichi_3d_gaussian_splatting/logs_new/tat_truck_every_8_origin_dataset/scene_30000.parquet"
USE_PLY = True
PLY_PATH = "/home/kuangyuan/hdd/Development/gaussian-splatting/output/dd7034aa-9/point_cloud/iteration_30000/point_cloud.ply"

plydata = PlyData.read(PLY_PATH)

xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

features_dc = np.zeros((xyz.shape[0], 3, 1))
features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

extra_f_names = [
    p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
max_sh_degree = 3
assert len(extra_f_names) == 3*(max_sh_degree + 1) ** 2 - 3
features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
for idx, attr_name in enumerate(extra_f_names):
    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
# Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
features_extra = features_extra.reshape(
    (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

scale_names = [
    p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
scales = np.zeros((xyz.shape[0], len(scale_names)))
for idx, attr_name in enumerate(scale_names):
    scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

rot_names = [
    p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
rots = np.zeros((xyz.shape[0], len(rot_names)))
for idx, attr_name in enumerate(rot_names):
    rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

print("rots.shape", rots.shape)
# wxyz -> xyzw
rots = np.roll(rots, shift=-1, axis=1)
# normalize quaternion
rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)

ti.init(arch=ti.cuda, print_kernel_llvm_ir_optimized=True)

if USE_PLY:
    point_cloud = torch.from_numpy(xyz).float()
    # rot, scale, opacities, features_dc[:, 0, 0], features_extra[:, 0], features_dc[:, 1, 0], features_extra[:, 1], features_dc[:, 2, 0], features_extra[:, 2]
    r = features_dc[:, 0, 0].reshape(-1, 1)
    g = features_dc[:, 1, 0].reshape(-1, 1)
    b = features_dc[:, 2, 0].reshape(-1, 1)
    r_extra = features_extra[:, 0, :].reshape(-1, (max_sh_degree + 1) ** 2 - 1)
    g_extra = features_extra[:, 1, :].reshape(-1, (max_sh_degree + 1) ** 2 - 1)
    b_extra = features_extra[:, 2, :].reshape(-1, (max_sh_degree + 1) ** 2 - 1)
    point_cloud_features_np = np.concatenate(
        [rots, scales, opacities, r, r_extra, g, g_extra, b, b_extra], axis=1)
    point_cloud_features = torch.from_numpy(point_cloud_features_np).float()
    scene = GaussianPointCloudScene(
        point_cloud=point_cloud,
        config=GaussianPointCloudScene.PointCloudSceneConfig(
            max_num_points_ratio=None),
        point_cloud_features=point_cloud_features
    )

else:
    scene = GaussianPointCloudScene.from_parquet(
        PARQUET_PATH, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))

scene = scene.to(DEVICE)
dataset = ImagePoseDataset(
    dataset_json_path=DATASET_JSON_PATH)


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=4)

data_loader_iter = cycle(data_loader)

rasteriser = GaussianPointCloudRasterisation(
    config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
        near_plane=0.8,
        far_plane=1000.,
        depth_to_sort_key_scale=100.))

print("Warming up...")
for _ in range(WARMUP_ITERATIONS):
    with torch.no_grad():
        _, q_pointcloud_camera, t_pointcloud_camera, camera_info = next(
            data_loader_iter)
        q_pointcloud_camera = q_pointcloud_camera.to(DEVICE).contiguous()
        t_pointcloud_camera = t_pointcloud_camera.to(DEVICE).contiguous()
        camera_info.camera_intrinsics = camera_info.camera_intrinsics.to(
            DEVICE).contiguous()
        image, _, _ = rasteriser(
            GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=scene.point_cloud.contiguous(),
                point_cloud_features=scene.point_cloud_features.contiguous(),
                point_invalid_mask=scene.point_invalid_mask,
                point_object_id=scene.point_object_id,
                camera_info=camera_info,
                q_pointcloud_camera=q_pointcloud_camera,
                t_pointcloud_camera=t_pointcloud_camera,
                color_max_sh_band=3,
            )
        )

print("Benchmarking...")
total_inference_time = 0.0
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for _ in range(ITERATIONS):
    with torch.no_grad():
        _, q_pointcloud_camera, t_pointcloud_camera, camera_info = next(
            data_loader_iter)
        q_pointcloud_camera = q_pointcloud_camera.to(DEVICE).contiguous()
        t_pointcloud_camera = t_pointcloud_camera.to(DEVICE).contiguous()
        camera_info.camera_intrinsics = camera_info.camera_intrinsics.to(
            DEVICE).contiguous()
        image, _, _ = rasteriser(
            GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                point_cloud=scene.point_cloud.contiguous(),
                point_cloud_features=scene.point_cloud_features.contiguous(),
                point_invalid_mask=scene.point_invalid_mask.contiguous(),
                point_object_id=scene.point_object_id.contiguous(),
                camera_info=camera_info,
                q_pointcloud_camera=q_pointcloud_camera,
                t_pointcloud_camera=t_pointcloud_camera,
                color_max_sh_band=3,
            )
        )
end_event.record()
torch.cuda.synchronize()
print("Inference time: {} ms".format(
    start_event.elapsed_time(end_event) / ITERATIONS))
print("FPS: {}".format(1000.0 / (start_event.elapsed_time(end_event) / ITERATIONS)))

# save one image for debugging
with torch.no_grad():
    _, q_pointcloud_camera, t_pointcloud_camera, camera_info = next(
        data_loader_iter)
    q_pointcloud_camera = q_pointcloud_camera.to(DEVICE).contiguous()
    t_pointcloud_camera = t_pointcloud_camera.to(DEVICE).contiguous()
    camera_info.camera_intrinsics = camera_info.camera_intrinsics.to(
        DEVICE).contiguous()
    image, _, _ = rasteriser(
        GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
            point_cloud=scene.point_cloud.contiguous(),
            point_cloud_features=scene.point_cloud_features.contiguous(),
            point_invalid_mask=scene.point_invalid_mask.contiguous(),
            point_object_id=scene.point_object_id.contiguous(),
            camera_info=camera_info,
            q_pointcloud_camera=q_pointcloud_camera,
            t_pointcloud_camera=t_pointcloud_camera,
            color_max_sh_band=3,
        )
    )
    print("image.shape", image.shape)
    # HWC -> CHW
    image = image.permute(2, 0, 1)
    save_image(image, "image.png")
