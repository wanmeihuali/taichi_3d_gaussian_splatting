import taichi as ti
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
import torch
import numpy as np
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.utils import se3_to_quaternion_and_translation_torch

RasterConifg = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig
def render(pts, pts_feat, c2w, intrin, HW):
    rasterisation = GaussianPointCloudRasterisation(
            config=RasterConifg(near_plane=0.4, far_plane=2000.0, depth_to_sort_key_scale=10.0, rgb_only=False),
        )
    camera_info = CameraInfo(camera_intrinsics=intrin.to(pts.device),camera_height=HW[0],camera_width=HW[1],
                             camera_id=0)  # TODO: caemra_id, does it matter
    q_pointcloud_camera, t_pointcloud_camera = se3_to_quaternion_and_translation_torch(c2w[None])
    gaussian_input = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
        point_cloud=pts.float(),
        point_cloud_features=pts_feat.cuda(),
        point_object_id=torch.zeros(pts.shape[0], dtype=torch.int32,device=pts.device),
        point_invalid_mask=torch.zeros(pts.shape[0], dtype=torch.int8,device=pts.device),
        camera_info=camera_info,
        q_pointcloud_camera=q_pointcloud_camera.cuda().contiguous(),
        t_pointcloud_camera=t_pointcloud_camera.cuda().contiguous(),
        color_max_sh_band=6,#TODO: check the number here, original it was iteration // self.config.increase_color_max_sh_band_interval
    )
    res = rasterisation(gaussian_input)
    return res
def plot3d(*data,fn='/d/del.html'):
    import plotly.graph_objs as go
    from plotly.offline import plot
    def plot_points_3d(pts):
        if torch.is_tensor(pts):
            pts = pts.detach().cpu().numpy()
        return go.Scatter3d( x=pts[:, 0], y=pts[:, 1], z=pts[:, 2] ,marker=dict(size=1),mode='markers')
    data = [plot_points_3d(_data) for _data in data]
    fig = go.Figure(data,layout={'scene': {'aspectmode': 'data'}})#, 'aspectratio': ar}})

    plot(fig,filename=fn, auto_open=False)
    return fig


if __name__ == '__main__':
    print('############ test depth grad ##############')
    ti.init(arch=ti.cuda, device_memory_GB=0.1)
    pts = torch.randn(10000,3,device='cuda')
    pts.requires_grad_()
    pts_feat = torch.zeros(pts.shape[0], 56,device=pts.device)
    pts_feat[:, 0:4] = torch.rand_like(pts_feat[:, 0:4])
    pts_feat[:, 4:7] = torch.randn(pts.shape[0],3,device=pts.device)*0.3-1 #size
    pts_feat[:, 7] =  0. # set high alpha
    pts_feat.requires_grad_(True)

    c2w = torch.eye(4,device='cuda')
    HW = [1080//64*16,1920//64*16]
    intrin = torch.Tensor([[100,0,200],[0,100,200],[0,0,1]]).cuda()
    iteration = 0
    import tqdm
    optimizer = torch.optim.Adam([pts],lr=0.01)
    for ii in tqdm.trange(1000):
        optimizer.zero_grad()
        res=render(pts, pts_feat, c2w, intrin, HW)
        mask = res[1]>0
        loss = (res[1]-3).abs()[mask].mean()
        loss.backward()
        optimizer.step()
        if ii%200==0:
            plot3d(pts)
            print(loss)

    print('############ test alpha grad from depth ##############')
    ti.init(arch=ti.cuda, device_memory_GB=0.1)
    pts = torch.randn(6000,3,device='cuda')
    pts[:,2]+=3
    pts.requires_grad_()
    pts_feat = torch.zeros(pts.shape[0], 56,device=pts.device)
    pts_feat[:, 0:4] = torch.rand_like( pts_feat[:, 0:4])
    pts_feat[:, 4:7] = torch.randn(pts.shape[0],3,device=pts.device)*0.3-1 #size
    pts_feat[:, 7] = -5 # Note: set alpha before sigmoid,  need to set small one, this is important
    pts_feat.requires_grad_(True)
   
    c2w = torch.eye(4,device='cuda')
    HW = [1080//64//1*16,1920//64//1*16]
    intrin = torch.Tensor([[50,0,200],[0,50,200],[0,0,1]]).cuda()
    import tqdm
    optimizer = torch.optim.Adam([pts_feat],lr=0.01)
    for ii in tqdm.trange(1000):
        optimizer.zero_grad()
        res=render(pts, pts_feat, c2w, intrin, HW)
        mask = res[-1]>0.5 # region with accumulated alpha>0.5
        loss = (res[1]-3).abs()[mask].mean()
        loss.backward()
        # only keep gradient of alpha
        pts_feat.grad.data[:,:7]=0
        pts_feat.grad.data[:,8:]=0
        optimizer.step()
        if ii%200==0:
            print(loss.item())
            fig=plot3d(pts[pts_feat[:,7].sigmoid()>0.5], pts[pts_feat[:,7].sigmoid()<=0.5],
                       fn='/d/del.html')

