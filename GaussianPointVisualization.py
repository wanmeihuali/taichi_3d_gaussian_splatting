# %%
import taichi as ti
import taichi.ui as tiui
from GaussianPointCloudRasterisation import GaussianPointCloudRasterisationInferenceOnly
from dataclasses import dataclass
import torch

class GaussianPointVisualization:
    @dataclass
    class GaussianPointVisualizationState:
        next_T_pointcloud_camera: torch.Tensor
        
    def __init__(self) -> None:
        pass

    def start(self):
        pass

    def _main_loop(self):
        pass

# %%
ti.init(ti.gpu)

# %%
res = (1920, 1080)
window = tiui.Window('Gaussian Point Cloud Visualization', res, vsync=True)
while window.running:
    window.show()
window.destroy()

# %%
