import torch
import torch.nn as nn
from dataclasses import dataclass
from pytorch_msssim import ssim
from dataclass_wizard import YAMLWizard
import cv2 as cv
import numpy as np

class LossFunction(nn.Module):
    @dataclass
    class LossFunctionConfig(YAMLWizard):
        lambda_value: float = 0.2
        lambda_depth_value: float = 5
        lambda_smooth_value: float = 0.02
        enable_regularization: bool = True
        regularization_weight: float = 2


    def __init__(self, config: LossFunctionConfig):
        super().__init__()
        self.config = config

    def forward(self, predicted_image, ground_truth_image, predicted_depth, ground_truth_depth, depth_mask, point_invalid_mask=None, pointcloud_features=None):
        """
        L = (1 − 𝜆)L1 + 𝜆LD-SSIM
        predicted_image: (B, C, H, W) or (C, H, W)
        ground_truth_image: (B, C, H, W) or (C, H, W)
        """
        if len(predicted_image.shape) == 3:
            predicted_image = predicted_image.unsqueeze(0)
        if len(ground_truth_image.shape) == 3:
            ground_truth_image = ground_truth_image.unsqueeze(0)
        L1 = torch.abs(predicted_image - ground_truth_image).mean()
        LD_SSIM = 1 - ssim(predicted_image, ground_truth_image,
                           data_range=1, size_average=True)
        
        masked_difference = torch.abs(predicted_depth - ground_truth_depth)[depth_mask]
        L_DEPTH = masked_difference.mean()
        if len(masked_difference) == 0:
            L_DEPTH = torch.tensor(0)  
        
        
        L_SMOOTH = self.smoothing_loss(ground_truth_image, predicted_depth)
        L = (1 - self.config.lambda_value) * L1 + \
            self.config.lambda_value * LD_SSIM  + \
            self.config.lambda_depth_value * L_DEPTH +\
            self.config.lambda_smooth_value * L_SMOOTH
        if pointcloud_features is not None and self.config.enable_regularization:
            regularization_loss = self._regularization_loss(point_invalid_mask, pointcloud_features)
            L = L + self.config.regularization_weight * regularization_loss
        return L, L1, LD_SSIM, L_DEPTH, L_SMOOTH
    
    def _regularization_loss(self, point_invalid_mask, pointcloud_features):
        """ add regularization loss to pointcloud_features, especially for s.
        exp(s) is the length of three-major axis of the ellipsoid. we don't want
        it to be too large. first we try L2 regularization.

        Args:
            pointcloud_features (_type_): _description_
        """
        s = pointcloud_features[point_invalid_mask == 0, 4:7]
        exp_s = torch.exp(s)
        regularization_loss = torch.norm(exp_s, dim=1).mean()
        return regularization_loss
        
    def smoothing_loss(self, gt_image_input, depth_image_input) -> torch.Tensor:
        # Permute image: CHANNELSxHxW -> HxWxCHANNELS
        gt_image = gt_image_input.squeeze()
        if len(gt_image.shape)>=3:
            gt_image = torch.permute(gt_image, (1, 2, 0))
        if torch.is_tensor(gt_image):
            gt_image = gt_image.cpu().numpy()

        if len(gt_image.shape)>=3:
            gt_image = cv.cvtColor(gt_image, cv.COLOR_BGR2GRAY)
        gt_image = (gt_image*255).astype(np.uint8)
        gt_edges = cv.Canny(gt_image,100,200)
        
        gt_edges = 1. - gt_edges # 0: pixel on edge 1: pixel not on edge
        gt_edges = torch.tensor(gt_edges, device='cuda')
        dx = torch.abs(depth_image_input[:, :-1] - depth_image_input[:, 1:])
        dy = torch.abs(depth_image_input[:-1, :] - depth_image_input[1:, :])

        dx_mask = gt_edges[ :, :-1] * gt_edges[:, 1:]
        dy_mask = gt_edges[ :-1, :] * gt_edges[ 1:, :]

        return (dx * dx_mask).mean() + (dy * dy_mask).mean()

