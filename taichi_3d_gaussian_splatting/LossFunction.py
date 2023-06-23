import torch
import torch.nn as nn
from dataclasses import dataclass
from pytorch_msssim import ssim
from dataclass_wizard import YAMLWizard


class LossFunction(nn.Module):
    @dataclass
    class LossFunctionConfig(YAMLWizard):
        lambda_value: float = 0.2
        enable_regularization: bool = True
        regularization_weight: float = 2


    def __init__(self, config: LossFunctionConfig):
        super().__init__()
        self.config = config

    def forward(self, predicted_image, ground_truth_image, point_invalid_mask=None, pointcloud_features=None):
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
        L = (1 - self.config.lambda_value) * L1 + \
            self.config.lambda_value * LD_SSIM
        if pointcloud_features is not None and self.config.enable_regularization:
            regularization_loss = self._regularization_loss(point_invalid_mask, pointcloud_features)
            L = L + self.config.regularization_weight * regularization_loss
        return L, L1, LD_SSIM

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
        
        

