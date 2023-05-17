import torch
import torch.nn as nn
from dataclasses import dataclass
from pytorch_msssim import ssim
from dataclass_wizard import YAMLWizard


class LossFunction(nn.Module):
    @dataclass
    class LossFunctionConfig(YAMLWizard):
        lambda_value: float = 0.2

    def __init__(self, config: LossFunctionConfig):
        super().__init__()
        self.config = config

    def forward(self, predicted_image, ground_truth_image):
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
        return L, L1, LD_SSIM
