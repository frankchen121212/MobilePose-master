
# Source: https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
# Implementation of the inverted residual block heavily used in the MobileNetV2 model

import torch.nn as nn
from utils.train_utils import initialize_weights_kaiming


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InvertedResidualBlock, self).__init__()
        scale = 2

        self.use_res_connect = in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * scale),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * scale, in_channels * scale, 3, 1, 1, groups=in_channels * scale, bias=False),
            nn.BatchNorm2d(in_channels * scale),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * scale, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.apply(initialize_weights_kaiming)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
