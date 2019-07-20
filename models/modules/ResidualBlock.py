
import torch.nn as nn
from utils.train_utils import initialize_weights_kaiming


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.use_res_connect = in_channels == out_channels
        hidden_channels = out_channels // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, 1)

        self.apply(initialize_weights_kaiming)

    def forward(self, x):
        if not self.use_res_connect:
            return self.conv(x)
        elif self.downsample is not None:
            return self.downsample(x) + self.conv(x)
        else:
            return x + self.conv(x)
