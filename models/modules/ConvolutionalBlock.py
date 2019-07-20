
import torch.nn as nn
import torch.nn.functional as F
from utils.train_utils import initialize_weights_kaiming


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)

        self.apply(initialize_weights_kaiming)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        return out
