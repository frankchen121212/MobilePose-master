

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.train_utils import initialize_weights_kaiming
from .modules.RecurrentHourglass import RecurrentHourglass


class PretrainRecurrentStackedHourglass(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, block, T=10, depth=4):
        super(PretrainRecurrentStackedHourglass, self).__init__()

        self.T = T

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        self.res1 = block(64, 128)
        self.res2 = block(128, 128)
        self.res3 = block(128, hidden_channels)

        self.hg_0 = RecurrentHourglass(depth, hidden_channels + 1, out_channels, device, block)
        self.hg_t = RecurrentHourglass(depth, hidden_channels + out_channels + 1, out_channels, device, block)

        self.apply(initialize_weights_kaiming)

    def forward(self, x, centers):
        centers = F.avg_pool2d(centers, 4)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.res1(x)
        x = F.max_pool2d(x, 2)
        x = self.res2(x)
        x = self.res3(x)

        x_0 = torch.cat([x, centers], dim=1)
        b_1 = self.hg_0(x_0)

        beliefs = [b_1]
        b_t_1 = b_1

        for t in range(self.T):
            x_t = torch.cat([x, b_t_1, centers], dim=1)
            b_t = self.hg_t(x_t)
            beliefs.append(b_t)
            b_t_1 = b_t

        out = torch.stack(beliefs, 1)
        return out
