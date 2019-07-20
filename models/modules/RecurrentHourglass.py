

import torch.nn as nn
import torch.nn.functional as F
from models.modules.ConvGRU import ConvGRU
from utils.train_utils import initialize_weights_kaiming


class RecurrentHourglass(nn.Module):
    def __init__(self, depth, hidden_channels, out_channels, device, block):
        super(RecurrentHourglass, self).__init__()

        self.depth = depth
        self.layers = []
        for i in range(depth):
            layer_out = out_channels if i == depth - 1 else hidden_channels
            layer = [block(hidden_channels, hidden_channels),
                     ConvGRU(hidden_channels, hidden_channels, 3, 1, device),
                     block(hidden_channels, layer_out)]
            self.layers.append(nn.ModuleList(layer))
        self.layers = nn.ModuleList(self.layers)
        self.apply(initialize_weights_kaiming)

    def recursive_forward(self, layer, x):
        x = F.max_pool2d(x, 2)
        x = self.layers[layer - 1][0](x)
        upper = self.layers[layer - 1][1](x)[-1]

        if layer == 0:
            out = self.layers[0][2](upper)
        else:
            out = self.layers[layer - 1][2](upper + self.recursive_forward(layer - 1, upper))

        out = F.upsample(out, scale_factor=2)
        return out

    def forward(self, x):
        return self.recursive_forward(self.depth, x)
