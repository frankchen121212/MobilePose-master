
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.ConvLSTM import ConvLSTM
from utils.train_utils import initialize_weights_kaiming
import dsntnn


class Processor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Processor, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 512, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
            nn.Conv2d(512, out_channels, 1, padding=0),
        )

    def forward(self, x):
        b = self.process(x)
        b = dsntnn.flat_softmax(b)
        c = dsntnn.dsnt(b)
        return b, c


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, out_channels, 5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encode(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.generate = nn.Sequential(
            nn.Conv2d(in_channels, 128, 11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1, padding=0),
        )

    def forward(self, x):
        return self.generate(x)


class Stage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(Stage, self).__init__()

        lstm_size = hidden_channels + out_channels + 1

        self.encode = Encoder(in_channels, hidden_channels)
        self.lstm = ConvLSTM(lstm_size, 3, 1, True, device)
        self.generate = Generator(lstm_size, out_channels)

    def forward(self, x, b_prev, h_prev, c_prev, centers):
        f = self.encode(x)
        f = torch.cat([f, b_prev, centers], dim=1)
        h, c = self.lstm(f, h_prev, c_prev)
        b = self.generate(h)

        b = dsntnn.flat_softmax(b)
        coords = dsntnn.dsnt(b)

        return b, h, c, coords


class CoordinateLPM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, T=5):
        super(CoordinateLPM, self).__init__()

        self.T = T

        self.process = Processor(in_channels, out_channels)
        self.stage = Stage(in_channels, hidden_channels, out_channels, device)

        self.apply(initialize_weights_kaiming)

        # Per http://proceedings.mlr.press/v37/jozefowicz15.pdf
        self.stage.lstm.f_x.bias.data.fill_(1.0)
        self.stage.lstm.f_h.bias.data.fill_(1.0)

    def forward(self, x, centers):
        centers = F.avg_pool2d(centers, 9, stride=8)

        b_0, coords = self.process(x)
        beliefs = [b_0]
        coordinates = [coords]

        b_prev, h_prev, c_prev = b_0, None, None
        for t in range(self.T):
            b, h, c, coords = self.stage(x, b_prev, h_prev, c_prev, centers)
            b_prev, h_prev, c_prev = b, h, c
            beliefs.append(b)
            coordinates.append(coords)

        return torch.stack(beliefs, 1), torch.stack(coordinates, 1)
