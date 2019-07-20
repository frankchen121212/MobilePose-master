
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.ConvLSTM import ConvLSTM
from utils.train_utils import initialize_weights_kaiming


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

    def forward(self, x_t):
        return self.process(x_t)


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

    def forward(self, x_t):
        return self.encode(x_t)


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

    def forward(self, h_t):
        return self.generate(h_t)


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
        return b, h, c


class LPM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, T=5):
        super(LPM, self).__init__()

        self.T = T

        self.process = Processor(in_channels, out_channels)
        self.stage = Stage(in_channels, hidden_channels, out_channels, device)

        self.apply(initialize_weights_kaiming)

        # Per http://proceedings.mlr.press/v37/jozefowicz15.pdf
        self.stage.lstm.f_x.bias.data.fill_(1.0)
        self.stage.lstm.f_h.bias.data.fill_(1.0)

    def forward(self, x_0, x_1, x_2, x_3, x_4, centers):
        centers = torch.unsqueeze(centers, 0)
        centers = F.avg_pool2d(centers, 9, stride=8)

        x_0 = torch.unsqueeze(x_0, 0)
        b = self.process(x_0)

        b, h, c = self.stage(x_0, b, None, None, centers)
        x_1 = torch.unsqueeze(x_1, 0)
        b, h, c = self.stage(x_1, b, h, c, centers)
        x_2 = torch.unsqueeze(x_2, 0)
        b, h, c = self.stage(x_2, b, h, c, centers)
        x_3 = torch.unsqueeze(x_3, 0)
        b, h, c = self.stage(x_3, b, h, c, centers)
        x_4 = torch.unsqueeze(x_4, 0)
        b, _, _ = self.stage(x_4, b, h, c, centers)

        return b
