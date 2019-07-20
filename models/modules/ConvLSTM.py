
import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, hidden_size, kernel_size, pad, bias, device):
        super(ConvLSTM, self).__init__()

        self.device = device

        self.i_x = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)
        self.o_x = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)
        self.g_x = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)
        self.f_x = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)

        self.i_h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)
        self.o_h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)
        self.g_h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)
        self.f_h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=pad, bias=bias)

    def forward(self, x, h_prev, c_prev):
        if h_prev is None:
            h_prev = torch.zeros_like(x).cuda()
        if c_prev is None:
            c_prev = torch.zeros_like(x).cuda()

        i = torch.sigmoid(self.i_x(x) + self.i_h(h_prev))
        o = torch.sigmoid(self.o_x(x) + self.o_h(h_prev))
        g = torch.sigmoid(self.g_x(x) + self.g_h(h_prev))
        f = torch.sigmoid(self.f_x(x) + self.f_h(h_prev))

        c = i.mul(g).add(f.mul(c_prev))
        h = o.mul(torch.tanh(c))
        return h, c


