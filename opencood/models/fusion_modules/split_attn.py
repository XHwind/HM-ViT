import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (B, L, 1, 1, 3C)
        batch = x.size(0)
        cav_num = x.size(1)

        if self.radix > 1:
            # x: (B, L, 1, 3, C)
            x = x.view(batch,
                       cav_num,
                       self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=3)
            # B, 3LC
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim, num_windows=3):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * num_windows, bias=False)

        self.num_windows = num_windows

        self.rsoftmax = RadixSoftmax(num_windows, 1)

    def forward(self, window_list):
        # window list: [(B, L, H, W, C) * 3]
        # assert len(window_list) == 3, 'only 3 windows are supported'

        B, L = window_list[0].shape[0], window_list[0].shape[1]

        # global average pooling, B, L, H, W, C
        x_gap = sum(window_list)
        # B, L, 1, 1, C
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # B, L, 1, 1, C*num_window
        x_attn = self.fc2(x_gap)
        # B L 1 1 3C
        x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)
        out = 0
        for i in range(len(window_list)):
            start = i * self.input_dim
            end = (i + 1) * self.input_dim
            out += window_list[i] * x_attn[:, :, :, :, start: end]

        return out
