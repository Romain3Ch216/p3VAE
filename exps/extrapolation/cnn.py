# Source: https://github.com/n-takeishi/phys-vae

import torch
import torch.nn as nn

import utils


class CNN(nn.Module):
    """1D Convolutional Neural Network.
    """
    # def __init__(self, n_planes, kernel_size, out_channels=64):
    def __init__(self, in_channels, out_channels=64):
        super(CNN, self).__init__()

        # self.conv_1 = nn.Conv1d(in_channels=1, out_channels=n_planes, kernel_size=kernel_size)
        # self.conv_2 = nn.Conv1d(in_channels=n_planes, out_channels=2*n_planes, kernel_size=kernel_size)
        # self.conv_3 = nn.Conv1d(in_channels=2*n_planes, out_channels=1, kernel_size=kernel_size)
        # self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dense = nn.Linear(in_channels // 2 + 1, out_channels)

    def forward(self, x:torch.Tensor):
        out = torch.fft.rfft(x)
        out = torch.real(out)**2
        max_values, _ = torch.max(out, dim=1)
        out = out / max_values.view(-1, 1)
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()
        # out = nn.functional.relu(self.conv_1(x.unsqueeze(1)))
        # # out = self.maxpool(out)
        # out = nn.functional.relu(self.conv_2(out))
        # # out = self.maxpool(out)
        # out = nn.functional.relu(self.conv_3(out))

        out = nn.functional.leaky_relu(self.dense(out))
        return out



class CNN(nn.Module):
    """1D Convolutional Neural Network.
    """
    def __init__(self, n_planes, kernel_size, out_channels=64):
        super(CNN, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=n_planes, kernel_size=kernel_size)
        self.conv_2 = nn.Conv1d(in_channels=n_planes, out_channels=2*n_planes, kernel_size=kernel_size)
        self.conv_3 = nn.Conv1d(in_channels=2*n_planes, out_channels=1, kernel_size=kernel_size)

    def forward(self, x:torch.Tensor):
        out = nn.functional.relu(self.conv_1(x.unsqueeze(1)))
        out = nn.functional.relu(self.conv_2(out))
        out = nn.functional.relu(self.conv_3(out))
        return out