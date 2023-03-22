import pdb

import torch
import  torch.nn as nn
import torch.nn.functional as F


class FgUnet(nn.Module):
    def __init__(self, n_channels, n_classes, patch_size):
        super(FgUnet, self).__init__()

        self.n_channels = n_channels
        self.patch_size = patch_size

        self.conv_1_1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 7), stride=(1, 1, 3), padding=(1, 1, 0), groups=1)
        self.conv_1_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 7), stride=(1, 1, 3), padding=(1, 1, 0), groups=8)

        self.conv_1_3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 7), groups=16)
        self.conv_1_4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 7), groups=32)
        self.conv_1_5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 7), groups=64)

        self.conv_2_1 = nn.Conv3d(in_channels=1, out_channels=256, kernel_size=(1, 1, n_channels), groups=1)
        self.conv_2_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), groups=32)
        self.conv_2_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), groups=16)

        self.max_pool_1 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.max_pool_2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.upsample_1 = nn.Upsample(scale_factor=(2, 2, 2))
        self.upsample_2 = nn.Upsample(scale_factor=(2, 2, 1))

        self.tconv_1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 7), groups=32)
        self.tconv_2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 7), groups=16)
        self.tconv_3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 7), groups=16)
        self.tconv_4 = nn.ConvTranspose3d(in_channels=48, out_channels=24, kernel_size=(3, 3, 7), stride=(1, 1, 3), padding=(1, 1, 0), groups=24)
        self.tconv_5 = nn.ConvTranspose3d(in_channels=24, out_channels=1, kernel_size=(3, 3, 7), stride=(1, 1, 3), padding=(1, 1, 0), groups=1)

        self.conv_4 = nn.Conv3d(in_channels=48, out_channels=16, kernel_size=(1, 1, self.out_dim_1_()))

        self.conv_3 = nn.Conv2d(in_channels=self.out_dim_2(), out_channels=n_classes, kernel_size=(1, 1))

    def out_dim_1_(self):
        x = torch.ones((1, self.patch_size, self.patch_size, self.n_channels))
        x1, x2, z = self.convs_1(x)
        x = self.tconv(x1, x2, z)
        return x.shape[-1]

    def out_dim_2(self):
        x = torch.ones((1, self.patch_size, self.patch_size, self.n_channels))
        x1, x2, z = self.convs_1(x)
        x1 = self.tconv(x1, x2, z)
        x1 = self.conv_4(x1).squeeze(-1)
        x2 = self.convs_2(x)
        x = torch.cat((x1, x2), dim=1)
        return x.shape[1]


    def convs_1(self, x):
        x1 = x.unsqueeze(1)
        x1 = F.relu(self.conv_1_1(x1))
        x1 = F.relu(self.conv_1_2(x1))
        x2 = self.max_pool_1(x1)
        x2 = F.relu(self.conv_1_3(x2))
        x2 = F.relu(self.conv_1_4(x2))
        z = self.max_pool_2(x2)
        z = F.relu(self.conv_1_5(z))
        return x1, x2, z

    def convs_2(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv_2_1(x))
        x = x.squeeze(-1)
        x = F.relu(self.conv_2_2(x))
        x = F.relu(self.conv_2_3(x))
        return x

    def tconv(self, x1, x2, z):
        x = F.relu(self.tconv_1(z))
        x = self.upsample_1(x)
        x = torch.cat((x, x2), dim=1)
        x = F.relu(self.tconv_2(x))
        x = F.relu(self.tconv_3(x))
        x = self.upsample_2(x)
        x = torch.cat((x, x1), dim=1)
        return x

    def decode(self, x):
        x = F.relu(self.tconv_4(x))
        x = F.pad(x, (1, 1))
        x = torch.sigmoid(self.tconv_5(x))
        x = x.squeeze(1)
        return x

    def forward(self, x):
        x1, x2, z = self.convs_1(x)
        x1 = self.tconv(x1, x2, z)
        r = self.decode(x1)
        x2 = self.convs_2(x)
        x1 = self.conv_4(x1).squeeze(-1)
        x = torch.cat((x1, x2), dim=1)
        y = self.conv_3(x)
        return r, y

