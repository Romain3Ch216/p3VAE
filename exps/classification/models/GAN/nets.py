import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights using the He et al. (2015) policy.

# Basic generator that maps: noise + condition -> fake samples
class Generator(nn.Module):
    def __init__(self, noise_dim, c_dim, z_dim, h_dim, x_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # LeakyReLU is preferred to keep gradients flowing even for negative activations
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(noise_dim + z_dim + c_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(h_dim, x_dim),
            torch.nn.Sigmoid() # smooth [0,1] outputs
        )
        # self.apply(weight_init, seed)

    def forward(self, z):
        # Concatenate the noise and condition
        return self.generator(z)

# class ConditionalConv1d(nn.Conv1d):
#     def __init__(self, in_channels, out_channels, kernel_size, bias=True):
#         super(ConditionalConv1d, self).__init__(in_channels, out_channels, kernel_size, bias)
#
#     def forward(self, input, y):
#         weight = nn.Parameter(self.weight[y])
#         pdb.set_trace()
#         return F.conv1d(input, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#
# class Generator(nn.Module):
#     def __init__(self, noise_dim, c_dim, z_dim, h_dim, x_dim, n_bands):
#         super(Generator, self).__init__()
#         self.z_dim = z_dim
#         # LeakyReLU is preferred to keep gradients flowing even for negative activations
#
#         self.seg_generator = nn.ModuleDict(
#             (f'seg-{i}', nn.Sequential(
#                 nn.Linear(noise_dim + z_dim + c_dim, h_dim),
#                 # nn.Linear(h_dim, h_dim),
#                 nn.Linear(h_dim, n // 2 + 64))
#             ) for (i, n) in enumerate(n_bands))
#
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=31)
#         self.conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=17)
#         self.conv3 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=11)
#         self.conv4 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7)
#         self.conv5 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3)
#
#         self.pad = [True, True, False, False, True, True]
#         self.upsample = nn.Upsample(scale_factor=2)
#
#
#     def forward(self, z):
#         X = []
#         for dense, pad in zip(self.seg_generator.values(), self.pad):
#             x = F.leaky_relu(dense(z))
#             x = x.unsqueeze(1)
#             x = F.leaky_relu(self.conv1(x))
#             x = F.leaky_relu(self.conv2(x))
#             x = F.leaky_relu(self.conv3(x))
#             x = F.leaky_relu(self.conv4(x))
#             x = F.leaky_relu(self.conv5(x))
#             x = self.upsample(x)
#             if pad:
#                 x = F.pad(x, (1, 0), mode='reflect')
#             x = x.squeeze(1)
#             X.append(x)
#         x = torch.cat(X, dim=1)
#         return x


# Basic fully connected discriminator: sample -> -infty -- fake - 0 - real -- +infty
class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(Discriminator, self).__init__()

        self.fc1 = torch.nn.Linear(x_dim, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, h_dim)
        self.fc3 = torch.nn.Linear(h_dim, h_dim)
        self.fc4 = torch.nn.Linear(h_dim+1, 1)

    def encode(self, X):
        x = F.leaky_relu(self.fc1(X))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

    def forward(self, X):
        features = self.encode(X)
        smooth_feature = torch.sum((X[:, 1:] - X[:, :-1])**2, dim=-1).unsqueeze(1)
        logits = self.fc4(torch.cat((features, smooth_feature), dim=-1))
        return features, logits

# Basic fully connected classifier: sample -> class
# class QHeadSS(nn.Module):
#     def __init__(self, h_dim, c_dim):
#         super(QHeadSS, self).__init__()
#         self.y_dim = c_dim
#         self.discriminator = torch.nn.Sequential(
#             torch.nn.Linear(h_dim, h_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(h_dim, h_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(h_dim, c_dim)
#         )
#
#         # self.apply(weight_init, seed)
#
#     def forward(self, X):
#         return self.discriminator(X)

# class QHeadSS(nn.Module):
#     def __init__(self, latent_cnn, c_dim):
#         super(QHeadSS, self).__init__()
#         self.y_dim = c_dim
#         self.discriminator = torch.nn.Sequential(
#             torch.nn.Linear(h_dim, h_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(h_dim, h_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(h_dim, c_dim)
#         )
#
#         # self.apply(weight_init, seed)
#
#     def forward(self, X):
#         return self.discriminator(X)

class QHeadUS(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()

        self.dense_mu_1 = nn.Linear(h_dim, h_dim)
        self.dense_mu_2 = nn.Linear(h_dim, z_dim)
        self.dense_var_1 = nn.Linear(h_dim, h_dim)
        self.dense_var_2 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        mu = self.dense_mu_2(F.leaky_relu(self.dense_mu_1(x)))
        var = torch.exp(self.dense_var_2(F.leaky_relu(self.dense_var_1(x))))
        return mu, var
