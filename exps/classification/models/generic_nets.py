# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from models.consistent_dropout import *
import numpy as np


class DenseNeuralNetwork(nn.Module):
    def __init__(self, dims, dropout=0):
        super(DenseNeuralNetwork, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.y_dim = y_dim
        self.dense_1 = nn.Linear(x_dim, h_dim)
        self.dense_2 = nn.Linear(h_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = self.dropout(x)
        x = F.relu(self.dense_2(x))
        x = self.dropout(x)
        x = self.logits(x)
        return x

    def regularization(self, norm="L2"):
        L = 0
        for param in self.parameters():
            if norm == "L2":
                L += torch.linalg.norm(param, dim=-1).mean()
            elif norm == "L1":
                L += torch.sum(torch.abs(param), dim=-1).mean()
            else:
                raise NotImplementedError
        return L


class SpectralConvolutionNetwork(nn.Module):
    """
    Two layers of spectral convolutions with a max pool operation.
    """
    def __init__(self, hyperparams):
        super(SpectralConvolutionNetwork, self).__init__()
        [n_channels, n_planes1, n_planes2, kernel_size1, kernel_size2, pool_size] = hyperparams

        self.n_channels = n_channels
        self.conv1 = nn.Conv1d(1, n_planes1, kernel_size1, padding='same', padding_mode='replicate')
        self.conv2 = nn.Conv1d(n_planes1, n_planes2, kernel_size2, padding='same', padding_mode='replicate')
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x_ = x.unsqueeze(1)
        x_ = F.relu(self.conv1(x_))
        x_ = F.relu(self.conv2(x_))
        x_ = x_ + x.unsqueeze(1)
        x_ = F.relu(x_)
        x_ = self.pool(x_)
        return x_


class CnnWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models, flatten=True, conv_dropout=False):
        super(CnnWrapper, self).__init__()
        self.models = nn.ModuleDict(models)
        self.flatten = flatten
        self.conv_dropout = conv_dropout

    @property
    def out_channels(self):
        with torch.no_grad():
            n_channels = sum([model.n_channels for model in self.models.values()])
            x = torch.ones((2, n_channels))
            x = self.forward(x)
        return x.numel()//2

    def forward(self, x):
        if len(x.shape)>2:
            patch_size = x.shape[-1]
            x = x[:,0,:,patch_size//2, patch_size//2]
        z, B = {}, 0

        for model_id, model in self.models.items():
            z[model_id] = model(x[:, B:B+model.n_channels])
            B += model.n_channels

        keys = list(z.keys())
        if self.conv_dropout:
            dropout = torch.ones(len(keys))
            dropout[np.random.randint(len(z))] = 0
            out = torch.cat([z[keys[i]]*dropout[i] for i in range(len(z))], dim=-1)
        else:
            out = torch.cat([z[keys[i]] for i in range(len(z))], dim=-1)

        if self.flatten:
            out = out.view(out.shape[0], -1)
        return out


class View(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


def convolutional_latent_net(conv_params: Dict):
    """
    conv_params: Dict with values (n_channels, n_planes1, n_planes2, kernel_size1, kernel_size2, pool_size)
    """
    cnn = {}
    for i, params in conv_params.items():
        cnn[f'conv-{i}'] = SpectralConvolutionNetwork(params)
    net = CnnWrapper(cnn)
    return net


def dense_classes_net(latent_dim, config):
    classifier = DenseNeuralNetwork([latent_dim, config['h_dim'], config['n_classes']], dropout=config['dropout'])
    classifier = MCConsistentDropoutModule(classifier)
    return classifier