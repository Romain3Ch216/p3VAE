# Source: https://github.com/n-takeishi/phys-vae

import torch
import torch.nn as nn

import utils

class MLPNET(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(MLPNET, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class MLP(nn.Module):
    """Multi-layer perceptron.
    """
    def __init__(self, dims_all:list, activation:str,
                 dropout:float=-1.0, batchnorm:bool=False, actfun_output:bool=False, binary_output:bool=False):
        super(MLP, self).__init__()

        modules = []

        # from first to second-last layer
        for i in range(len(dims_all)-2):
            # fully-connected
            modules.append(nn.Linear(dims_all[i], dims_all[i+1]))
            # batch normalization if any
            if batchnorm:
                modules.append(nn.BatchNorm1d(dims_all[i+1]))
            # nonlinear activation
            modules.append(utils.actmodule(activation))
            # dropout if any
            if dropout>0.0:
                modules.append(nn.Dropout(p=dropout))

        # last layer
        modules.append(nn.Linear(dims_all[-2], dims_all[-1]))
        if actfun_output:
            modules.append(utils.actmodule(activation))
        if binary_output:
            modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*modules)
        self.dim_in = dims_all[0]


    def forward(self, x:torch.Tensor):
        out = self.net(x.view(-1, self.dim_in))
        return out
