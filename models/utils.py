# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np 


class HyperspectralWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models, flatten=True, conv_dropout=False):
        super(HyperspectralWrapper, self).__init__()
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


class SpectralConvolution(nn.Module):
    """
    Two layers of spectral convolutions with a max pool operation.
    """
    def __init__(self, hyperparams):
        super(SpectralConvolution, self).__init__()
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

class View(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)

def sam_(x, y, reduction='mean'):
    """
    Calculates the spectral angle between two batches of vectors.
    """
    x_norm = 1e-6 + torch.linalg.norm(x, dim=-1)
    y_norm = 1e-6 + torch.linalg.norm(y, dim=-1)
    dot_product = torch.bmm(x.unsqueeze(1), y.unsqueeze(-1))
    prod = dot_product.view(-1)/(x_norm*y_norm)
    prod = torch.clamp(prod, 1e-6, 1-1e-6)
    assert all(prod >= torch.zeros_like(prod)) and all(prod <= torch.ones_like(prod)), "Out of [0,1]"
    sam = torch.acos(prod)
    assert not torch.isnan(sam.mean()), "SAM contains NaN"
    if reduction == 'mean':
        return sam.mean()
    elif reduction == 'none':
        return sam

def dist_sam(X, Y):
    dot_product = torch.mm(X, Y.T)
    x_norm = 1e-6 + torch.linalg.norm(X, dim=-1)
    y_norm = 1e-6 + torch.linalg.norm(Y, dim=-1)
    prod = dot_product/x_norm.unsqueeze(1)/y_norm.unsqueeze(0)
    prod = torch.clamp(prod, 1e-6, 1-1e-6)
    sam = torch.acos(prod)
    return sam


def Lr_(u, v, alpha=1, weights=None):
    if weights is None:
        mse = F.mse_loss(u, v)
        sam = sam_(u, v)
    else:
        mse = F.mse_loss(u, v, reduction='none')
        mse = torch.mean(mse, dim=-1)*weights
        sam = sam_(u, v, reduction='none')*weights
    return mse + alpha*sam

def one_hot(y, y_dim):
    """
    Returns labels in a one-hot format.
    """
    batch_size = len(y)
    one_hot = torch.zeros(batch_size, y_dim)
    one_hot[torch.arange(batch_size), y] = 1
    return one_hot

def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())