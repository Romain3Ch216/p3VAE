# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score
from typing import Union
from itertools import cycle


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
    return mse + sam

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

def accuracy_metrics(y_true, y_pred) -> Union[float, float]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred, average='micro')
    return acc, f1score

def cycle_loader(labeled_data, unlabeled_data):
    return zip(cycle(labeled_data), unlabeled_data)

class View(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)

def NormalNLLLoss(x, mu, var):
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """
    logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
    nll = -(logli.sum(1).mean())
    return nll

def ceil_int(x: float) -> int:
    """
    Floor function.

    :param x: a float
    :return: the floor of the input
    """
    return int(np.ceil(x))


def _compute_number_of_patches(patch_size: int, image_size: int, overlapping: int) -> int:
    """
    Computes the minimum number of patches to cover a bigger image
    given the patch size and the overlapping between patches.

    :param patch_size: size of the patch
    :param image_size: size of the image
    :param overlapping: length of the overlapping
    :return: number of patches
    """
    return ceil_int(image_size * 1.0 / (patch_size - overlapping))


def _compute_float_overlapping(patch_size: int, image_size: int, n: int) -> float:
    """
        Computes the maximum overlapping between patches to cover a bigger image.

        :param patch_size: size of the patch
        :param image_size: size of the image
        :param n: number of patches
        :return: the overlapping
    """
    if n == 1:
        return 0
    else:
        return (patch_size * n - image_size) * 1.0 / (n - 1.0)


def patch_data_loader_from_image(image, patch_size):
        image_size = image.shape[0]
        assert image_size == image.shape[1], "Image should be a square"
        n_patches = _compute_number_of_patches(patch_size, image_size, overlapping=0)
        overlapping = _compute_float_overlapping(patch_size, image_size, n_patches)
        for i in range(n_patches):
            for j in range(n_patches):
                top = int(i * (patch_size - overlapping))
                left = int(j * (patch_size - overlapping))
                yield image[top: top + patch_size, left: left + patch_size], top, left


def data_loader_from_image(image, batch_size):
    image = image.view(-1, image.shape[-1])
    data = torch.utils.data.TensorDataset(image)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    return loader
