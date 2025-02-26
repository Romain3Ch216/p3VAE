# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

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
