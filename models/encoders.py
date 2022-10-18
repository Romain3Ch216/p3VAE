# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from models.utils import View
import numpy as np

class PhysicsGuidedEncoder(nn.Module):
    """
    Physics guided encoder: 
      * p(z_phi) := Beta(z_phi|alpha, beta)
      * p(z_eta) := Dir(z_eta|gamma)
    """
    def conv_dim(self):
        x = torch.ones((1, 1, self.n_bands))
        x = self.conv(x)
        return x.numel()

    def __init__(self, dims, theta):
        super(PhysicsGuidedEncoder, self).__init__()
        [x_dim, y_dim, z_eta_dim, h_dim] = dims
        self.theta = theta

        self.n_bands = x_dim
        self.y_dim = y_dim
        self.z_eta_dim = z_eta_dim
        self.view = View()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, groups=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=9, groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, groups=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=4, kernel_size=5),
            nn.ReLU()
        )

        self.encoder_z_phi = nn.Sequential(
            nn.Linear(self.n_bands + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )

        self.encoder_z_eta = nn.Sequential(
            nn.Linear(self.conv_dim()+y_dim, h_dim),
            # nn.Linear(self.conv_dim()+y_dim+1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_eta_dim)
        )

    def kl_divergence(self, reduction='mean'):
        kld = torch.distributions.kl.kl_divergence(self.q_z_phi, self.p_z_phi) \
            + torch.distributions.kl.kl_divergence(self.q_z_eta, self.p_z_eta)
        if reduction == 'mean':
            return kld.mean()
        elif reduction == 'none':
            return kld

    def regularization(self):
        "Computes the entropy of z_eta samples"
        z_eta = self.q_z_eta.rsample()
        H = - torch.sum(z_eta * torch.log(z_eta + 1e-5), dim=1).mean()
        return H

    def forward(self, x, y):
        x_phi = self.encoder_z_phi(torch.cat([x, y], dim=-1))
        alpha = 1e-1 + F.relu(x_phi[:,0])
        beta = 1e-1 + F.relu(x_phi[:,1])

        x = self.conv(x.unsqueeze(1))
        x = x.view(x.shape[0], -1)
        x_eta = self.encoder_z_eta(torch.cat([x,y], dim=-1))
        gamma = 1e-3 + F.relu(x_eta)

        self.q_z_phi = torch.distributions.beta.Beta(alpha, beta)
        self.p_z_phi = torch.distributions.beta.Beta(torch.ones_like(alpha), ((1-np.cos(self.theta))/(1e-4+np.cos(self.theta)))*torch.ones_like(alpha))
        self.q_z_eta = torch.distributions.dirichlet.Dirichlet(gamma)
        self.p_z_eta = torch.distributions.dirichlet.Dirichlet(torch.ones_like(gamma))

        z_phi = self.q_z_phi.rsample()
        z_eta = self.q_z_eta.rsample()
        z = torch.cat((z_phi.unsqueeze(-1), z_eta), dim=-1)
        return z


class GaussianEncoder(nn.Module):
    """
    Conventional gaussian encoder: 
      * p(z) := N(z|mu, Sigma)
    """
    def conv_dim(self):
        x = torch.ones((1, 1, self.n_bands))
        x = self.conv(x)
        return x.numel()

    def __init__(self, dims):
        super(GaussianEncoder, self).__init__()
        [x_dim, y_dim, z_eta_dim, h_dim] = dims

        self.n_bands = x_dim
        self.y_dim = y_dim
        self.z_dim = z_eta_dim + 1
        self.view = View()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=11, groups=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=9, groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, groups=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=4, kernel_size=5),
            nn.ReLU()
        )

        self.encoder_z = nn.Sequential(
            nn.Linear(self.conv_dim()+y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.z_dim)
        )

        self.encoder_mu = nn.Sequential(
            nn.Linear(self.conv_dim() + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.z_dim)
            )

        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.conv_dim() + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.z_dim)
            )

    def kl_divergence(self, reduction='mean'):
        kld = torch.distributions.kl.kl_divergence(self.q_z, self.p_z) 
        if reduction == 'mean':
            return kld.mean()
        elif reduction == 'none':
            return torch.sum(kld, dim=1)

    def regularization(self):
        return 0.

    def forward(self, x, y):
        x = self.conv(x.unsqueeze(1))
        x = x.view(x.shape[0], -1)

        mu = self.encoder_mu(torch.cat([x, y], dim=1))
        logvar = self.encoder_logvar(torch.cat([x, y], dim=1))

        self.q_z = torch.distributions.normal.Normal(mu, torch.exp(0.5*logvar))
        self.p_z = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(logvar))

        z = self.q_z.rsample()
        return z
    
