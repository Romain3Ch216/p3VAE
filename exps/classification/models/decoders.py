# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class HybridDecoder(nn.Module):
    def __init__(self, dims, phy_params):
        super(HybridDecoder, self).__init__()
        [x_dim, y_dim, z_eta_dim, h_dim, n_bands_] = dims
        [E_dir, E_dif, theta] = phy_params

        self.E_dir = torch.from_numpy(E_dir)
        self.E_dif = torch.from_numpy(E_dif)
        self.theta = torch.tensor([theta])
        self.z_eta_dim = z_eta_dim

        self.decoder_1 = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
            )

        self.decoder_2 = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim*z_eta_dim),
            # nn.Tanh()
            nn.Sigmoid()
            )

    def rho_(self, y):
        x = self.decoder_1(y)
        v = self.decoder_2(y)
        v = v.view(v.shape[0], self.z_eta_dim, -1)
        x = x.unsqueeze(1).repeat(1, self.z_eta_dim, 1)
        # x = torch.sigmoid(x * (1+v))
        # x = x * (1+v)
        x = x*v
        return x

    def omega(self, z_phi):
        return z_phi + 0.2


    def forward(self, z, y):
        z_phi, z_eta = z[:,0], z[:,1:]
        z_phi = z_phi.unsqueeze(1)
        self.rho = self.rho_(y)
        self.sp = torch.sum(self.rho*z_eta.unsqueeze(-1), dim=1)
        self.E_dir, self.E_dif, self.theta = \
            self.E_dir.to(z_phi.device), self.E_dif.to(z_phi.device), self.theta.to(z_phi.device)
        theta = self.theta * torch.ones_like(z_phi)
        ratio = (z_phi*self.E_dir + self.omega(z_phi)*self.E_dif)/(torch.cos(theta)*self.E_dir + self.E_dif)
        x = ratio*self.sp
        return x


class PhysicsGuidedDecoder(nn.Module):
    def __init__(self, dims):
        super(PhysicsGuidedDecoder, self).__init__()
        [x_dim, y_dim, z_eta_dim, h_dim, n_bands_] = dims
        self.z_eta_dim = z_eta_dim

        self.decoder_1 = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
            )

        self.decoder_2 = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim*z_eta_dim),
            nn.Sigmoid()
            )

        self.decoder_3 = nn.Sequential(
            nn.Linear(x_dim+1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
            )

    def rho_(self, y):
        x = self.decoder_1(y)
        v = self.decoder_2(y)
        v = v.view(v.shape[0], self.z_eta_dim, -1)
        x = x.unsqueeze(1).repeat(1, self.z_eta_dim, 1)
        x = x*v
        return x

    def omega(self, z_phi):
        return z_phi + 0.2

    def forward(self, z, y):
        z_phi, z_eta = z[:,0], z[:,1:]
        z_phi = z_phi.unsqueeze(1)
        self.rho = self.rho_(y)
        self.sp = torch.sum(self.rho*z_eta.unsqueeze(-1), dim=1)
        x = self.decoder_3(torch.cat([z_phi, self.sp], dim=-1))
        return x


class GaussianDecoder(nn.Module):
    def __init__(self, dims):
        super(GaussianDecoder, self).__init__()
        [x_dim, y_dim, z_eta_dim, h_dim] = dims
        self.z_dim = z_eta_dim + 1

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim+y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
            )

    def forward(self, z, y):
        x = self.decoder(torch.cat([z, y], dim=1))
        self.rho = x
        return x
