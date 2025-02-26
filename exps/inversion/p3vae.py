import numpy as np
from utils import read_csv
from resample import fwhm2std, std2sfr, resample_from_wv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
import argparse


def kldiv_normal_normal(mean1:torch.Tensor, lnvar1:torch.Tensor, mean2:torch.Tensor, lnvar2:torch.Tensor):
    """
    KL divergence between normal distributions, KL( N(mean1, diag(exp(lnvar1))) || N(mean2, diag(exp(lnvar2))) )
    """
    if lnvar1.ndim==2 and lnvar2.ndim==2:
        return 0.5 * torch.sum((lnvar1-lnvar2).exp() - 1.0 + lnvar2 - lnvar1 + (mean2-mean1).pow(2)/lnvar2.exp(), dim=1)
    elif lnvar1.ndim==1 and lnvar2.ndim==1:
        d = mean1.shape[1]
        return 0.5 * (d*((lnvar1-lnvar2).exp() - 1.0 + lnvar2 - lnvar1) + torch.sum((mean2-mean1).pow(2), dim=1)/lnvar2.exp())
    else:
        raise ValueError()

class p3VAE(nn.Module):
    def __init__(self, input_dim, num_bands, zI_dim, CH4_absorption, theta, min_data, max_data, zE_mean, zE_var, zE_max, h_dim=256):
        super(p3VAE, self).__init__()
        self.CH4_absorption = torch.from_numpy(CH4_absorption).float()
        self.theta = torch.tensor([theta])
        self.min_data = min_data
        self.max_data = max_data
        self.zE_mean = zE_mean
        self.zE_var = zE_var
        self.zE_max = zE_max
        self.zI_dim = zI_dim

        self.enc_feat_zI = nn.Sequential(
            nn.Linear(num_bands[0], h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)          
        )

        self.enc_feat_zE = nn.Sequential(
            nn.Linear(num_bands[1] + zI_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)          
        )

        self.enc_zE = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2),
        )

        self.enc_zI_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, zI_dim)
        )

        self.enc_zI_lnvar = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, zI_dim)
        )
        
        self.dec = nn.Sequential(
            nn.Linear(zI_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
        )

    @staticmethod
    def draw_normal(z_stat):
        mean = z_stat['mean']
        lnvar = z_stat['lnvar']
        std = torch.exp(0.5*lnvar)
        eps = torch.randn_like(std) # reparametrization trick
        return mean + eps*std
    
    @staticmethod
    def draw_beta(z_stat):
        alpha = z_stat['alpha']
        beta = z_stat['beta']
        dist = torch.distributions.beta.Beta(alpha, beta)
        z = dist.rsample()
        return z
    
    
    def priors(self, n, device):
        prior_zE_stat = {'alpha': torch.ones(n,1,device=device),
                         'beta': torch.ones(n,1,device=device)
                         }
        
        prior_zI_stat = {'mean': torch.zeros(n,self.zI_dim,device=device),
                         'lnvar': torch.zeros(n,self.zI_dim,device=device)
                         }

        return prior_zE_stat, prior_zI_stat
    
    def kl_div(self, zE_stat, zI_stat, device):
        n = zE_stat['alpha'].shape[0]
        prior_zE_stat, prior_zI_stat = self.priors(n, device)
        q_zE = torch.distributions.beta.Beta(zE_stat['alpha'], zE_stat['beta'])
        p_zE = torch.distributions.beta.Beta(prior_zE_stat['alpha'], prior_zE_stat['beta'])
        kldiv_zE = torch.distributions.kl.kl_divergence(q_zE, p_zE)
        kldiv_zI = kldiv_normal_normal(zI_stat['mean'], zI_stat['lnvar'], prior_zI_stat['mean'], prior_zI_stat['lnvar'])
        return (kldiv_zE + kldiv_zI).mean()
    
    def encode_zI(self, data):
        features = self.enc_feat_zI(data)
        zI_stat = {'mean':self.enc_zI_mean(features), 'lnvar':self.enc_zI_lnvar(features)}
        return zI_stat
    
    def encode_zE(self, data, zI_stat):
        features = self.enc_feat_zE(torch.cat([data, zI_stat['mean']], dim=1))

        zE_features = self.enc_zE(features)
        alpha = 1e-1 + F.relu(zE_features[:,0])
        beta = 1e-1 + F.relu(zE_features[:,1])

        zE_stat = {'alpha': alpha, 'beta': beta}

        return zE_stat

    def forward(self, zE_stat, zI_stat):
        device = zE_stat['alpha'].device
        zE = self.draw_beta(zE_stat).unsqueeze(1) * self.zE_max
        zI = self.draw_normal(zI_stat)

        radiance_no_plume = self.dec(zI)
        CH4_transmittance = torch.exp(- zE * self.CH4_absorption.to(device) * (1 + 1 / torch.cos(self.theta.to(device))))
        radiance = radiance_no_plume.detach() * CH4_transmittance

        return zE, radiance_no_plume, radiance

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
    
def cycle_loader(labeled_data, unlabeled_data):
    return zip(cycle(labeled_data), unlabeled_data)

def run(args, model):
    epochs = args.epochs
    best_loss = np.inf
    balance_kl = args.balance_kl
    balance_sam = args.balance_sam
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mse_weights = torch.ones((CH4_absorption.shape[0])).to(device)
    mse_weights[bands] = args.mse_weight_vnir_bands

    if args.restore is False:
        model.train()
        model = model.to(device)
        for epoch in range(epochs):
            train_rec_loss = []
            train_kl_loss = []
            train_u_rec_loss = []
            train_u_kl_loss = []
            train_loss = []
            val_loss = []        

            for data in cycle_loader(labeled_loader, unlabeled_loader):
                (data_noplume, gt_noplume), (data_plume) = data

                data_noplume = data_noplume.to(device)
                gt_noplume = gt_noplume.to(device)
                data_plume = data_plume[0].to(device)

                zI_stat = model.encode_zI(data_noplume[:, bands])
                zE_stat = model.encode_zE(data_noplume[:, CH4_bands], zI_stat)

                _, np_rec, _ = model.forward(zE_stat, zI_stat)

                rec_loss = F.mse_loss(data_noplume, np_rec) + balance_sam * sam_(data_noplume, np_rec, reduction='mean')
                kl_loss = model.kl_div(zE_stat, zI_stat, device)
                labeled_loss = rec_loss + balance_kl * kl_loss #+ reg_loss

                zI_stat = model.encode_zI(data_plume[:, bands])
                zE_stat = model.encode_zE(data_plume[:, CH4_bands], zI_stat)
                _, np_rec, u_rec = model.forward(zE_stat, zI_stat)

                u_mse = F.mse_loss(data_plume[:, CH4_bands], u_rec[:, CH4_bands])
                p_rec_loss = u_mse + balance_sam * sam_(data_plume[:, CH4_bands], u_rec[:, CH4_bands], reduction='mean')

                np_rec_loss = F.mse_loss(data_plume[:, bands], np_rec[:, bands]) + balance_sam * sam_(data_plume[:, bands], np_rec[:, bands], reduction='mean')
                u_rec_loss = p_rec_loss + np_rec_loss

                u_kl_loss = model.kl_div(zE_stat, zI_stat, device)
                unlabeled_loss = u_rec_loss + balance_kl * u_kl_loss

                unlabeled_loss.backward(retain_graph=True)

                for param in model.dec.parameters():
                    if param.requires_grad:
                        param.grad.zero_()

                labeled_loss.backward()

                optim.step()
                optim.zero_grad()

                train_loss.append(labeled_loss.item())

                train_rec_loss.append(rec_loss.item())
                train_kl_loss.append(kl_loss.item())

                train_u_rec_loss.append(u_rec_loss.item())
                train_u_kl_loss.append(u_kl_loss.item())

            for data in val_loader:
                data = data[0].to(device)
                with torch.no_grad():
                    zI_stat = model.encode_zI(data[:, bands])
                    zE_stat = model.encode_zE(data[:, CH4_bands], zI_stat)
                    _, _, rec = model.forward(zE_stat, zI_stat)
                
                rec_loss = F.mse_loss(data, rec) + balance_sam * sam_(data, rec, reduction='mean')
                kl_div = model.kl_div(zE_stat, zI_stat, device)

                loss = rec_loss + balance_kl * kl_div
                val_loss.append(loss.item())

            train_loss = sum(train_loss) / len(train_loss)
            train_rec_loss = sum(train_rec_loss) / len(train_rec_loss)
            train_kl_loss = sum(train_kl_loss) / len(train_kl_loss)
            train_u_rec_loss = sum(train_u_rec_loss) / len(train_u_rec_loss)
            train_u_kl_loss = sum(train_u_kl_loss) / len(train_u_kl_loss)
            val_loss = sum(val_loss) / len(val_loss)

            if val_loss <= best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch + 1, 
                    'best_loss': best_loss,
                    'state_dict': model.state_dict()}, './results/best_model.pth.tar')

            print("Epoch {} - Train loss: {:.4f} - Train rec loss: {:.4f} - Train kl loss: {:.4f} - Train u rec loss: {:.4f} - Train u kl loss: {:.4f} - Val loss: {:.4f}".format(
                epoch + 1, train_loss, train_rec_loss, train_kl_loss, train_u_rec_loss, train_u_kl_loss, val_loss))
            
        torch.save({
            'epoch': epoch + 1, 
            'loss': val_loss,
            'state_dict': model.state_dict()}, './results/final_model.pth.tar')

    else:
        checkpoint = torch.load('./results/best_model.pth.tar')
        print("Loading model at epoch {}".format(checkpoint["epoch"]))
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            test_data = test_data.to(device)
            zI_stat = model.encode_zI(test_data[:, bands])
            zE_stat = model.encode_zE(test_data[:, CH4_bands], zI_stat)
            zE, rec_np, rec = model.forward(zE_stat, zI_stat)

        test_loss = F.mse_loss(rec, test_data)
        print('Test loss: {}'.format(test_loss))

        mae = torch.abs(test_gt - zE.view(-1).cpu()).mean()
        print('MAE: {}'.format(mae))

        fig, ax = plt.subplots()
        fontsize=20
        x = np.linspace(test_gt.min(), test_gt.max(), 1000)
        plt.plot(x, x, color='black', linestyle='--')
        plt.scatter(zE.view(-1).cpu(), test_gt, s=1, color="#FFA41B")
        plt.title('MAE: {.2E} ppm.m'.format(mae), fontsize=fontsize)
        plt.xlabel('Predicted concentration (ppm.m)', fontsize=fontsize)
        plt.ylabel('True concentration (ppm.m)', fontsize=fontsize)
        plt.xlim(0, 200000)
        plt.xticks(ticks=[0, 50000, 100000, 150000], labels=[0, 50000, 100000, 150000])
        ax.tick_params(axis='both', labelsize=0.7*fontsize)
        plt.grid(visible=True, linestyle='--', alpha=0.5)
        plt.savefig('./results/p3vae_pred.pdf', dpi=100, bbox_inches='tight', pad_inches=0.05)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="A model among {'baseline', 'vae', 'hybrid'}")
    parser.add_argument('--device', type=str, default='cpu', help="Specify cpu or gpu")
    parser.add_argument('--dataset', type=str, default='simulation', help="simulation or real")

    # Training options
    training_options = parser.add_argument_group('Training')
    training_options.add_argument('--epochs', type=int, default=200)
    training_options.add_argument('--lr', type=float, default=1e-4)
    training_options.add_argument('--batch_size', type=int, default=128)
    training_options.add_argument('--balance_sam', type=float, default=1e-2) 
    training_options.add_argument('--balance_kl', type=float, default=1e-6) 
    training_options.add_argument('--mse_weight_vnir_bands', type=float, default=1e-3)
    training_options.add_argument('--restore', action='store_true')

    args = parser.parse_args()

    root = '../../data/methane/'

    # Constants
    wv_CH4, CH4_absorption = read_csv(root + '../constants/CH4_absorption.csv')
    wv_CH4 = wv_CH4 * 1000 # convert µm to nm
    theta = 30 * np.pi / 180 # solar zenith angle

    # Prior
    rho_p = 6E4
    var_rho_p = 533333333
    max_rho_p = 1.2E5

    # Wavelengths & bad band list
    prisma_wv = np.load(root + 'prisma_wv.npy')
    prisma_bbl = np.load(root + 'prisma_bbl.npy')

    # Standard wv
    wv = np.load(root + '../constants/wv.npy')
    wv = wv * 1000 # convert µm to nm

    # Fwhm
    prisma_fwhm = np.load(root + 'prisma_fwhm.npy')
    prisma_fwhm = prisma_fwhm[prisma_bbl]
    prisma_wv = prisma_wv[prisma_bbl]
    prisma_wv, prisma_fwhm = prisma_wv[:-15], prisma_fwhm[:-15]
    prisma_std = fwhm2std(prisma_fwhm) 
    wv_sfr, prisma_sfr = std2sfr(prisma_wv, prisma_std, wv_sfr=wv) 
    CH4_absorption = resample_from_wv(wv_sfr, wv_CH4, CH4_absorption)
    CH4_absorption = np.sum(CH4_absorption[ np.newaxis, :] * prisma_sfr, axis=1) / np.sum(prisma_sfr, axis=1)
    bands = np.arange(96)
    CH4_bands = np.arange(96, CH4_absorption.shape[0])

    wv_H2O, H2O_transmittance = read_csv(root + '../constants/H2O_transmittance.csv')
    wv_H2O = wv_H2O * 1000 # convert µm to nm
    H2O_transmittance = resample_from_wv(wv_sfr, wv_H2O, H2O_transmittance)
    H2O_transmittance = np.sum(H2O_transmittance[ np.newaxis, :] * prisma_sfr, axis=1) / np.sum(prisma_sfr, axis=1)


    # Load train data
    train_data = np.load(root + 'train_data.npy')
    labeled_data = np.load(root + 'no_plume_data.npy')
    labels = np.zeros(labeled_data.shape[0])

    max_data = np.max(train_data)
    min_data = np.min(train_data)
    train_data = (train_data - min_data) / (max_data - min_data)
    labeled_data = (labeled_data - min_data) / (max_data - min_data)

    # Load test data
    test_data = np.load(root + 'test_data.npy')
    test_gt = np.load(root + 'test_labels.npy')

    test_data = (test_data - min_data) / (max_data - min_data) 

    model = p3VAE(
        input_dim=train_data.shape[1],
        num_bands=[96, 62],
        zI_dim=32,
        CH4_absorption=CH4_absorption, 
        theta=theta,
        min_data=min_data,
        max_data=max_data,
        zE_mean=rho_p,
        zE_var=var_rho_p,
        zE_max=max_rho_p
        )

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    labeled_dataset  = torch.utils.data.TensorDataset(
        torch.from_numpy(labeled_data).float(),
        torch.from_numpy(labels).float(),
    )

    unlabeled_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_data).float()
    )

    unlabeled_dataset, val_dataset = torch.utils.data.random_split(unlabeled_dataset, [0.8, 0.2])

    test_data = torch.from_numpy(test_data).float()
    test_gt = torch.from_numpy(test_gt).float()

    labeled_loader = torch.utils.data.DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
    )

    run(args, model)