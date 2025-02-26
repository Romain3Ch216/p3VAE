import numpy as np
from scipy.spatial.distance import cdist
from utils import read_csv
from resample import fwhm2std, std2sfr, resample_from_wv
import matplotlib.pyplot as plt
import pdb


class MethaneInversion:
    def __init__(
            self, 
            estimate_background, 
            CH4_absorption,
            theta, 
            CH4_bands,
            alpha=1,
            max_it=100, 
            epsilon=1e-1,
            verbose=100):
        
        self.method = None
        self.estimate_background = estimate_background
        self.CH4_absorption = CH4_absorption
        self.theta = theta
        self.CH4_bands = CH4_bands
        self.vnir_bands = np.arange(CH4_bands[0])
        self.alpha = alpha
        self.max_it = max_it
        self.epsilon = epsilon
        self.verbose = verbose

    def update(self):
        raise NotImplementedError
    
    def forward_model(self, radiance, CH4_concentration):
        CH4_transmittance = np.exp(- CH4_concentration[:, np.newaxis] * CH4_absorption[np.newaxis, :] * (1 + 1 / np.cos(self.theta)))
        obs = CH4_transmittance * radiance
        return obs
    
    def jacobian(self, x):
        return - self.CH4_absorption[np.newaxis, :] * x
    
    def __call__(self, init_parameter, obs_x, obs_np):
        # Initialisation
        delta = np.inf
        param = init_parameter

        # Mean and covariance of obs w/o plume
        assert len(obs_x.shape) == 2
        n, d = obs_x.shape
        assert n > d
        obs_mean = np.mean(obs_np, axis=0).reshape(1, -1)
        obs_diff = obs_np - obs_mean
        obs_cov = np.matmul(obs_diff.T, obs_diff) / n

        # Estimate background obs
        obs_back = self.estimate_background(obs_np, obs_x, self.vnir_bands)

        it = 0
        while delta > self.epsilon and it < self.max_it:
            if it % self.verbose == 0:
                print('Forward model - ', it + 1)
            pred_x = self.forward_model(obs_back, param)
            jac = self.jacobian(pred_x)
            if self.method == 'GD':
                if self.obs_cov:
                    new_param = self.update(param, obs_x, pred_x, jac, self.CH4_bands, obs_cov=obs_cov)
                else:
                    new_param = self.update(param, obs_x, pred_x, jac, self.CH4_bands)
            elif self.method == 'OE':
                new_param = self.update(param, obs_x, pred_x, jac, obs_cov, self.CH4_bands)
            abs_delta = np.abs(new_param - param)
            delta = abs_delta.max()
            
            if it % self.verbose == 0:
                print('Max abs diff: {}ppm.m'.format(delta))
            param = new_param
            it += 1

        return it, delta, param, pred_x
    

class GradientDescent(MethaneInversion):
    def __init__(
        self,
        estimate_background, 
        CH4_absorption,
        theta,
        CH4_bands,
        obs_cov=False,
        alpha=1,
        max_it=100, 
        epsilon=1e-1,
        verbose=10):
        super().__init__(estimate_background, CH4_absorption, theta, CH4_bands, alpha, max_it, epsilon, verbose)
        self.method = 'GD'
        self.obs_cov = obs_cov

    def update(self, param, obs_x, pred_x, jac, bands, obs_cov=None):
        # Select bands
        obs_x = obs_x[:, bands]
        pred_x = pred_x[:, bands]
        jac = jac[:, bands]
        if self.obs_cov:
            obs_cov = obs_cov[:, bands][bands, :]
            obs_cov_inv = np.linalg.inv(obs_cov)

        # Compute gradient
        obs_diff = obs_x - pred_x
        if obs_cov is None:
            grad = - np.matmul(obs_diff[:, np.newaxis, :], jac[:, :, np.newaxis])
        else:
            grad = - np.matmul(
                np.matmul(obs_diff[:, np.newaxis, :], obs_cov_inv),
                jac[:, :, np.newaxis]
            )
        grad = grad.reshape(-1)

        # Compute update
        new_param = param - self.alpha * grad
        return new_param

class OptimalEstimation(MethaneInversion):
    def __init__(
        self,
        estimate_background, 
        CH4_absorption,
        theta,
        CH4_bands,
        param_prior,
        param_prior_var,
        alpha=1,
        max_it=100, 
        epsilon=1e-1,
        verbose=10):
        super().__init__(estimate_background, CH4_absorption, theta, CH4_bands, alpha, max_it, epsilon, verbose)
        self.method = 'OE'
        self.param_prior = param_prior
        self.param_prior_var = param_prior_var
    
    def update(self, param, obs_x, pred_x, jac, obs_cov, bands):
        # Select bands
        obs_x = obs_x[:, bands]
        pred_x = pred_x[:, bands]
        jac = jac[:, bands]
        obs_cov = obs_cov[:, bands][bands, :]
        obs_cov_inv = np.linalg.inv(obs_cov)

        # Obs diff: $y - F(x_i,b)$
        obs_diff = obs_x - pred_x

        # Prior diff $x_i - x_a$
        prior_diff = (param - self.param_prior).reshape(-1, 1)

        # Numerator: $K_i^T S_e^{-1} [y - F(x_i, b) + K_i(x_i - x_a)]
        jac_cov_inv = np.matmul(jac[:, np.newaxis, :], obs_cov_inv[np.newaxis, :, :])

        num_1 = np.matmul(
            jac_cov_inv,
            obs_diff[:, :, np.newaxis]) 
        num_2 = prior_diff.reshape(-1, 1, 1) / self.param_prior_var
        num = num_1 + num_2 
                
        # Denominator: S_a^{-1} + K_i^T S_e^{-1} K_i
        den = 1 / self.param_prior_var + np.matmul(jac_cov_inv, jac[:, :, np.newaxis])

        # Compute update
        new_param = param + self.alpha * (num / den).reshape(-1)
        return new_param

def estimate_background(noplume, plume, bands):
    dist = cdist(plume[:, bands], noplume[:, bands])
    closest = np.argmin(dist, axis=1)
    return noplume[closest]

def true_background_for_debug(noplume, plume, bands):
    return data_gt

# Load test data
data_noplume = np.load('no_plume_data.npy')
data_plume = np.load('test_data.npy')
data_gt = np.load('test_data_no_plume.npy')
labels = np.load('test_labels.npy')

# A priori information
rho_p = 6E4
var_rho_p = 529000000

# Constants
wv_CH4, CH4_absorption = read_csv('../constants/CH4_absorption.csv')
wv_CH4 = wv_CH4 * 1000 # convert µm to nm
theta = 30 * np.pi / 180 # solar zenith angle

# Wavelengths & bad band list
prisma_wv = np.load('prisma_wv.npy')
prisma_bbl = np.load('prisma_bbl.npy')

# Standard wv
wv = np.load('../constants/wv.npy')
wv = wv * 1000 # convert µm to nm

# Prisma fwhm
prisma_fwhm = np.load('prisma_fwhm.npy')
prisma_fwhm = prisma_fwhm[prisma_bbl]
prisma_wv = prisma_wv[prisma_bbl]
prisma_wv, prisma_fwhm = prisma_wv[:-15], prisma_fwhm[:-15]
prisma_std = fwhm2std(prisma_fwhm) 
wv_sfr, prisma_sfr = std2sfr(prisma_wv, prisma_std, wv_sfr=wv) 
CH4_absorption = resample_from_wv(wv_sfr, wv_CH4, CH4_absorption)
CH4_absorption = np.sum(CH4_absorption[ np.newaxis, :] * prisma_sfr, axis=1) / np.sum(prisma_sfr, axis=1)
CH4_bands = np.arange(96, CH4_absorption.shape[0])


inversion = OptimalEstimation(
    estimate_background, 
    CH4_absorption,
    theta, 
    CH4_bands,
    param_prior=rho_p,
    param_prior_var=var_rho_p,
    alpha=1e-3,
    max_it=1000,
    verbose=100,
    epsilon=10
)

init_rho = rho_p * np.ones(data_plume.shape[0])
it, delta, param, pred_x = inversion(init_rho, data_plume, data_noplume)
mae = np.abs(param - labels).mean()


fig, ax = plt.subplots()
fontsize=20
x = np.linspace(labels.min(), labels.max(), 1000)
plt.plot(x, x, color='black', linestyle='--')
plt.scatter(param, labels, s=1, color="#744BFC")
plt.title('MAE: {.2E} ppm.m'.format(mae), fontsize=fontsize)
plt.xlabel('Predicted concentration (ppm.m)', fontsize=fontsize)
plt.ylabel('True concentration (ppm.m)', fontsize=fontsize)
ax.tick_params(axis='both', labelsize=0.7*fontsize)
plt.grid(visible=True, linestyle='--', alpha=0.5)
plt.savefig('./results/oe_pred.pdf', dpi=100, bbox_inches='tight', pad_inches=0.05)