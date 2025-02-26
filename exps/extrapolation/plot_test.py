import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from physvae import utils
from physvae.pendulum.model import VAE
from physvae.pendulum.p3vae_model_2 import VAE as p3VAE
from physvae.dis_metrics import jemmig

# setting
datadir = './data/pendulum/free_damped_pendulum'
dataname = 'test'
modeldir = './out_pendulum/free_damped_pendulum/phy_vae/phy_vae_1'
p3vaedir = './out_pendulum/free_damped_pendulum/p3vae/p3vae_1'
nnonlydir = './out_pendulum/free_damped_pendulum/nn_only/nn_only_1'

metrics = {
    'phyvae': {
        'in_mae': 0,
        'out_mae': 0,
        'mig': []
    },
    'p3vae': {
        'in_mae': 0,
        'out_mae': 0,
        'mig': []
    },
    'vae': {
        'in_mae': 0,
        'out_mae': 0,
        'mig': []
    }
}

# load data
data_test = np.loadtxt('{}/data_{}.txt'.format(datadir, dataname))

# load true parameters
params_test = np.loadtxt('{}/true_params_{}.txt'.format(datadir, dataname))
factors = params_test[:, [1, 2]]

# import pdb; pdb.set_trace()
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

################## phy-VAE ########################

# set and load model (aux only)
with open('{}/args.json'.format(modeldir), 'r') as f:
    args_tr_dict = json.load(f)

args_tr_dict['dim_z_add'] = 1
model = VAE(args_tr_dict).to(device)
model.load_state_dict(torch.load('{}/model.pt'.format(modeldir), map_location=device))
model.eval()

dim_t_tr = args_tr_dict['dim_t']
dt = args_tr_dict['dt']

# infer latent variables using short data
data_tensor = torch.Tensor(data_test[:, :args_tr_dict['dim_t']]).to(device)
init_y = data_tensor[:,0].clone().view(-1,1)

with torch.no_grad():
    # aux only
    z_phy_stat, z_aux1_stat, z_aux2_stat, unmixed = model.encode(data_tensor)
    z_phy, z_aux1, z_aux2 = model.draw(z_phy_stat, z_aux1_stat, z_aux2_stat, hard_z=False)

codes = np.concatenate((z_aux1_stat['mean'], z_phy_stat['mean']), axis=1)
jemmig_score, jemmig_scores, je, gap = jemmig(factors, codes)
print("Phy-VAE: disentanglement: ", gap, jemmig_score)

model.len_intg = (data_test.shape[1] - 1) * model.intg_lev + 1
model.t_intg = torch.linspace(0.0, model.dt_intg*(model.len_intg-1), model.len_intg)

data_tensor = torch.Tensor(data_test).to(device)
with torch.no_grad():
    x_mean, _ = model.decode(z_phy, z_aux1, z_aux2, init_y, full=False)

test_loss = torch.sum(torch.abs(x_mean[:, :args_tr_dict['dim_t']] - data_test[:, :args_tr_dict['dim_t']]), dim=1).mean()
print("Phy-VAE: Test loss on in-distribution data: ", test_loss)

out_test_loss = torch.sum(torch.abs(x_mean[:, args_tr_dict['dim_t']:] - data_test[:, args_tr_dict['dim_t']:]), dim=1).mean()
print("Phy-VAE: Test loss on out-distribution data: ", out_test_loss)

metrics['phyvae']['mig'] = gap
metrics['phyvae']['in_mae'] = test_loss.item()
metrics['phyvae']['out_mae'] = out_test_loss.item()

# ################## NN ONLY ########################
# set and load model (aux only)
with open('{}/args.json'.format(nnonlydir), 'r') as f:
    args_tr_dict = json.load(f)

args_tr_dict['dim_z_add'] = 3
model = VAE(args_tr_dict).to(device)
model.load_state_dict(torch.load('{}/model.pt'.format(nnonlydir), map_location=device))
model.eval()

dim_t_tr = args_tr_dict['dim_t']
dt = args_tr_dict['dt']

# infer latent variables using short data
data_tensor = torch.Tensor(data_test[:, :args_tr_dict['dim_t']]).to(device)
init_y = data_tensor[:,0].clone().view(-1,1)

with torch.no_grad():
    # aux only
    z_phy_stat, z_aux1_stat, z_aux2_stat, unmixed = model.encode(data_tensor)
    z_phy, z_aux1, z_aux2 = model.draw(z_phy_stat, z_aux1_stat, z_aux2_stat, hard_z=False)

codes = np.concatenate((z_aux1_stat['mean'], z_phy_stat['mean']), axis=1)
jemmig_score, jemmig_scores, je, gap = jemmig(factors, codes)
print("VAE: disentanglement: ", gap, jemmig_score)

# prediction + extrapolation

model.len_intg = (data_test.shape[1] - 1) * model.intg_lev + 1
model.t_intg = torch.linspace(0.0, model.dt_intg*(model.len_intg-1), model.len_intg)
model.dim_t = data_test.shape[1]
model.dim_z_aux2 = - 1

data_tensor = torch.Tensor(data_test).to(device)
with torch.no_grad():
    vae_x_mean, _ = model.decode(z_phy, z_aux1, z_aux2, init_y, full=False)

test_loss = torch.sum(torch.abs(vae_x_mean[:, :dim_t_tr] - data_test[:, :dim_t_tr]), dim=1).mean()
print("VAE: Test loss on in-distribution data: ", test_loss)

out_test_loss = torch.sum(torch.abs(vae_x_mean[:, dim_t_tr:] - data_test[:, dim_t_tr:]), dim=1).mean()
print("VAE: Test loss on out-distribution data: ", out_test_loss)

metrics['vae']['mig'] = gap
metrics['vae']['in_mae'] = test_loss.item()
metrics['vae']['out_mae'] = out_test_loss.item()

###################### p3VAE ################################
with open('{}/args.json'.format(p3vaedir), 'r') as f:
    args_tr_dict = json.load(f)

p3vae = p3VAE(args_tr_dict).to(device)
p3vae.load_state_dict(torch.load('{}/model.pt'.format(p3vaedir), map_location=device))
p3vae.eval()

dim_t_tr = args_tr_dict['dim_t']
dt = args_tr_dict['dt']

# infer latent variables using short data
data_tensor = torch.Tensor(data_test[:, :args_tr_dict['dim_t']]).to(device)
init_y = data_tensor[:,0].clone().view(-1,1)

with torch.no_grad():
    omega_stat = p3vae.encode_omega(data_tensor)
    q_phi = torch.distributions.normal.Normal(loc=omega_stat['mean'], scale=torch.exp(omega_stat['lnvar'])**0.5)
    omega = torch.nn.functional.relu(q_phi.sample()).view(-1, 1)
    
    xi_stat = p3vae.encode_xi(data_tensor, omega)
    xi = p3vae.draw(xi_stat)

codes = np.concatenate((omega_stat['mean'].reshape(-1, 1), np.abs(xi_stat['mean'])), axis=1)
jemmig_score, jemmig_scores, je, gap = jemmig(factors, codes)
print("p3VAE: disentanglement: ", gap, jemmig_score)

p3vae.len_intg = (data_test.shape[1] - 1) * p3vae.intg_lev + 1
p3vae.t_intg = torch.linspace(0.0, p3vae.dt_intg*(p3vae.len_intg-1), p3vae.len_intg)

data_tensor = torch.Tensor(data_test).to(device)

with torch.no_grad():
    init_y = data_tensor[:,0].clone().view(-1,1)
    p3vae_x_mean = p3vae.decode(xi, omega, omega, init_y, full=True)

test_loss = torch.sum(torch.abs(p3vae_x_mean[:, :args_tr_dict['dim_t']] - data_test[:, :args_tr_dict['dim_t']]), dim=1).mean()

print("p3VAE: Test loss on in-distribution data: ", test_loss)

out_test_loss = torch.sum(torch.abs(p3vae_x_mean[:, args_tr_dict['dim_t']:] - data_test[:, args_tr_dict['dim_t']:]), dim=1).mean()
print("p3VAE: Test loss on out-distribution data: ", out_test_loss)

metrics['p3vae']['mig'] = gap
metrics['p3vae']['in_mae'] = test_loss.item()
metrics['p3vae']['out_mae'] = out_test_loss.item()

sample_id = np.random.randint(0, 1000, size=6)

fontsize = 45
linewidth = 8
xticks = [round(x, 2) for x in model.t_intg.numpy()]

for i in range(len(sample_id)):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(data_test[sample_id[i]], linestyle='-', lw=linewidth, color='black', label='truth')
    # ax.plot(x_mean_noreg[sample_id[i]], color='#2E79CE', lw=linewidth, label=r'$\phi$-VAE w/o reg')
    ax.plot(p3vae_x_mean[sample_id[i]], linestyle=(0, (5, 1)), color='#54BA63', lw=linewidth, label=r'p$^3$VAE')
    ax.plot(vae_x_mean[sample_id[i]], linestyle=(0, (5, 1)), color='#f38b2b', lw=linewidth, label='VAE')
    ax.plot(x_mean[sample_id[i]], linestyle=(0, (5, 1)), color='#2E79CE', lw=linewidth, label=r'$\phi$-VAE')
    ax.axvline(x=50, linestyle='--', lw=linewidth, color='gray')
    ax.set_xlabel('t (s)', fontsize=fontsize)
    ax.set_xticklabels(xticks)
    ax.tick_params(axis='both', labelsize=0.7*fontsize)
    if i in [0, 3]:
        ax.set_ylabel(r'$\theta(t)$', fontsize=fontsize)
    if i == 0:
        ax.legend(fontsize=fontsize, ncol=3, loc='lower center')
    if i == 0:
        plt.text(-105, 0.29, 'reconstruction', fontsize=fontsize, weight='bold')
        plt.text(50, 0.27, r'$\longleftarrow$', fontsize=fontsize, weight='bold')
        plt.text(110, 0.29, 'extrapolation', fontsize=fontsize, weight='bold')
        plt.text(105, 0.27, r'$\longrightarrow$', fontsize=fontsize, weight='bold')
    # fig.set_size_inches(18.5, 10.5)
    plt.savefig('./out_pendulum/free_damped_pendulum/figures/pendulum_sample_{}.pdf'.format(i+1), dpi=100, bbox_inches='tight', pad_inches=0.05)


i=4

fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(data_test[sample_id[i], :300], lw=linewidth, color='black', label='truth') #'-gD', markevery=10, markersize=20,
ax.plot(p3vae_x_mean[sample_id[i], :300], linestyle=(0, (5, 1)), color='#54BA63', lw=linewidth, label=r'p$^3$VAE')
ax.axvline(x=50, linestyle='--', lw=linewidth, color='gray')
ax.set_xlabel('t (s)', fontsize=fontsize)
ax.set_xticklabels(xticks[:300])
ax.set_ylabel(r'$\vartheta(t)$', fontsize=fontsize)
ax.legend(fontsize=fontsize, ncol=3, loc='lower right')
ax.tick_params(axis='both', labelsize=0.7*fontsize)
plt.text(-100, 1.5, 'reconstruction', fontsize=fontsize, weight='bold')
plt.text(15, 1.35, r'$\longleftarrow$', fontsize=fontsize, weight='bold')
plt.text(55, 1.5, 'extrapolation', fontsize=fontsize, weight='bold')
plt.text(50, 1.35, r'$\longrightarrow$', fontsize=fontsize, weight='bold')
plt.savefig('./out_pendulum/free_damped_pendulum/figures/p3vae_{}.pdf'.format(i), dpi=100, bbox_inches='tight', pad_inches=0.05)


fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(data_test[sample_id[i], :300], linestyle='-', lw=linewidth, color='black')#, label='truth')
ax.plot(x_mean[sample_id[i], :300], linestyle=(0, (5, 1)), color='#2E79CE', lw=linewidth, label=r'$\phi$-VAE')
ax.axvline(x=50, linestyle='--', lw=linewidth, color='gray')
ax.set_xlabel('t (s)', fontsize=fontsize)
ax.set_xticklabels(xticks[:300])
ax.tick_params(axis='both', labelsize=0.7*fontsize)
ax.legend(fontsize=fontsize, ncol=3, loc='lower right')
plt.savefig('./out_pendulum/free_damped_pendulum/figures/phi_vae_{}.pdf'.format(i), dpi=100, bbox_inches='tight', pad_inches=0.05)


fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(data_test[sample_id[i], :300], linestyle='-', lw=linewidth, color='black')#, label='truth')
ax.plot(vae_x_mean[sample_id[i], :300], linestyle=(0, (5, 1)), color='#F5AF00', lw=linewidth, label='VAE + ODE solver')
ax.axvline(x=50, linestyle='--', lw=linewidth, color='gray')
ax.set_xlabel('t (s)', fontsize=fontsize)
ax.set_xticklabels(xticks[:300])
ax.tick_params(axis='both', labelsize=0.7*fontsize)
ax.legend(fontsize=fontsize, ncol=3, loc='lower right')
plt.savefig('./out_pendulum/free_damped_pendulum/figures/vae_{}.pdf'.format(i), dpi=100, bbox_inches='tight', pad_inches=0.05)

with open('./out_pendulum/free_damped_pendulum/metrics.json', 'w') as f:
	json.dump(metrics, f, indent=4)