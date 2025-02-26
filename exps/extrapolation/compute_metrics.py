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

# load data
data_test = np.loadtxt('{}/data_{}.txt'.format(datadir, dataname))

# load true parameters
params_test = np.loadtxt('{}/true_params_{}.txt'.format(datadir, dataname))
factors = params_test[:, [1, 2]]

n_runs = 5

metrics = {
    'phyvae': np.zeros((5, 4)),
    'vae': np.zeros((5, 4)),
    'p3vae': np.zeros((5, 4))
}


for i in range(1, n_runs + 1):
    
    modeldir = './out_pendulum/free_damped_pendulum/phy_vae/phy_vae_{}'.format(i)
    modeldir_noreg = './out_pendulum/free_damped_pendulum/phy_vae_no_reg/phy_vae_no_reg_{}'.format(i)
    p3vaedir = './out_pendulum/free_damped_pendulum/p3vae/p3vae_{}'.format(i)
    nnonlydir = './out_pendulum/free_damped_pendulum/nn_only/nn_only_{}'.format(i)

    

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
    print("Phy-VAE: Test loss on in-distribution data: ", gap)

    # prediction + extrapolation

    model.len_intg = (data_test.shape[1] - 1) * model.intg_lev + 1
    model.t_intg = torch.linspace(0.0, model.dt_intg*(model.len_intg-1), model.len_intg)

    data_tensor = torch.Tensor(data_test).to(device)
    with torch.no_grad():
        x_mean, _ = model.decode(z_phy, z_aux1, z_aux2, init_y, full=False)

    test_loss = torch.sum(torch.abs(x_mean[:, :args_tr_dict['dim_t']] - data_test[:, :args_tr_dict['dim_t']]), dim=1).mean()
    print("Phy-VAE: Test loss on in-distribution data: ", test_loss)

    out_test_loss = torch.sum(torch.abs(x_mean[:, args_tr_dict['dim_t']:] - data_test[:, args_tr_dict['dim_t']:]), dim=1).mean()
    print("Phy-VAE: Test loss on out-distribution data: ", out_test_loss)

    metrics['phyvae'][i-1, 0] = test_loss.item()
    metrics['phyvae'][i-1, 1] = out_test_loss.item()
    metrics['phyvae'][i-1, 2] = gap[0]
    metrics['phyvae'][i-1, 3] = gap[1]

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
    print("VAE: Test loss on in-distribution data: ", gap)

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

    metrics['vae'][i-1, 0] = test_loss.item()
    metrics['vae'][i-1, 1] = out_test_loss.item()
    metrics['vae'][i-1, 2] = gap[0]
    metrics['vae'][i-1, 3] = gap[1]

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
    print("p3VAE: Test loss on in-distribution data: ", gap)


    # prediction + extrapolation

    # p3vae.dim_t = data_test.shape[1]
    p3vae.len_intg = (data_test.shape[1] - 1) * p3vae.intg_lev + 1
    p3vae.t_intg = torch.linspace(0.0, p3vae.dt_intg*(p3vae.len_intg-1), p3vae.len_intg)

    data_tensor = torch.Tensor(data_test).to(device)
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        init_y = data_tensor[:,0].clone().view(-1,1)
        p3vae_x_mean = p3vae.decode(xi, omega, omega, init_y, full=True)

    # test_loss = torch.sum(torch.abs(p3vae_x_mean[:, :args_tr_dict['dim_t']] - data_test[:, :args_tr_dict['dim_t']]), dim=1).mean()
    test_loss = torch.sum(torch.abs(p3vae_x_mean[:, :args_tr_dict['dim_t']] - data_test[:, :args_tr_dict['dim_t']]), dim=1).mean()

    print("p3VAE: Test loss on in-distribution data: ", test_loss)

    # out_test_loss = torch.sum(torch.abs(p3vae_x_mean[:, args_tr_dict['dim_t']:] - data_test[:, args_tr_dict['dim_t']:]), dim=1).mean()
    out_test_loss = torch.sum(torch.abs(p3vae_x_mean[:, args_tr_dict['dim_t']:] - data_test[:, args_tr_dict['dim_t']:]), dim=1).mean()
    print("p3VAE: Test loss on out-distribution data: ", out_test_loss)

    metrics['p3vae'][i-1, 0] = test_loss.item()
    metrics['p3vae'][i-1, 1] = out_test_loss.item()
    metrics['p3vae'][i-1, 2] = gap[0]
    metrics['p3vae'][i-1, 3] = gap[1]




avg_metrics = {
    'p3vae': {
        'mean': {
            'in_mae': metrics['p3vae'][:, 0].mean(),
            'out_mae': metrics['p3vae'][:, 1].mean(),
            'gap': [
                metrics['p3vae'][:, 2].mean(),
                metrics['p3vae'][:, 3].mean()
            ]
        },
        'std': {
            'in_mae': metrics['p3vae'][:, 0].std(),
            'out_mae': metrics['p3vae'][:, 1].std(),
            'gap': [
                metrics['p3vae'][:, 2].std(),
                metrics['p3vae'][:, 3].std()
            ]
        }
    },
    'phyvae': {
        'mean': {
            'in_mae': metrics['phyvae'][:, 0].mean(),
            'out_mae': metrics['phyvae'][:, 1].mean(),
            'gap': [
                metrics['phyvae'][:, 2].mean(),
                metrics['phyvae'][:, 3].mean()
            ]
        },
        'std': {
            'in_mae': metrics['phyvae'][:, 0].std(),
            'out_mae': metrics['phyvae'][:, 1].std(),
            'gap': [
                metrics['phyvae'][:, 2].std(),
                metrics['phyvae'][:, 3].std()
            ]
        }
    },
    'vae': {
        'mean': {
            'in_mae': metrics['vae'][:, 0].mean(),
            'out_mae': metrics['vae'][:, 1].mean(),
            'gap': [
                metrics['vae'][:, 2].mean(),
                metrics['vae'][:, 3].mean()
            ]
        },
        'std': {
            'in_mae': metrics['vae'][:, 0].std(),
            'out_mae': metrics['vae'][:, 1].std(),
            'gap': [
                metrics['vae'][:, 2].std(),
                metrics['vae'][:, 3].std()
            ]
        }
    }
}

with open('./out_pendulum/free_damped_pendulum/metrics.json', 'w') as f:
	json.dump(avg_metrics, f, indent=4)