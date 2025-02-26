import argparse
import os
import json
import time
import numpy as np

import torch
from torch import optim
import torch.utils.data

from p3vae_model import VAE
import utils


def set_parser():
    parser = argparse.ArgumentParser(description='')

    # input/output setting
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--dataname-train', type=str, default='train')
    parser.add_argument('--dataname-valid', type=str, default='valid')

    # prior knowledge
    parser.add_argument('--range-xi', type=float, nargs=2, default=[0.0, 1.0])

    # model (general)
    parser.add_argument('--dim-z-aux1', type=int, required=True, help="if 0, aux1 is still alive without latent variable; set -1 to deactivate")
    parser.add_argument('--dim-z-aux2', type=int, required=True, help="if 0, aux2 is still alive without latent variable; set -1 to deactivate")
    parser.add_argument('--activation', type=str, default='elu') #choices=['relu','leakyrelu','elu','softplus','prelu'],
    parser.add_argument('--ode-solver', type=str, default='euler')
    parser.add_argument('--intg-lev', type=int, default=1)
    parser.add_argument('--no-phy', action='store_true', default=False)

    # model (decoder)
    parser.add_argument('--x-lnvar', type=float, default=-8.0)
    parser.add_argument('--hidlayers-aux1-dec', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-aux2-dec', type=int, nargs='+', default=[128,])

    # model (encoder)
    parser.add_argument('--hidlayers-aux1-enc', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-aux2-enc', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-unmixer', type=int, nargs='+', default=[128,])
    parser.add_argument('--hidlayers-xi', type=int, nargs='+', default=[128])
    parser.add_argument('--arch-feat', type=str, default='mlp')
    parser.add_argument('--num-units-feat', type=int, default=256)
    parser.add_argument('--hidlayers-feat', type=int, nargs='+', default=[256,])
    parser.add_argument('--num-rnns-feat', type=int, default=1)

    # optimization (base)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--adam-eps', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--balance-kld', type=float, default=1.0)
    parser.add_argument('--balance-reg', type=float, default=1.0)
    parser.add_argument('--balance-entropy', type=float, default=1e-3)
    parser.add_argument('--importance-sampling', action='store_true', default=False)
    # otherstraint, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234567890)

    return parser


def loss_function(data, xi_stat, x_mean, reduction=None):
    n = data.shape[0]
    device = data.device

    recerr_sq = torch.sum((x_mean - data).pow(2), dim=1)

    prior_xi_stat = model.priors(n, device)

    kldiv = utils.kldiv_normal_normal(xi_stat['mean'], xi_stat['lnvar'],
        prior_xi_stat['mean'], prior_xi_stat['lnvar'])
    
    if reduction == 'mean':
        recerr_sq = recerr_sq.mean()
        kldiv = kldiv.mean()

    return recerr_sq, kldiv


def train(epoch, args, device, loader, model, optimizer):
    model.train()
    logs = {'recerr_sq': [], 'kldiv': [], 'reg_loss': [], 'entropy': [], 'u_recerr_sq': [], 'u_kldiv': [], 'unsup_loss': []}

    for batch_idx, (data, omega_true) in enumerate(loader):
        data = data.to(device)
        omega_true = omega_true.to(device)
        batch_size = len(data)
        optimizer.zero_grad()

        # process labeled / unlabeled data
        mask_labels = omega_true > 0
        labeled_data = data[mask_labels]
        omega_true = omega_true[mask_labels]
        unlabeled_data = data[mask_labels == False]
        omega_true = omega_true.view(-1, 1)

        # inference & reconstruction on labeled data
        omega_stat = model.encode_omega(labeled_data)

        reg_loss = torch.nn.functional.l1_loss(omega_true.view(-1), omega_stat['mean'].view(-1))

        # ELBO
        xi_stat, x_PAB = model.forward(labeled_data, omega_true)
        recerr_sq, kldiv = loss_function(labeled_data, xi_stat, x_PAB, reduction='mean')

        # loss function
        kldiv_balanced = args.balance_kld * kldiv
        supervised_loss = recerr_sq + kldiv_balanced + args.balance_reg * reg_loss

        # inference & reconstruction on unlabeled data
        u_omega_stat = model.encode_omega(unlabeled_data)
        entropy = (0.5 * u_omega_stat['lnvar']).mean() # + cst terms = 0.5 + 0.5 * torch.log(torch.tensor([2*np.pi]))

        q_phi = torch.distributions.normal.Normal(loc=u_omega_stat['mean'], scale=1e-2+torch.exp(u_omega_stat['lnvar'])**0.5)
        u_omega = torch.nn.functional.relu(q_phi.rsample()).view(-1, 1)
        lik_ratio = torch.ones_like(u_omega)

        # ELBOphysvae
        u_xi_stat, u_x_PAB = model.forward(unlabeled_data, u_omega)
        u_recerr_sq, u_kldiv = loss_function(unlabeled_data, u_xi_stat, u_x_PAB, reduction=None)
        u_kldiv_balanced = args.balance_kld * u_kldiv.mean()
        unsupervised_loss = (u_recerr_sq * lik_ratio.view(u_recerr_sq.shape)).mean() + u_kldiv_balanced #- entropy
        
        unsupervised_loss.backward(retain_graph=True)

        for param in model.dec.parameters():
            if param.requires_grad:
                param.grad.zero_()

        supervised_loss.backward()

        optimizer.step()

        logs['recerr_sq'].append(recerr_sq.detach())
        logs['kldiv'].append(kldiv.detach())
        logs['reg_loss'].append(reg_loss.detach())
        logs['entropy'].append(entropy.detach())
        logs['u_recerr_sq'].append(u_recerr_sq.mean().detach())
        logs['u_kldiv'].append(u_kldiv.mean().detach())
        logs['unsup_loss'].append(unsupervised_loss.detach())
    
    for key in logs:
        logs[key] = sum(logs[key]) / len(logs[key])
    print('====> Epoch: {}  Training (rec. err.)^2: {:.4f}  kldiv: {:.4f} reg_loss: {:.4f} entropy: {:.4f}  u_recerr_sq: {:.4f}  u_kldiv:  {:.4f}  unsup_loss: {:.4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv'], logs['reg_loss'], logs['entropy'], logs['u_recerr_sq'], logs['u_kldiv'], logs['unsup_loss']))
    return logs


def valid(epoch, args, device, loader, model):
    model.eval()
    logs = {'recerr_sq': [], 'kldiv': [], 'reg_loss': [], 'entropy': [], 'u_recerr_sq': [], 'u_kldiv': [], 'unsup_loss': []}
    for _, (data, omega_true) in enumerate(loader):
        data = data.to(device)
        omega_true = omega_true.to(device)

        # process labeled / unlabeled data
        mask_labels = omega_true > 0
        labeled_data = data[mask_labels]
        omega_true = omega_true[mask_labels]
        unlabeled_data = data[mask_labels == False]
        omega_true = omega_true.view(-1, 1)

        # inference & reconstruction on labeled data
        with torch.no_grad():
            omega_stat = model.encode_omega(labeled_data)

        reg_loss = torch.nn.functional.l1_loss(omega_true.view(-1), omega_stat['mean'].view(-1))

        # ELBO
        with torch.no_grad():
            xi_stat, x_PAB = model.forward(labeled_data, omega_true)
        recerr_sq, kldiv = loss_function(labeled_data, xi_stat, x_PAB, reduction='mean')

        # inference & reconstruction on unlabeled data
        with torch.no_grad():
            u_omega_stat = model.encode_omega(unlabeled_data)
        entropy = (0.5 * u_omega_stat['lnvar']).mean() # + cst terms = 0.5 + 0.5 * torch.log(torch.tensor([2*np.pi]))

        q_phi = torch.distributions.normal.Normal(loc=u_omega_stat['mean'], scale=1e-2+torch.exp(u_omega_stat['lnvar'])**0.5)
        u_omega = torch.nn.functional.relu(q_phi.rsample()).view(-1, 1)
        lik_ratio = torch.ones_like(u_omega)

        # ELBO
        with torch.no_grad():
            u_xi_stat, u_x_PAB = model.forward(unlabeled_data, u_omega)
        u_recerr_sq, u_kldiv = loss_function(unlabeled_data, u_xi_stat, u_x_PAB, reduction=None)
        u_kldiv_balanced = args.balance_kld * u_kldiv.mean()
        unsupervised_loss = (u_recerr_sq * lik_ratio.view(u_recerr_sq.shape)).mean() + u_kldiv_balanced - args.balance_entropy * entropy

        logs['recerr_sq'].append(recerr_sq.detach())
        logs['kldiv'].append(kldiv.detach())
        logs['reg_loss'].append(reg_loss.detach())
        logs['entropy'].append(entropy.detach())
        logs['u_recerr_sq'].append(u_recerr_sq.mean().detach())
        logs['u_kldiv'].append(u_kldiv.mean().detach())
        logs['unsup_loss'].append(unsupervised_loss.detach())
    
    for key in logs:
        logs[key] = sum(logs[key]) / len(logs[key])
    print('====> Epoch: {}  Validation (rec. err.)^2: {:.4f}  kldiv: {:.4f} reg_loss: {:.4f} entropy: {:.4f}  u_recerr_sq: {:.4f}  u_kldiv:  {:.4f}  unsup_loss: {:.4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv'], logs['reg_loss'], logs['entropy'], logs['u_recerr_sq'], logs['u_kldiv'], logs['unsup_loss']))
    return logs


if __name__ == '__main__':

    parser = set_parser()
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # set random seed
    torch.manual_seed(args.seed)

    if args.importance_sampling:
        print('Using importance sampling')

    # load training/validation data
    data_train = np.loadtxt('{}/data_{}.txt'.format(args.datadir, args.dataname_train), ndmin=2)
    labels_train = np.loadtxt('{}/true_params_{}.txt'.format(args.datadir, args.dataname_train))
    data_valid = np.loadtxt('{}/data_{}.txt'.format(args.datadir, args.dataname_valid), ndmin=2)
    labels_valid = np.loadtxt('{}/true_params_{}.txt'.format(args.datadir, args.dataname_valid))

    # extract omega and mask 80% of data
    labels_train = labels_train[:, 1]
    labels_valid = labels_valid[:, 1]

    random_train_ind = torch.randperm(len(labels_train))
    random_train_ind = random_train_ind[:int(0.8 * labels_train.shape[0])]
    labels_train[random_train_ind] = -1
    
    random_valid_ind = torch.randperm(len(labels_valid))
    random_valid_ind = random_valid_ind[:int(0.8 * labels_valid.shape[0])]
    labels_valid[random_valid_ind] = - 1

    args.dim_t = data_train.shape[1]

    # load data args
    with open('{}/args_{}.json'.format(args.datadir, args.dataname_train), 'r') as f:
        args_data_dict = json.load(f)

    args.dt = args_data_dict['dt']

    # set data loaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    loader_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.Tensor(data_train).float(),
        torch.Tensor(labels_train).float()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    loader_valid = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.Tensor(data_valid).float(),
        torch.Tensor(labels_valid).float()),
        batch_size=args.batch_size, shuffle=False, **kwargs)


    # set model
    model = VAE(vars(args)).to(device)

    # set optimizer
    kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay, 'eps': args.adam_eps}
    optimizer = optim.Adam(model.parameters(), **kwargs)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('start training with device', device)
    print(vars(args))
    print()

    # save args
    with open('{}/args.json'.format(args.outdir), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # create log files
    with open('{}/log.txt'.format(args.outdir), 'w') as f:
        print('# epoch recerr_sq kldiv reg_loss entropy u_recerr_sq u_kldiv val_recerr_sq val_kldiv val_reg_loss val_entropy val_u_recerr_sq val_u_kldiv duration', file=f)


    # main iteration
    info = {'bestvalid_epoch':0, 'bestvalid_recerr':1e10}
    dur_total = .0
    for epoch in range(1, args.epochs + 1):
        # training
        start_time = time.time()
        logs_train = train(epoch, args, device, loader_train, model, optimizer)
        dur_total += time.time() - start_time

        # validation
        logs_valid = valid(epoch, args, device, loader_valid, model)

        # save loss information
        with open('{}/log.txt'.format(args.outdir), 'a') as f:
            print('{} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e}'.format(epoch,
                logs_train['recerr_sq'], logs_train['kldiv'], logs_train['reg_loss'], logs_train['entropy'], logs_train['u_recerr_sq'], logs_train['u_kldiv'],
                logs_valid['recerr_sq'], logs_valid['kldiv'], logs_valid['reg_loss'], logs_valid['entropy'], logs_valid['u_recerr_sq'], logs_valid['u_kldiv'],
                dur_total), file=f)
            

        # save model if best validation loss is achieved
        if logs_valid['recerr_sq'] < info['bestvalid_recerr']:
            info['bestvalid_epoch'] = epoch
            info['bestvalid_recerr'] = logs_valid['recerr_sq']
            torch.save(model.state_dict(), '{}/model.pt'.format(args.outdir))
            print('best model saved')

        # save model at interval
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), '{}/model_e{}.pt'.format(args.outdir, epoch))

        print()

    print()
    print('end training')
