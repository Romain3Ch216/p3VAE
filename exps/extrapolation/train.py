# Source: https://github.com/n-takeishi/phys-vae

import argparse
import os
import json
import time
import numpy as np

import torch
from torch import optim
import torch.utils.data

from model import VAE
import utils
# import sys; sys.path.append('../'); import utils


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
    parser.add_argument('--dim-z-add', type=int, default=1)

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
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--balance-kld', type=float, default=1.0)
    parser.add_argument('--balance-unmix', type=float, default=0.0)
    parser.add_argument('--balance-dataug', type=float, default=0.0)
    parser.add_argument('--balance-lact-dec', type=float, default=0.0)
    parser.add_argument('--balance-lact-enc', type=float, default=0.0)
    parser.add_argument('--balance-int-reg', type=float, default=0.1)
    parser.add_argument('--reg-len-intg', type=int, default=300)

    # others
    parser.add_argument('--train-size', type=int, default=-1)
    parser.add_argument('--save-interval', type=int, default=999999999)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234567890)

    return parser


def loss_function(args, data, xi_stat, z_aux1_stat, z_aux2_stat, x_mean):
    n = data.shape[0]
    device = data.device

    recerr_sq = torch.sum((x_mean - data).pow(2), dim=1).mean()

    prior_xi_stat, prior_z_aux1_stat, prior_z_aux2_stat = model.priors(n, device)

    KL_z_aux1 = utils.kldiv_normal_normal(z_aux1_stat['mean'], z_aux1_stat['lnvar'],
        prior_z_aux1_stat['mean'], prior_z_aux1_stat['lnvar']) if args.dim_z_aux1 > 0 else torch.zeros(1, device=device)
    KL_z_aux2 = utils.kldiv_normal_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'],
        prior_z_aux2_stat['mean'], prior_z_aux2_stat['lnvar']) if args.dim_z_aux2 > 0 else torch.zeros(1, device=device)
    KL_xi = utils.kldiv_normal_normal(xi_stat['mean'], xi_stat['lnvar'],
        prior_xi_stat['mean'], prior_xi_stat['lnvar']) if not args.no_phy else torch.zeros(1, device=device)

    kldiv = (KL_z_aux1 + KL_z_aux2 + KL_xi).mean()

    return recerr_sq, kldiv


def train(epoch, args, device, loader, model, optimizer):
    model.train()
    logs = {'recerr_sq':.0, 'kldiv':.0, 'unmix':.0, 'dataug':.0, 'lact_dec':.0}

    for batch_idx, (data,) in enumerate(loader):
        data = data.to(device)
        batch_size = len(data)
        optimizer.zero_grad()

        # inference & reconstruction on original data
        xi_stat, z_aux1_stat, z_aux2_stat, unmixed = model.encode(data)
        xi, z_aux1, z_aux2 = model.draw(xi_stat, z_aux1_stat, z_aux2_stat, hard_z=False)
        init_y = data[:,0].clone().view(-1,1)
        x_PAB, x_PA, x_PB, x_P, x_lnvar, _ = model.decode(xi, z_aux1, z_aux2, init_y, full=True)
        x_var = torch.exp(x_lnvar)
        
        # ELBO
        recerr_sq, kldiv = loss_function(args, data, xi_stat, z_aux1_stat, z_aux2_stat, x_PAB)

        # unmixing regularization (R_{DA,1})
        if not args.no_phy:
            reg_unmix = torch.sum((unmixed - x_P.detach()).pow(2), dim=1).mean()
        else:
            reg_unmix = torch.zeros(1, device=device).squeeze()

        # data augmentation regularization (R_{DA,2})
        if not args.no_phy:
            model.eval()
            with torch.no_grad():
                aug_xi = torch.rand((batch_size,1), device=device)*(args.range_xi[1]-args.range_xi[0])+args.range_xi[0]
                aug_x_P = model.generate_physonly(aug_xi, init_y.detach())
            model.train()
            aug_feature_phy = model.enc.func_feat_phy(aug_x_P.detach())
            reg_dataug = (model.enc.func_xi_mean(aug_feature_phy) - aug_xi).pow(2).mean()
        else:
            reg_dataug = torch.zeros(1, device=device).squeeze()

        # least action principle (R_ppc)
        if not args.no_phy:
            dif_PA_P = torch.sum((x_PA - x_P).pow(2), dim=1).mean()
            dif_PB_P = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
            dif_PAB_PA =  torch.sum((x_PAB - x_PA).pow(2), dim=1).mean()
            dif_PAB_PB =  torch.sum((x_PAB - x_PB).pow(2), dim=1).mean()
            reg_lact_dec = 0.25*dif_PA_P + 0.25*dif_PB_P + 0.25*dif_PAB_PA + 0.25*dif_PAB_PB
        else:
            reg_lact_dec = torch.zeros(1, device=device).squeeze()

        # loss function
        kldiv_balanced = (args.balance_kld + args.balance_lact_enc) * kldiv * x_var.detach()
        loss = recerr_sq + kldiv_balanced \
            + args.balance_unmix*reg_unmix + args.balance_dataug*reg_dataug + args.balance_lact_dec*reg_lact_dec

        # update model parameters
        loss.backward()

        if args.grad_clip>0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

        # log
        logs['recerr_sq'] += recerr_sq.detach()*batch_size
        logs['kldiv'] += kldiv.detach()*batch_size
        logs['unmix'] += reg_unmix.detach()*batch_size
        logs['dataug'] += reg_dataug.detach()*batch_size
        logs['lact_dec'] += reg_lact_dec.detach()*batch_size

    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Training (rec. err.)^2: {:.4f}  kldiv: {:.4f}  unmix: {:4f}  dataug: {:4f}  lact_dec: {:4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv'], logs['unmix'], logs['dataug'], logs['lact_dec']))
    return logs


def valid(epoch, args, device, loader, model):
    model.eval()
    logs = {'recerr_sq':.0, 'kldiv':.0}
    with torch.no_grad():
        for i, (data,) in enumerate(loader):
            data = data.to(device)
            batch_size = len(data)
            xi_stat, z_aux1_stat, z_aux2_stat, x_mean, _ = model(data)
            recerr_sq, kldiv = loss_function(args, data, xi_stat, z_aux1_stat, z_aux2_stat, x_mean)

            logs['recerr_sq'] += recerr_sq.detach()*batch_size
            logs['kldiv'] += kldiv.detach()*batch_size

    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Validation (rec. err.)^2: {:.4f}  kldiv: {:.4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv']))
    return logs


if __name__ == '__main__':

    parser = set_parser()
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")


    # set random seed
    torch.manual_seed(args.seed)


    # load training/validation data
    data_train = np.loadtxt('{}/data_{}.txt'.format(args.datadir, args.dataname_train), ndmin=2)
    data_valid = np.loadtxt('{}/data_{}.txt'.format(args.datadir, args.dataname_valid), ndmin=2)

    args.dim_t = data_train.shape[1]

    if args.train_size > 0:
        if args.train_size > data_train.shape[0]:
            raise ValueError('train_size must be <= {}'.format(data_train.shape[0]))
        idx = torch.randperm(data_train.shape[0]).numpy()[0:args.train_size]
        data_train = data_train[idx]


    # load data args
    with open('{}/args_{}.json'.format(args.datadir, args.dataname_train), 'r') as f:
        args_data_dict = json.load(f)

    args.dt = args_data_dict['dt']


    # set data loaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    loader_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(data_train).float()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    loader_valid = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(data_valid).float()),
        batch_size=args.batch_size, shuffle=False, **kwargs)


    # set model
    model = VAE(vars(args)).to(device)


    # set optimizer
    kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay, 'eps': args.adam_eps}
    optimizer = optim.Adam(model.parameters(), **kwargs)


    print('start training with device', device)
    print(vars(args))
    print()


    # save args
    with open('{}/args.json'.format(args.outdir), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


    # create log files
    with open('{}/log.txt'.format(args.outdir), 'w') as f:
        print('# epoch recerr_sq kldiv unmix dataug lact_dec valid_recerr_sq valid_kldiv duration', file=f)


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
                logs_train['recerr_sq'], logs_train['kldiv'], logs_train['unmix'], logs_train['dataug'], logs_train['lact_dec'],
                logs_valid['recerr_sq'], logs_valid['kldiv'], dur_total), file=f)

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
