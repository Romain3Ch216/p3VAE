""" ODE-based physics-augmented VAE model.
Source: https://github.com/n-takeishi/phys-vae
Modifications: prior physical knowledge 
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint

from .. import utils
from ..mlp import MLP
# import sys; sys.path.append('../'); import utils; from mlp import MLP


# NOTE: z_aux(s) is/are z_A in the paper, and z_phy is z_P in the paper


xi_feasible_range = [0.0, 1.0] # xi is also named gamma in some files


class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        dim_t = config['dim_t']
        dim_z_aux1 = config['dim_z_aux1']
        dim_z_add = config['dim_z_add']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        x_lnvar = config['x_lnvar']

        # x_lnvar
        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        if dim_z_aux1 >= 0:
            hidlayers_aux1 = config['hidlayers_aux1_dec']

            # z_aux1, yy (=[y, y_dot]) & t --> time-derivative of y_dot
            self.func_aux1 = MLP([dim_z_aux1+dim_z_add,]+hidlayers_aux1+[1,], activation)

        if dim_z_aux2 >= 0:
            hidlayers_aux2 = config['hidlayers_aux2_dec']

            # z_phy, z_aux2 --> x - y_seq
            dim_z_phy = 0 if no_phy else 1
            # self.func_aux2_res = MLP([dim_z_phy+max(0,dim_z_aux1)+dim_z_aux2,]+hidlayers_aux2+[dim_t,], activation)
            self.func_aux2_res = MLP([dim_z_phy+max(0,dim_z_aux1)+dim_z_aux2,]+hidlayers_aux2+[400,], activation)


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        dim_t = config['dim_t']
        dim_z_aux1 = config['dim_z_aux1']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        no_phy = config['no_phy']
        num_units_feat = config['num_units_feat']

        if dim_z_aux1 > 0:
            hidlayers_aux1_enc = config['hidlayers_aux1_enc']

            # x --> feature_aux1
            self.func_feat_aux1 = FeatureExtractor(config)

            # feature_aux1 --> z_aux1
            self.func_z_aux1_mean = MLP([num_units_feat,]+hidlayers_aux1_enc+[dim_z_aux1,], activation)
            self.func_z_aux1_lnvar = MLP([num_units_feat,]+hidlayers_aux1_enc+[dim_z_aux1,], activation)

        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['hidlayers_aux2_enc']

            # x --> feature_aux2
            self.func_feat_aux2 = FeatureExtractor(config)

            # feature_aux2 --> z_aux2
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)

        if not no_phy:
            hidlayers_unmixer = config['hidlayers_unmixer']
            hidlayers_xi = config['hidlayers_xi']

            # x, z_aux1, z_aux2 --> unmixed - x
            self.func_unmixer_res = MLP([dim_t+max(dim_z_aux1,0)+max(dim_z_aux2,0),]+hidlayers_unmixer+[dim_t,], activation)
            # self.func_unmixer = MLP([dim_t+max(dim_z_aux1,0)+max(dim_z_aux2,0),]+hidlayers_unmixer+[dim_t,], activation)

            # unmixed --> feature_phy
            self.func_feat_phy = FeatureExtractor(config)

            # features_phy --> xi
            self.func_xi_mean = nn.Sequential(MLP([num_units_feat,]+hidlayers_xi+[1,], activation), nn.Softplus())
            self.func_xi_lnvar = MLP([num_units_feat,]+hidlayers_xi+[1,], activation)


class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        dim_t = config['dim_t']
        activation = config['activation']
        arch_feat = config['arch_feat']
        num_units_feat = config['num_units_feat']

        self.dim_t = dim_t
        self.arch_feat = arch_feat
        self.num_units_feat = num_units_feat

        if arch_feat=='mlp':
            hidlayers_feat = config['hidlayers_feat']

            self.func= MLP([dim_t,]+hidlayers_feat+[num_units_feat,], activation, actfun_output=True)
        elif arch_feat=='rnn':
            num_rnns_feat = config['num_rnns_feat']

            self.num_rnns_feat = num_rnns_feat
            self.func = nn.GRU(1, num_units_feat, num_layers=num_rnns_feat, bidirectional=False)
        else:
            raise ValueError('unknown feature type')

    def forward(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_t)
        n = x_.shape[0]
        device = x_.device

        if self.arch_feat=='mlp':
            feat = self.func(x_)
        elif self.arch_feat=='rnn':
            h_0 = torch.zeros(self.num_rnns_feat, n, self.num_units_feat, device=device)
            out, h_n = self.func(x_.T.unsqueeze(2), h_0)
            feat = out[-1]

        return feat


class Physics(nn.Module):
    def __init__(self):
        super(Physics, self).__init__()

    def forward(self, xi:torch.Tensor, yy:torch.Tensor):
        """
        given parameter and yy=[y, dy/dt], return dyy/dt=[dy/dt, d^2y/dt^2]
        [state]
            yy: shape <n x 2>
        [physics parameter]
            xi: shape <n x 1>
        """
        return torch.cat([yy[:,1].reshape(-1,1), -xi*yy[:,1].view(-1,1)], dim=1)


class VAE(nn.Module):
    def __init__(self, config:dict):
        super(VAE, self).__init__()

        assert config['range_xi'][0] <= config['range_xi'][1]

        self.dim_t = config['dim_t']
        self.dim_z_add = config['dim_z_add']
        self.dim_z_aux1 = config['dim_z_aux1']
        self.dim_z_aux2 = config['dim_z_aux2']
        self.range_xi = config['range_xi']
        self.activation = config['activation']
        self.dt = config['dt']
        self.intg_lev = config['intg_lev']
        self.ode_solver = config['ode_solver']
        self.no_phy = config['no_phy']

        # Decoding part
        self.dec = Decoders(config)

        # Encoding part
        self.enc = Encoders(config)

        # Physics
        self.physics_model = Physics()

        # set time indices for integration
        self.dt_intg = self.dt / float(self.intg_lev)
        self.len_intg = (self.dim_t - 1) * self.intg_lev + 1
        self.register_buffer('t_intg', torch.linspace(0.0, self.dt_intg*(self.len_intg-1), self.len_intg))


    def priors(self, n:int, device:torch.device):
        prior_xi_stat = {'mean': torch.ones(n,1,device=device) * 0.5 * (self.range_xi[0] + self.range_xi[1]),
            'lnvar': 2.0*torch.log( torch.ones(n,1,device=device) * max(1e-3, 0.866*(self.range_xi[1] - self.range_xi[0])) )}
        prior_z_aux1_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux1), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux1), device=device)}
        prior_z_aux2_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux2), device=device),
            'lnvar': torch.zeros(n, max(0,self.dim_z_aux2), device=device)}
        return prior_xi_stat, prior_z_aux1_stat, prior_z_aux2_stat


    def generate_physonly(self, xi:torch.Tensor, init_y:torch.Tensor):
        n = xi.shape[0]
        device = xi.device

        # define ODE
        def ODEfunc(t:torch.Tensor, yy:torch.Tensor):
            return self.physics_model(xi, yy)

        # solve ODE
        initcond = torch.cat([init_y, torch.zeros(n,1,device=device)], dim=1) # <n x 2>
        yy_seq = odeint(ODEfunc, initcond, self.t_intg, method=self.ode_solver) # <len_intg x n x 2>
        y_seq = yy_seq[range(0, self.len_intg, self.intg_lev), :, 0].T # subsample, extract, and reshape to <n x dim_t>

        return y_seq


    def decode(self, xi:torch.Tensor, z_aux1:torch.Tensor, z_aux2:torch.Tensor, init_y:torch.Tensor, full:bool=False):
        n = xi.shape[0]
        device = xi.device

        # define ODE
        def ODEfunc(t:torch.Tensor, _yy:torch.Tensor):
            """Gives gradient of vector _yy, whose shape is <n x 4> or <n x 2>.
            - t should be a scalar
            - _yy should be shape <n x 4> or <n x 2>
            """

            yy_PA = _yy[:, [0,1]]
            if full:
                yy_P = _yy[:, [2,3]]

            if not self.no_phy:
                # physics part (xi & yy --> time-deriv of yy)
                yy_dot_phy_PA = self.physics_model(xi, yy_PA)
                if full:
                    yy_dot_phy_P = self.physics_model(xi, yy_P)
            else:
                # when model has no physics part *originally*
                yy_dot_phy_PA = torch.cat([yy_PA[:,1].reshape(-1,1), torch.zeros(n, 1)], dim=1)
                if full:
                    yy_dot_phy_P = torch.cat([yy_P[:,1].reshape(-1,1), torch.zeros(n, 1)], dim=1)

            if self.dim_z_aux1 >= 0:
                # in-ODE auxiliary part (z_aux1, yy & t --> time-deriv of y_dot)

                if self.dim_z_add > 1:
                    yy_dot_aux_PA = torch.cat([torch.zeros(n,1,device=device),
                        self.dec.func_aux1(torch.cat([z_aux1, yy_PA, t.expand(n,1)], dim=1))], dim=1)
                elif self.dim_z_add == 1:
                    yy_dot_aux_PA = torch.cat([torch.zeros(n,1,device=device),
                        self.dec.func_aux1(torch.cat([z_aux1, yy_PA[:, :1]], dim=1))], dim=1)
            else:
                # when model has no in-ODE auxiliary part *originally*
                yy_dot_aux_PA = torch.zeros(n, 2, device=device)

            if full:
                return torch.cat([yy_dot_phy_PA+yy_dot_aux_PA, yy_dot_phy_P], dim=1)
            else:
                return torch.cat([yy_dot_phy_PA+yy_dot_aux_PA], dim=1)

        # solve
        tmp = torch.zeros(n,1,device=device)
        if full:
            initcond = torch.cat([init_y, tmp, init_y, tmp.clone()], dim=1) # <n x 4>
        else:
            initcond = torch.cat([init_y, tmp], dim=1) # <n x 2>
        yy_seq = odeint(ODEfunc, initcond, self.t_intg, method=self.ode_solver) # <len_intg x n x 2or4>
        yy_seq = yy_seq[range(0, self.len_intg, self.intg_lev)] # subsample to <dim_t x n x 2or4>
        # extract to <n x dim_t>
        y_seq_PA = yy_seq[:,:,0].T
        if full:
            y_seq_P = yy_seq[:,:,2].T

        # out-ODE auxiliary part (y_seq, z_aux2 --> x)
        x_PA = y_seq_PA; x_PAB = x_PA.clone()
        if full:
            x_P = y_seq_P; x_PB = x_P.clone()

        if self.dim_z_aux2 >= 0:
            residual = self.dec.func_aux2_res(torch.cat((xi, z_aux1, z_aux2), dim=1))
            residual = residual[:, :min(self.dim_t, residual.shape[1])]
            x_PAB += residual
            if full:
                residual = self.dec.func_aux2_res(torch.cat((xi, z_aux1, z_aux2), dim=1))
                residual = residual[:, :min(self.dim_t, residual.shape[1])]
                x_PB += residual

        if full:
            return x_PAB, x_PA, x_PB, x_P, self.dec.param_x_lnvar, yy_seq
        else:
            return x_PAB, self.dec.param_x_lnvar


    def encode(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_t)
        n = x_.shape[0]
        device = x_.device

        # infer z_aux1, z_aux2
        if self.dim_z_aux1 > 0:
            feature_aux1 = self.enc.func_feat_aux1(x_)
            z_aux1_stat = {'mean':self.enc.func_z_aux1_mean(feature_aux1), 'lnvar':self.enc.func_z_aux1_lnvar(feature_aux1)}
        else:
            z_aux1_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        if self.dim_z_aux2 > 0:
            feature_aux2 = self.enc.func_feat_aux2(x_)
            z_aux2_stat = {'mean':self.enc.func_z_aux2_mean(feature_aux2), 'lnvar':self.enc.func_z_aux2_lnvar(feature_aux2)}
        else:
            z_aux2_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        # infer xi
        if not self.no_phy:
            # unmixing
            unmixed = x_ + self.enc.func_unmixer_res(torch.cat((x_, z_aux1_stat['mean'], z_aux2_stat['mean']), dim=1))
            # unmixed = self.enc.func_unmixer(torch.cat((x_, z_aux1_stat['mean'], z_aux2_stat['mean']), dim=1))
            # unmixed += self.enc.func_unmixer_res(torch.cat((x_, z_aux1_stat['mean'], z_aux2_stat['mean']), dim=1))

            # after unmixing
            feature_phy = self.enc.func_feat_phy(unmixed)
            xi_stat = {'mean': self.enc.func_xi_mean(feature_phy), 'lnvar': self.enc.func_xi_lnvar(feature_phy)}
        else:
            unmixed = torch.zeros(n, self.dim_t, device=device)
            xi_stat = {'mean':torch.empty(n, 0, device=device), 'lnvar':torch.empty(n, 0, device=device)}

        return xi_stat, z_aux1_stat, z_aux2_stat, unmixed


    def draw(self, xi_stat:dict, z_aux1_stat:dict, z_aux2_stat:dict, hard_z:bool=False):
        if not hard_z:
            xi = utils.draw_normal(xi_stat['mean'], xi_stat['lnvar'])
            z_aux1 = utils.draw_normal(z_aux1_stat['mean'], z_aux1_stat['lnvar'])
            z_aux2 = utils.draw_normal(z_aux2_stat['mean'], z_aux2_stat['lnvar'])
        else:
            xi = xi_stat['mean'].clone()
            z_aux1 = z_aux1_stat['mean'].clone()
            z_aux2 = z_aux2_stat['mean'].clone()

        # cut infeasible regions
        xi = torch.max(torch.ones_like(xi)*xi_feasible_range[0], xi)
        xi = torch.min(torch.ones_like(xi)*xi_feasible_range[1], xi)

        return xi, z_aux1, z_aux2


    def forward(self, x:torch.Tensor, reconstruct:bool=True, hard_z:bool=False):
        xi_stat, z_aux1_stat, z_aux2_stat, _ = self.encode(x)

        if not reconstruct:
            return xi_stat, z_aux1_stat, z_aux2_stat

        # draw & reconstruction
        init_y = x[:,0].clone().view(-1,1)
        x_mean, x_lnvar = self.decode(*self.draw(xi_stat, z_aux1_stat, z_aux2_stat, hard_z=hard_z), init_y, full=False)

        return xi_stat, z_aux1_stat, z_aux2_stat, x_mean, x_lnvar
