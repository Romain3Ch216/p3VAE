import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint

import utils
from mlp import MLP


xi_feasible_range = [0.0, 1.0] # xi is also named gamma in some files

class MLP_NET(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(MLP_NET, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    

class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        dim_z_aux1 = config['dim_z_aux1']
        activation = config['activation']
        x_lnvar = config['x_lnvar']

        # x_lnvar
        self.register_buffer('param_x_lnvar', torch.ones(1)*x_lnvar)

        assert dim_z_aux1 >= 0
        hidlayers_aux1 = config['hidlayers_aux1_dec']

        self.func_aux1 = MLP([dim_z_aux1+1,]+hidlayers_aux1+[1,], activation)


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        dim_t = config['dim_t']
        dim_z_aux1 = config['dim_z_aux1']
        dim_z_aux2 = config['dim_z_aux2']
        activation = config['activation']
        num_units_feat = config['num_units_feat']

        if dim_z_aux1 > 0:
            self.omega_encoder = MLP_NET(dim_t, 256, 2)

        if dim_z_aux2 > 0:
            hidlayers_aux2_enc = config['hidlayers_aux2_enc']

            # x --> feature_aux2
            self.func_feat_aux2 = FeatureExtractor(config)

            # feature_aux2 --> z_aux2
            self.func_z_aux2_mean = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)
            self.func_z_aux2_lnvar = MLP([num_units_feat,]+hidlayers_aux2_enc+[dim_z_aux2,], activation)


        hidlayers_xi = config['hidlayers_xi']
        feat_phy_config = dict((k, v) for (k, v) in config.items())
        feat_phy_config['dim_t'] += 1
        self.func_feat_phy = FeatureExtractor(feat_phy_config)

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
        return torch.cat([yy[:,1].reshape(-1,1), torch.zeros_like(yy[:, 1].reshape(-1, 1))], dim=1)


class VAE(nn.Module):
    def __init__(self, config:dict):
        super(VAE, self).__init__()

        assert config['range_xi'][0] <= config['range_xi'][1]

        self.dim_t = config['dim_t']
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
        return prior_xi_stat
    
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

            # physics part (xi & yy --> time-deriv of yy), which is actually a tensor of zeros here
            yy_dot_phy_PA = self.physics_model(xi, yy_PA)
            
            yy_dot_aux_PA = torch.cat([torch.zeros(n,1,device=device),
                    self.dec.func_aux1(torch.cat([z_aux1, yy_PA[:, :1]], dim=1))], dim=1)

            return torch.cat([yy_dot_phy_PA+yy_dot_aux_PA], dim=1)

        # solve
        tmp = torch.zeros(n,1,device=device)
        initcond = torch.cat([init_y, tmp], dim=1) # <n x 2>
        yy_seq = odeint(ODEfunc, initcond, self.t_intg, method=self.ode_solver) # <len_intg x n x 2or4>
        yy_seq = yy_seq[range(0, self.len_intg, self.intg_lev)] # subsample to <dim_t x n x 2or4>

        # extract to <n x dim_t>
        y_seq_PA = yy_seq[:,:,0].T
        x_PA = y_seq_PA; x_PAB = x_PA.clone()

        # Physical prior applied here
        x_PAB = torch.exp(- torch.abs(xi) * self.t_intg.view(1, -1)) * x_PA

        return x_PAB

    def encode_omega(self, x:torch.Tensor):
        x_ = x.view(-1, self.dim_t)

        assert self.dim_z_aux1 > 0
        pred = self.enc.omega_encoder(x_)
        z_aux1_stat = {'mean': pred[:, 0], 'lnvar': pred[:, 1]}

        return z_aux1_stat
    
    def encode_xi(self, x:torch.Tensor, omega:torch.Tensor):
        x_ = x.view(-1, self.dim_t)
        x_ = torch.cat((x_, omega.view(-1, 1)), dim=1)

        feature_phy = self.enc.func_feat_phy(x_)
        z_xi_stat = {'mean':self.enc.func_xi_mean(feature_phy), 'lnvar':self.enc.func_xi_lnvar(feature_phy)}

        return z_xi_stat

    def draw(self, xi_stat:dict, hard_z:bool=False):
        if not hard_z:
            xi = utils.draw_normal(xi_stat['mean'], xi_stat['lnvar'])
        else:
            xi = xi_stat['mean'].clone()

        # cut infeasible regions
        xi = torch.max(torch.ones_like(xi)*xi_feasible_range[0], xi)
        xi = torch.min(torch.ones_like(xi)*xi_feasible_range[1], xi)

        return xi
    
    def forward(self, data:torch.Tensor, omega:torch.Tensor):
        xi_stat = self.encode_xi(data, omega)
        xi = self.draw(xi_stat)

        init_y = data[:,0].clone().view(-1,1)
        x_PAB = self.decode(xi, omega, omega, init_y, full=True)

        return xi_stat, x_PAB