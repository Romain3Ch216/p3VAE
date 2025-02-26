# Source: https://github.com/n-takeishi/phys-vae

import torch
import torch.nn as nn


def actmodule(activation:str):
    if activation == 'softplus':
        return nn.Softplus()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('unknown activation function specified')


# #@torch.jit.script
# def kernel_mat_linear(sample1:torch.Tensor, sample2:torch.Tensor=None,):
#     if sample2 is None:
#         sample2 = sample1
#     return torch.mm(sample1, sample2.t())


# #@torch.jit.script
# def kernel_mat_gauss(sample1:torch.Tensor, sample2:torch.Tensor=None, width:float=None):
#     # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/
#     norm1 = (sample1**2).sum(1).view(-1, 1)
#     if sample2 is None:
#         sample2_t = torch.transpose(sample1, 0, 1)
#         norm2 = norm1.view(1, -1)
#     else:
#         sample2_t = torch.transpose(sample2, 0, 1)
#         norm2 = (sample2**2).sum(1).view(1, -1)
#     dist_mat = norm1 + norm2 - 2.0 * torch.mm(sample1, sample2_t)
#     dist_mat[dist_mat != dist_mat] = 0.0
#     if width is None:
#         width = torch.median(dist_mat).detach()
#         width = 1e-6 if width < 1e-6 else width
#     return torch.exp(-dist_mat / width)

@torch.jit.script
def kernel_mat_gauss(sample:torch.Tensor, width:float):
    # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/
    norm = (sample**2).sum(1).view(-1, 1)
    dist_mat = norm + norm.view(1,-1) - 2.0 * torch.mm(sample, torch.transpose(sample,0,1))
    dist_mat[dist_mat != dist_mat] = 0.0
    # width = torch.max(torch.ones(1,device=sample.device)*1e-4, torch.median(dist_mat).detach())
    return torch.exp(-dist_mat / width)


@torch.jit.script
def hsic(kmat1:torch.Tensor, kmat2:torch.Tensor):
    """
    Unbiased estimator of HSIC [Song+ ICML 2007]
    """
    m = kmat1.shape[0]
    assert m>3
    device = kmat1.device
    K1 = (1.0 - torch.eye(m, device=device)) * kmat1
    K2 = (1.0 - torch.eye(m, device=device)) * kmat2
    return ( torch.sum(K1*K2.T) + torch.sum(K1)*torch.sum(K2)/(m-1)/(m-2) \
            - torch.sum(torch.sum(K1,dim=0)*torch.sum(K2,dim=1))*2/(m-2) ) / m / (m-3)


@torch.jit.script
def mmd(kmat11:torch.Tensor, kmat22:torch.Tensor, kmat12:torch.Tensor):
    """
    Estimator of MMD
    """
    m1 = kmat11.shape[0]
    m2 = kmat22.shape[0]
    return torch.sum(kmat11)/m1/m1 + torch.sum(kmat22)/m2/m2 - 2.0*torch.sum(kmat12)/m1/m2


@torch.jit.script
def nll_normal(data:torch.Tensor, mean:torch.Tensor, lnvar:torch.Tensor):
    """
    Negative log likelihood based on normal observation model, -log N(data | mean1, diag(exp(lnvar)))
    """
    d = data.shape[1]
    if lnvar.ndim==2:
        nll = 0.5*d*1.8379 + 0.5*torch.sum((data-mean).pow(2)/lnvar.exp(), dim=1) + 0.5*torch.sum(lnvar, dim=1)
    else:
        nll = 0.5*d*1.8379 + 0.5*torch.sum((data-mean).pow(2), dim=1)/lnvar.exp() + 0.5*d*lnvar
    return nll


@torch.jit.script
def kldiv_logits_logits(logits1:torch.Tensor, logits2:torch.Tensor):
    """
    KL divergence between categorical distributions represented by sets of logits
    """
    # KL(q(z_phy|x) or p(z_phy) || p(z_phy))
    #   = \sum_i q_i (logit_q_i - logit_p_i) - (lse(logits_q) - lse(logits_p))
    logits1_logsumexp = torch.logsumexp(logits1, dim=1)
    logits2_logsumexp = torch.logsumexp(logits2, dim=1)
    probs1 = torch.exp(logits1 - logits1_logsumexp.unsqueeze(1))
    return torch.sum(probs1 * (logits1-logits2), dim=1) - (logits1_logsumexp-logits2_logsumexp)


@torch.jit.script
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


@torch.jit.script
def pdfratio_normal(data:torch.Tensor, mean1:torch.Tensor, lnvar1:torch.Tensor, mean2:torch.Tensor, lnvar2:torch.Tensor):
    """
    Value of ratio of pdfs, N(mean1, diag(exp(lnvar1))) / N(mean2, diag(exp(lnvar2)))
    """
    lnpdf1 = -nll_normal(data, mean1, lnvar1)
    lnpdf2 = -nll_normal(data, mean2, lnvar2)
    return torch.exp(lnpdf1 - lnpdf2)


def draw_normal(mean:torch.Tensor, lnvar:torch.Tensor):
    std = torch.exp(0.5*lnvar)
    eps = torch.randn_like(std) # reparametrization trick
    return mean + eps*std
