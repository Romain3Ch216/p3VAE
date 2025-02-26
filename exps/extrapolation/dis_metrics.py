# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import minmax_scale


def get_mutual_information(x, y, normalize=True):
    ''' Compute mutual information between two random variables
    
    :param x:      random variable
    :param y:      random variable
    '''
    if normalize:
        return drv.information_mutual_normalised(x, y, norm_factor='Y', cartesian_product=True)
    else:
        return drv.information_mutual(x, y, cartesian_product=True)

def jemmig(factors, codes, continuous_factors=True, nb_bins=10):
    ''' JEMMIG metric from K. Do and T. Tran,
        “Theory and evaluation metrics for learning disentangled representations,”
        in ICLR, 2020.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes

    # quantize latent codes
    codes = minmax_scale(codes)  # normalize in [0, 1] all columns
    codes = get_bin_index(codes, nb_bins)  # quantize values and get indexes

    # compute mutual information matrix
    mi_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            mi_matrix[f, c] = get_mutual_information(factors[:, f], codes[:, c], normalize=False)

    # compute joint entropy matrix 
    je_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            X = np.stack((factors[:, f], codes[:, c]), 0)
            je_matrix[f, c] = drv.entropy_joint(X)

    # compute the mean gap for all factors
    sum_gap = 0
    jemmig_scores = []; je = []; gap = []
    for f in range(nb_factors):
        mi_f = np.sort(mi_matrix[f, :])
        je_idx = np.argsort(mi_matrix[f, :])[-1]; je.append(je_matrix[f, je_idx])
        gap.append(mi_f[-1] - mi_f[-2])
        jemmig_not_normalized = je_matrix[f, je_idx] - mi_f[-1] + mi_f[-2]

        # normalize by H(f) + log(#bins)
        jemmig_f = jemmig_not_normalized / (drv.entropy_joint(factors[:, f]) + np.log2(nb_bins))
        jemmig_f = 1 - jemmig_f
        jemmig_scores.append(jemmig_f)
        sum_gap += jemmig_f
    
    # compute the mean gap
    jemmig_score = sum_gap / nb_factors
    
    return jemmig_score, jemmig_scores, je, gap

def get_bin_index(x, nb_bins):
    ''' Discretize input variable
    
    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)