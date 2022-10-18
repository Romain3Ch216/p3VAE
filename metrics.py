# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to compute quantitative metrics and to reproduce figure 6

# Part of the following code is under the following license:
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
import numpy as np
import math

from numpy.core.numeric import NaN
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
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
    jemmig_scores = []
    for f in range(nb_factors):
        mi_f = np.sort(mi_matrix[f, :])
        je_idx = np.argsort(mi_matrix[f, :])[-1]

        # Compute unormalized JEMMIG
        jemmig_not_normalized = je_matrix[f, je_idx] - mi_f[-1] + mi_f[-2]

        # normalize by H(f) + log(#bins)
        jemmig_f = jemmig_not_normalized / (drv.entropy_joint(factors[:, f]) + np.log2(nb_bins))
        jemmig_f = 1 - jemmig_f
        jemmig_scores.append(jemmig_f)
        sum_gap += jemmig_f
    
    # compute the mean gap
    jemmig_score = sum_gap / nb_factors
    
    return jemmig_score, jemmig_scores

def get_bin_index(x, nb_bins):
    ''' Discretize input variable
    
    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)

#==================== Following code is original  ========================

import torch   
import torch.nn.functional as F 
from models.utils import sam_, one_hot
from data import spectra_bbm, load_dataset
from models import * 
from utils import * 
from models.models import load_model
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import json  
import matplotlib.pyplot as plt 
import os 
import sys 
import pdb 
import math 


def build_data_under_diff_irradiance(dataset, class_id):
	spectra = []
	z_phi_list = []
	omega_list = []
	alpha_list = []
	eta_list = []
	rho = [torch.from_numpy(rho_).unsqueeze(0) for rho_ in dataset.classes[class_id]['spectrum']]
	if len(rho) > 1:
		z_phi = torch.linspace(0, 1, math.ceil(1e4/(10*len(rho)*10)))
	else:
		z_phi = torch.linspace(0, 1, math.ceil(1e4/(10*len(rho))))
	omega = torch.linspace(0.2, 1, 10)
	E_dir = torch.from_numpy(dataset.E_dir)
	E_dif = torch.from_numpy(dataset.E_dif)
	theta = torch.tensor([dataset.theta])

	if len(rho) > 1:
		for k in range(len(rho)):
			for alpha in torch.linspace(0, 1, 10):
				rho_ = (1-alpha)*rho[k] + alpha*rho[(k+1)%len(rho)]
				for z1 in z_phi:
					for O in omega:
						cochise_correction = (z1*E_dir + O*E_dif)/(torch.cos(theta)*E_dir + E_dif)
						sp = rho_*cochise_correction
						spectra.append(sp)  
						z_phi_list.append(z1)
						omega_list.append(O)
						alpha_list.append(alpha)
						eta_list.append(torch.tensor([k+(k+1)%len(rho)]))
	else:
		rho_ = rho[0]
		for z1 in z_phi:
			for O in omega:
				cochise_correction = (z1*E_dir + O*E_dif)/(torch.cos(theta)*E_dir + E_dif)
				sp = rho_*cochise_correction
				spectra.append(sp)  
				z_phi_list.append(z1)
				omega_list.append(O)
				alpha_list.append(torch.ones(1))
				eta_list.append(torch.ones(1))

	spectra = torch.cat(spectra).float()
	z_phi = torch.cat([x.view(1, 1) for x in z_phi_list])
	omega = torch.cat([x.view(1, 1) for x in omega_list])
	alpha = torch.cat([x.view(1, 1) for x in alpha_list])
	eta = torch.cat([x.view(1, 1) for x in alpha_list])
	factors = torch.cat((torch.ones((spectra.shape[0], 1))*class_id, z_phi, omega, alpha, eta), dim=-1)
	return spectra, factors

def plot_z_true_vs_z_pred(z_true, z_pred, confusion, z_std, class_id, fontsize=20, colors = ['#F2A65A', '#909CC2', '#89A7A7', '#6320EE', '#E94974']):
	x=np.linspace(0,1,100)
	z_pred = z_pred.numpy()
	z_true = z_true.numpy()
	z_std = torch.exp(10*z_std).numpy()
	fig = plt.figure()
	plt.scatter(z_true[confusion==1], z_pred[confusion==1], alpha=0.15, color=colors[3], s=z_std[confusion==1])
	plt.scatter(z_true[confusion==0], z_pred[confusion==0], alpha=0.15, color=colors[0], s=z_std[confusion==0])
	plt.plot(x, x, lw=2, color=colors[4], label='y=x')
	plt.xlabel(r"$\delta_{dir} cos \: \Theta$", fontsize=fontsize)
	plt.ylabel(r'$z_P$', fontsize=fontsize)
	plt.legend(loc=4, prop={'size': 20})
	# plt.show()
	# pdb.set_trace()
	plt.savefig('./results/{}/Figures/cos_{}.pdf'.format(config['model'], class_id), dpi=200, bbox_inches='tight', pad_inches=0.05)

def plot_omega_true_vs_omega_pred(omega_true, z_pred, confusion, fontsize=20, colors = ['#F2A65A', '#909CC2', '#89A7A7', '#6320EE', '#E94974']):
	x=np.linspace(0,1,100)
	z_pred = z_pred.numpy()
	omega_true = omega_true.numpy()
	fig = plt.figure()
	plt.scatter(omega_true[confusion==1], z_pred[confusion==1]+0.2, alpha=0.25, color=colors[3], s=10)
	plt.scatter(omega_true[confusion==0], z_pred[confusion==0]+0.2, alpha=0.25, color=colors[0], s=10)
	plt.plot(x, x, lw=2, color=colors[4], label='y=x')
	plt.xlabel(r"$\Omega$", fontsize=fontsize)
	plt.ylabel(r'$\hat{Omega}$', fontsize=fontsize)
	plt.legend(loc=4, prop={'size': 20})
	plt.show()
	# pdb.set_trace()
	# plt.savefig('./results/{}/Figures/cos_{}.pdf'.format(config['model'], class_id), dpi=200, bbox_inches='tight', pad_inches=0.05)

def plot_irradiance(z_true, omega_true, confusion, fontsize=20, colors = ['#F2A65A', '#909CC2', '#89A7A7', '#6320EE', '#E94974']):
	fig = plt.figure()
	plt.scatter(z_pred[confusion==1], omega_true[confusion==1], alpha=0.25, color=colors[3], s=10)
	plt.scatter(z_pred[confusion==0], omega_true[confusion==0], alpha=0.25, color=colors[0], s=10)
	plt.xlabel(r"$\delta_{dir} cos \: \Theta$", fontsize=fontsize)
	plt.ylabel(r'$\Omega$', fontsize=fontsize)
	plt.legend(loc=4, prop={'size': 20})
	plt.show()

def plot_confusions(dataset, model, spectra, confusion, z_pred_phi, z_pred_eta, z_true, omega_true, logits):
	confusion_spectra = spectra[confusion==0]
	z_phi = z_pred_phi[confusion==0]
	z_eta = z_pred_eta[confusion==0]
	omega = z_phi+0.2
	z_true = z_true[confusion==0]
	omega_true = omega_true[confusion==0]
	logits = logits[confusion==0]

	for class_id in np.unique(confusion_pred):
		rho = dataset.classes[class_id+1]['spectrum']
		fig, ax = plt.subplots(1, 4)
		plt.title(dataset.classes[class_id+1]['label'])
		sp = confusion_spectra[confusion_pred==class_id]
		z_phi_ = z_phi[confusion_pred==class_id]
		z_eta_ = z_eta[confusion_pred==class_id]
		z = torch.cat((z_phi_.unsqueeze(1), z_eta_), dim=-1)
		omega_ = omega[confusion_pred==class_id]
		y = one_hot(np.array([class_id]*z.shape[0]), model.y_dim)
		y_true = one_hot(np.array([true_class_id-1]*z.shape[0]), model.y_dim)

		z_true_ = z_true[confusion_pred==class_id]
		omega_true_ = omega_true[confusion_pred==class_id]

		logits_pred = logits[confusion_pred==class_id]

		with torch.no_grad():
			x = model.decoder(z, y)
			s = model.decoder(z, y_true)

		loss_x = (torch.mean(F.mse_loss(sp, x, reduction='none'), dim=-1) + config['lambda_sam']*sam_(sp, x, reduction='none')).mean()
		loss_s = (torch.mean(F.mse_loss(sp, s, reduction='none'), dim=-1) + config['lambda_sam']*sam_(sp, s, reduction='none')).mean()

		for i in range(sp.shape[0]):
			ax[0].plot(sp[i,:], alpha=0.2) # Les spectres de test

		for j in range(sp.shape[0]):
			ax[1].plot(s[i], alpha=0.2)
			ax[2].plot(x[i], alpha=0.2) # La reconstruction avec la classe mal prédite 


			for rho_ in rho:
				rho_ = torch.from_numpy(rho_).unsqueeze(0)
				cochise_correction = (z_phi_[i]*E_dir + omega_[i]*E_dif)/(torch.cos(theta)*E_dir + E_dif)
				other_class_sp = (rho_*cochise_correction).view(-1)
				ax[3].plot(other_class_sp, alpha=0.2)

		ax[0].set_ylim(0, 1)
		ax[0].set_title('Test spectra of {} confused with {}'.\
			format(dataset.classes[true_class_id]['label'], dataset.classes[class_id+1]['label']), fontdict={'fontsize': 8})
		ax[1].set_ylim(0, 1)
		ax[1].set_title('Reconstructed spectra of {} under predicted irradiance - {:.2f}'.format(dataset.classes[true_class_id]['label'], loss_s), fontdict={'fontsize': 8})
		ax[2].set_ylim(0, 1)
		ax[2].set_title('Reconstructed spectra of {} under predicted irradiance - {:.2f}'.format(dataset.classes[class_id+1]['label'], loss_x), fontdict={'fontsize': 8})
		ax[3].set_ylim(0, 1)
		ax[3].set_title('Test spectra of {} under predicted irradiance'.format(dataset.classes[class_id+1]['label']), fontdict={'fontsize': 8})
		plt.show()

def write_confusions(pred, true_class_id, report):
	confusion = (pred == true_class_id-1).long()
	confusion_pred = pred[confusion==0]
	unique, counts = np.unique(confusion_pred, return_counts=True)
	counts = counts / len(confusion_pred)
	
	for i, class_id in enumerate(unique):
		report[true_class_id]['confusions'][int(class_id+1)] = counts[i]

def plot_entropy_std(entropy, z_std, fontsize=20, colors = ['#F2A65A', '#909CC2', '#89A7A7', '#6320EE', '#E94974']):
	plt.scatter(entropy, z_std, alpha=0.25, color=colors[3], s=10)
	plt.xlabel("Entropy", fontsize=fontsize)
	plt.ylabel(r'$z_P$ standard deviation ', fontsize=fontsize)
	plt.legend(loc=4, prop={'size': 20})
	plt.show()

if __name__ == "__main__":

	global_report = []

	for k in range(1, len(sys.argv)):
		print('Model ', k)
		print(sys.argv[k])

		results_path = sys.argv[k]

		with open(results_path + '/config.json') as f:
			config = json.load(f)

		dataset = load_dataset(config)
		target_names = [dataset.classes[i]['label'] for i in range(len(dataset.classes))][1:] 
		E_dir = torch.from_numpy(dataset.E_dir)
		E_dif = torch.from_numpy(dataset.E_dif)
		theta = torch.tensor([dataset.theta])
		
		model, _, _ = load_model(dataset, config)
		checkpoint = torch.load(results_path + '/best_model.pth.tar')
		model.load_state_dict(checkpoint['state_dict'])

		pred_q, pred_p, labels = [], [], []
		factors, codes = [], []
		entropy, z_std_ = [], []
		report = {}
		report['jemmig_score'] = {}
		report['avg_f1_score_p'] = 0 
		report['avg_f1_score_q'] = 0 

		for true_class_id in range(1, model.classifier.y_dim+1):
			report[true_class_id] = {}
			report[true_class_id]['confusions'] = {}

			spectra, factors_ = build_data_under_diff_irradiance(dataset, true_class_id)
			labels_, z_true, omega_true, alpha, eta = torch.split(factors_, 1, dim=1)
			labels.extend(labels_.long().numpy().reshape(-1))

			try:
				pred = model.classifier.predict(spectra)
			except:
				pred = model.predict(spectra)

			pred = torch.argmax(pred, dim=-1).numpy()
			pred_q.extend(pred)

			try:
				Lr, pred, z_pred_phi, z_pred_eta, logits, z_std = model.argmax_q_z_x_batch(spectra, config)
				codes_ = torch.cat((pred.unsqueeze(1), z_pred_phi.unsqueeze(1), z_pred_eta), dim=-1)
				pred_p.extend(pred.numpy())
				# z_std_.extend(z_std.numpy())
				# logits = torch.softmax(logits, dim=-1)
				# H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
				# entropy.extend(H.numpy())

				write_confusions(pred, true_class_id, report)
				plot_z_true_vs_z_pred(z_true.squeeze(1), z_pred_phi, confusion, z_std, true_class_id)

				factors.append(factors_)
				codes.append(codes_)
				# plot_irradiance(z_true.squeeze(1), omega_true.squeeze(1), confusion)
				# plot_confusions(dataset, model, spectra, confusion, z_pred_phi, z_pred_eta, z_true, omega_true, logits)
			except:
				pred_p.extend(pred)
				pass 

		# plot_entropy_std(entropy, z_std_)
		# pdb.set_trace()

		labels = np.array(labels)-1
		pred_q = np.array(pred_q)
		pred_p = np.array(pred_p)

		f1_score_q = f1_score(labels, pred_q, average=None)
		f1_score_p = f1_score(labels, pred_p, average=None)

		for class_id in range(1, model.classifier.y_dim+1):
			report[class_id]['f1_score_q'] = f1_score_q[class_id-1]
			report[class_id]['f1_score_p'] = f1_score_p[class_id-1]
			report['avg_f1_score_q'] += f1_score_q[class_id-1]/len(f1_score_q)
			report['avg_f1_score_p'] += f1_score_p[class_id-1]/len(f1_score_p)

		if config['model'] in ['clf', 'clf_all']:
			global_report.append(report)
		else: 
			factors = torch.cat(factors, dim=0).numpy()
			codes = torch.cat(codes, dim=0).numpy()

			jemmig_score, jemmig_scores = jemmig(factors, codes, nb_bins=20)
			dci_score = dci(factors, codes)

			for i in range(len(jemmig_scores)):
				report['jemmig_score'][i] = jemmig_scores[i]

			report['jemmig_score']['avg_jemmig_score'] = jemmig_score
			report['disentanglement'] = dci_score[0]
			report['completeness'] = dci_score[1]
			report['informativeness'] = dci_score[2]

			global_report.append(report)

	avg_report = {}
	avg_report['jemmig_score'] = {}
	for class_id in range(1, len(f1_score_q)+1):
		avg_report[class_id] = {}
		for metric in report[class_id]:
			avg_report[class_id][metric] = 0

	try:
		for i in range(len(jemmig_scores)):
			avg_report['jemmig_score'][i] = 0
	except:
		pass

	avg_report['jemmig_score']['avg_jemmig_score'] = 0
	avg_report['avg_f1_score_q'] = 0 
	avg_report['avg_f1_score_p'] = 0
	avg_report['disentanglement'] = 0
	avg_report['completeness'] = 0
	avg_report['informativeness'] = 0
	avg_report['f1_score_p'] = []; avg_report['f1_score_q'] = []

	for report in global_report:
		for key in report:
			if key in ['avg_f1_score_q', 'avg_f1_score_p', 'disentanglement', 'completeness', 'informativeness']:
				avg_report[key] += report[key]/len(global_report)
			elif key == 'jemmig_score':
				try:
					avg_report[key]['avg_jemmig_score'] += report[key]['avg_jemmig_score']/len(global_report)
					for i in range(len(jemmig_scores)):
						avg_report[key][i] += report[key][i]/len(global_report)
				except:
					pass
			else:
				for metric in report[key]:
					if metric == 'confusions':
						avg_report[key][metric] = report[key][metric]
					else:
						avg_report[key][metric] += report[key][metric]/len(global_report)
		avg_report['f1_score_p'].append(report['avg_f1_score_p']) ; avg_report['f1_score_q'].append(report['avg_f1_score_q'])

    
	with open('./results/{}/classification_report.json'.format(config['model']), 'w') as f:
		json.dump(avg_report, f, indent=4)