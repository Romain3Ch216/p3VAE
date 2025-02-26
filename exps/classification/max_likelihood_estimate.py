# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to reproduce figure 7

import torch   
import numpy as np
from data import spectra_bbm, SimulatedDataSet
from models.model_loader import load_model
import json  
import matplotlib.pyplot as plt 
import sys
from models.utils import one_hot


def extreme_values_(dataset, gt, z_phi, z_eta):
	v = {}
	for class_id in dataset.classes:
		if class_id != 0:
			z1 = z_phi[gt==class_id]
			z2 = z_eta[gt==class_id]
			v[class_id] = {'z_phi': {'min': z1.min(), 'avg': z1.mean(), 'max': z1.max()}, 
			               'z_eta': {'min': z2.min(), 'avg': z2.mean(), 'max': z2.max()}}
	return v 

def generate(model, class_id, z_phi, z_eta):
    y = torch.zeros((1, model.y_dim))
    y[0, class_id-1] = 1.
    z_phi = torch.ones((1, 1))*z_phi
    z_eta = torch.ones((1, 1))*z_eta
    z = torch.cat((z_phi, z_eta), dim=1)
    with torch.no_grad():
        s = model.decoder(z, y)
    return s.view(-1)


def plot_from_latent(save, model, dataset, class_id, extreme_values, z_min, z_max, n_samples=25, fontsize=3.5, rho_max=0.9):
	z_phi = torch.linspace(extreme_values[class_id]['z_phi']['min'],extreme_values[class_id]['z_phi']['max'], int(n_samples**0.5))
	alpha = torch.linspace(0,1, int(n_samples**0.5))
	grid_x, grid_y = torch.meshgrid(z_phi, alpha, indexing='ij')
	fig, ax = plt.subplots(grid_x.shape[0], grid_x.shape[1])
	for i in range(grid_x.shape[0]):
		for j in range(grid_x.shape[1]):
			z1 = torch.tensor([grid_x[i,j]]).unsqueeze(0)
			z2 = torch.tensor([grid_y[i,j]])
			z2 = z2*z_min + (1-z2)*z_max
			z2 = z2.unsqueeze(0)
			sp = generate(model, class_id, z1, z2)
			sp = spectra_bbm(sp.reshape(1,-1), dataset.bbl).reshape(-1)
			ax[i,j].plot(sp, color='black', lw=0.6)
			ax[i,j].set_ylim(0,rho_max)
			ax[i,j].grid(True, linestyle='--', lw=0.2, alpha=0.5)
			y_labels = np.linspace(0,rho_max,4)
			y_labels = [round(e, 2) for e in y_labels]
			y_ticks = np.arange(len(y_labels))
			x_ticks = list(range(0,len(dataset.bbl),max(1,len(dataset.bbl)//10)))
			xlabels = []
			ax[i,j].set_xticks(x_ticks)
			for label in ax[i,j].get_yticklabels():
				label.set_fontsize(fontsize)
			for e in x_ticks:
			    xlabels.append(round(dataset.wv[e],2))
			ax[i,j].set_xticklabels(xlabels, rotation=45, fontsize=fontsize)	
			if i == grid_x.shape[0]-1 and j == grid_x.shape[1]//2:
				ax[i,j].set_xlabel("Wavelength (µm)", fontsize=10)
			if i!=grid_x.shape[0]-1:
				ax[i,j].tick_params(labelbottom=False) 
			if j==0 and i == grid_x.shape[0]//2:
				ax[i,j].set_ylabel('Reflectance', fontsize=10)
			if j!=0:
				ax[i,j].tick_params(labelleft=False)    

	plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.05)


def plot_GAN_sample_from_latent(save, model, dataset, class_id, z1, z2, n_samples=5, fontsize=3.5, rho_max=0.9):
	alpha = torch.linspace(0, 1, n_samples)
	# fig, ax = plt.subplots(1, n_samples, figsize=(10, 2))
	fig, ax = plt.subplots(int(n_samples**0.5), int(n_samples**0.5))
	torch.manual_seed(np.random.randint(10**4))
	z_noise = torch.randn(1, model.config['noise_dim'], device=model.config['device'])
	k = 0
	N = int(n_samples**0.5)
	for i in range(N):
		for j in range(N):
			z = alpha[k] * z1 + (1 - alpha[k]) * z2
			k += 1
			noise, _ = model.sample_noise(1, class_id, z=z_noise, con_c=z)
			with torch.no_grad():
				sp = model.netG(noise).cpu().numpy()
			sp = spectra_bbm(sp.reshape(1, -1), dataset.bbl).reshape(-1)
			ax[i,j].plot(sp, color='black', lw=0.6)
			ax[i,j].set_ylim(0,rho_max)
			ax[i,j].grid(True, linestyle='--', lw=0.2, alpha=0.5)
			y_labels = np.linspace(0, rho_max, 4)
			y_labels = [round(e, 2) for e in y_labels]
			y_ticks = np.arange(len(y_labels))
			x_ticks = list(range(0,len(dataset.bbl),max(1,len(dataset.bbl)//10)))
			xlabels = []
			ax[i,j].set_xticks(x_ticks)
			for label in ax[i,j].get_yticklabels():
				label.set_fontsize(fontsize)
			for e in x_ticks:
			    xlabels.append(round(dataset.wv[e],2))
			ax[i,j].set_xticklabels(xlabels, rotation=45, fontsize=fontsize)
			if i == N-1 and j == N//2:
				ax[i,j].set_xlabel("Wavelength (µm)", fontsize=10)
			if i!=N-1:
				ax[i,j].tick_params(labelbottom=False)
			if j==0 and i == N//2:
				ax[i,j].set_ylabel('Reflectance', fontsize=10)
			if j!=0:
				ax[i,j].tick_params(labelleft=False)

	plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.05)


def plot_gaussian_VAE_sample_from_latent(save, model, dataset, class_id, z1, z2, n_samples=5, fontsize=3.5, rho_max=0.9):
	alpha = torch.linspace(0, 1, n_samples)
	fig, ax = plt.subplots(int(n_samples**0.5), int(n_samples**0.5))
	k = 0
	y = one_hot(torch.tensor([class_id]).long(), model.n_classes)
	N = int(n_samples**0.5)
	for i in range(N):
		for j in range(N):
			z = alpha[k] * z1 + (1 - alpha[k]) * z2
			k += 1
			with torch.no_grad():
				sp = model.decoder(z, y).cpu().numpy()
			sp = spectra_bbm(sp.reshape(1, -1), dataset.bbl).reshape(-1)
			ax[i,j].plot(sp, color='black', lw=0.6)
			ax[i,j].set_ylim(0,rho_max)
			ax[i,j].grid(True, linestyle='--', lw=0.2, alpha=0.5)
			y_labels = np.linspace(0, rho_max, 4)
			y_labels = [round(e, 2) for e in y_labels]
			y_ticks = np.arange(len(y_labels))
			x_ticks = list(range(0,len(dataset.bbl),max(1,len(dataset.bbl)//10)))
			xlabels = []
			ax[i,j].set_xticks(x_ticks)
			for label in ax[i,j].get_yticklabels():
				label.set_fontsize(fontsize)
			for e in x_ticks:
			    xlabels.append(round(dataset.wv[e],2))
			ax[i,j].set_xticklabels(xlabels, rotation=45, fontsize=fontsize)
			if i == N-1 and j == N//2:
				ax[i,j].set_xlabel("Wavelength (µm)", fontsize=10)
			if i!=N-1:
				ax[i,j].tick_params(labelbottom=False)
			if j==0 and i == N//2:
				ax[i,j].set_ylabel('Reflectance', fontsize=10)
			if j!=0:
				ax[i,j].tick_params(labelleft=False)

	plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.05)


def save_fig(img, save):
	fig = plt.figure()
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0.05)

if __name__ == "__main__":

	results_path = sys.argv[1]
	with open(results_path + '/config.json') as f:
		config = json.load(f)

	config['device'] = 'cpu'
	dataset = SimulatedDataSet()
	model = load_model(dataset, config)
	checkpoint = torch.load(results_path + '/best_model.pth.tar')
	model.load_state_dict(checkpoint['state_dict'])
	img = torch.from_numpy(dataset.train_img).float()

	if config['model'] in ['p3VAE', 'guided']:
		pred, Lr, rec, z_P_std, z_P, z_A = model.inference_on_image(img, config, mode='argmax_p_y_x')

		extreme_values = extreme_values_(dataset, dataset.train_gt + dataset.val_gt, z_P, z_A)

		z_min, z_max, class_id = z_A[18, 63, :], z_A[62, 68, :], 4
		plot_from_latent('./results/simulation/{}/Figures/generated_spectra_tile.pdf'.format(config['model']),
						 model,
						 dataset,
						 class_id,
						 extreme_values,
						 z_min,
						 z_max)

		# z_min, z_max, class_id = z_A[13, 18, :], z_A[66, 45, :], 1
		# plot_from_latent(
		# 	'./results/simulation/{}/Figures/generated_spectra_vegetation.pdf'.format(config['model']),
		# 	model,
		# 	dataset,
		# 	class_id,
		# 	extreme_values,
		# 	z_min,
		# 	z_max)

	elif config['model'] == 'gaussian':
		pred, Lr, rec, z_P_std, z_P, z_A = model.inference_on_image(img, config, mode='argmax_p_y_x')

		z = torch.cat((z_P.unsqueeze(-1), z_A), dim=-1)

		z1, z2, class_id = z[13, 18, :].unsqueeze(0), z[66, 45, :].unsqueeze(0), 3
		plot_gaussian_VAE_sample_from_latent(
			'./results/simulation/{}/Figures/generated_spectra_tile.pdf'.format(config['model']),
			model,
			dataset,
			class_id,
			z1,
			z2,
			n_samples=25
		)

	elif config['model'] == 'ssInfoGAN':
		pred, z = model.inference_on_image(img, config)

		for i in range(z.shape[-1]):
			fig = plt.figure()
			plt.imshow(z[:,:,i])
			plt.show()

		z1, z2, class_id = z[13, 18, :].unsqueeze(0), z[66, 45, :].unsqueeze(0), 3
		plot_GAN_sample_from_latent(
			'./results/simulation/{}/Figures/generated_spectra_tile.pdf'.format(config['model']),
			model,
			dataset,
			class_id,
			z1,
			z2,
			n_samples=25
		)



	else:
		print("{} is not a valid generative model.".format(config['model']))