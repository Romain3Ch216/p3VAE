# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to reproduce figure 7

import torch   
import torch.nn.functional as F 
from models.utils import sam_
from data import spectra_bbm, load_dataset
from models import * 
from utils import * 
from models.models import load_model
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import json  
import matplotlib.pyplot as plt 
import os 
import sys 

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

	plt.savefig(save, dpi=200, bbox_inches = 'tight', pad_inches = 0.05)


def save_fig(img, save):
	fig = plt.figure()
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(save, dpi=200, bbox_inches = 'tight', pad_inches = 0.05)

if __name__ == "__main__":

	results_path = sys.argv[1]
	with open(results_path + '/config.json') as f:
		config = json.load(f)

	dataset = load_dataset(config)
	model, _, _ = load_model(dataset, config)
	checkpoint = torch.load(results_path + '/best_model.pth.tar')
	model.load_state_dict(checkpoint['state_dict'])
	try:
		scene = dataset.load(dataset.train_img, np.ones_like(dataset.train_gt), batch_size=config['batch_size'], unlabeled=False, test=True)
		H, W = dataset.train_gt.shape
		rec, obs, rho, pred, z_phi, z_eta, Lr, entropy = model.map(scene, config)
		z_phi, z_eta = z_phi.view((H, W)), z_eta.view((H, W, -1))

		extreme_values = extreme_values_(dataset, dataset.train_gt + dataset.val_gt, z_phi, z_eta)

		z_min, z_max, class_id = z_eta[18, 63, :], z_eta[62, 68, :], 4
		plot_from_latent('./results/{}/Figures/generated_spectra_vegetation.pdf'.format(config['model'], class_id), model, dataset, class_id, extreme_values, z_min, z_max)

		z_min, z_max, class_id = z_eta[13, 18, :], z_eta[66, 45, :], 4
		plot_from_latent('./results/{}/Figures/generated_spectra_vegetation.pdf'.format(config['model'], class_id), model, dataset, class_id, extreme_values, z_min, z_max)
	except:
		print("{} is not a valid generative model.".format(config['model']))