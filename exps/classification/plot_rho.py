# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to reproduce figure 6

import torch   
import torch.nn.functional as F 
from data import spectra_bbm, load_dataset
from models import * 
from utils import * 
from models.models import load_model
import json  
import matplotlib.pyplot as plt 
import os 
import sys 

def plot_rho(dataset, class_id, model, save):
	colors = dataset.palette
	y = torch.zeros((1, model.y_dim))
	y[0, class_id-1] = 1.
	with torch.no_grad():
		rho = model.decoder.rho_(y)
	fig = plt.figure(figsize=(15,15))
	for num_sp in range(len(dataset.classes[class_id]['spectrum'])):
		sp = dataset.classes[class_id]['spectrum'][num_sp].reshape(1,-1)
		sp = spectra_bbm(sp, dataset.bbl).reshape(-1)
		plt.plot(dataset.wv, sp, linestyle='--', color='black')
	for i in range(rho.shape[1]):
		sp = spectra_bbm(rho[:,i,:], dataset.bbl).reshape(-1)
		plt.plot(dataset.wv, sp, color=colors[i+1])
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.xlabel(r'Wavelenght ($\mu m$)', fontsize=25)
	plt.ylabel('Reflectance', fontsize=25)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.savefig(save, dpi=200, bbox_inches = 'tight', pad_inches = 0.05)

if __name__ == "__main__":

	results_path = sys.argv[1]

	with open(results_path + '/config.json') as f:
		config = json.load(f)

	dataset = load_dataset(config)	
	model, _, _ = load_model(dataset, config)
	checkpoint = torch.load(results_path + '/best_model.pth.tar')
	model.load_state_dict(checkpoint['state_dict'])

	for class_id in range(1, model.y_dim+1):
		plot_rho(dataset, class_id, model, save=results_path+'/hat_rho_{}.pdf'.format(class_id))