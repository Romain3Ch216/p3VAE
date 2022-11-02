# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to train a model

import argparse
from tqdm import tqdm
from itertools import cycle
import numpy as np
from utils import *
from models.models import load_model
from data import load_dataset  
import pdb
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help="A model among {'baseline', 'vae', 'hybrid'}")
parser.add_argument('--device', type=str, default='cpu', help="Specify cpu or gpu")
parser.add_argument('--dataset', type=str, default='simulation', help="simulation or real")


# Training options
training_options = parser.add_argument_group('Training')
training_options.add_argument('--epochs', type=int, default=100)
training_options.add_argument('--lr', type=float, default=1e-4)
training_options.add_argument('--batch_size', type=int, default=64)
training_options.add_argument('--z_eta_dim', type=int, default=4)
training_options.add_argument('--h_dim', type=int, default=256)
training_options.add_argument('--lambda_sam', type=float, default=1)
training_options.add_argument('--beta', type=float, default=1e-4)
training_options.add_argument('--lambda_classifier', type=float, default=1e-2)
training_options.add_argument('--lambda_encoder', type=float, default=1e-2)
training_options.add_argument('--lambda_entropy', type=float, default=1e-2)
training_options.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
config = parser.parse_args()
config = vars(config)

if config['seed'] == 0:
    config['seed'] = np.random.randint(1e5)

dataset = load_dataset(config)
if config['model'] == 'CNN_full_annotations':
    train_data, val_data = dataset.load(dataset.train_img, dataset.train_gt+dataset.val_gt, batch_size=config['batch_size'], unlabeled=False, test=False, split=True)
else:
    labeled_data = dataset.load(dataset.train_img, dataset.train_gt, batch_size=config['batch_size'], unlabeled=False, test=False)
    unlabeled_data = dataset.load(dataset.train_img, dataset.train_gt, batch_size=config['batch_size'], unlabeled=True, test=False)
    val_data = dataset.load(dataset.train_img, dataset.val_gt, batch_size=config['batch_size'], unlabeled=False, test=True)

config['n_classes'] = len(np.unique(dataset.train_gt))-1
config['n_channels'] = dataset.n_channels


model, optimizer, scheduler = load_model(dataset, config)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO,
                    filename=model.path + 'log.txt',
                    filemode='w')

if config['model'] == 'CNN_full_annotations':
    def loader(labeled_data, unlabeled_data):
        return labeled_data
    labeled_data = train_data
    unlabeled_data = None
    n_loader = len(labeled_data)
else:
    def loader(labeled_data, unlabeled_data):
        return zip(cycle(labeled_data), unlabeled_data)
    n_loader = len(unlabeled_data)

try:    
    logs = model.logs_()
    for epoch in range(1, config['epochs']+1):
        model.train()
        for data in tqdm(loader(labeled_data, unlabeled_data), total=n_loader, desc='Epoch {}'.format(epoch)):
            try:
                (x, y), (u, _) = data
            except:
                x, y = data
                u = x
            x, y, u = x.to(config['device']), y.to(config['device']), u.to(config['device'])
            Ll, Lu = model.loss(x, y, u, 
                                lambda_entropy=config['lambda_entropy'],
                                lambda_encoder=config['lambda_encoder'],
                                lambda_classifier=config['lambda_classifier'],
                                lambda_sam=config['lambda_sam'],
                                beta=config['beta'])

            
            if config['model'] in ['p3VAE', 'guided', 'gaussian', 'p3VAE_no_gs']:
                Lu.backward(retain_graph=True)
                if config['model'] in ['p3VAE', 'guided']:
                    for param in model.decoder.parameters():
                        if param.requires_grad:
                            param.grad.zero_()

            Ll.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        y_true, y_pred = [], []
        for (x, y) in val_data:
            x, y = x.to(config['device']), y.to(config['device'])
            y_true_, y_pred_ = model.validation(x, y, lambda_sam=config['lambda_sam'])
            y_true.extend(y_true_)
            y_pred.extend(y_pred_)

        logs = model.update_logs(logs, y_true, y_pred)
        train_log, val_log = model.print(logs)
        logger.info(train_log)
        logger.info(val_log)

except KeyboardInterrupt:
    # Allow the user to stop the training
    pass
