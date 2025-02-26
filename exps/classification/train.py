# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to train a model

import argparse
import numpy as np
from models.model_loader import load_model
from data import SimulatedDataSet, RealDataSet
import logging


def train(config):
    if config['seed'] == 0:
        config['seed'] = np.random.randint(1e5)

    if config['dataset'] == 'simulation':
        dataset = SimulatedDataSet()
    elif config['dataset'] == 'real_data':
        dataset = RealDataSet()
    else:
        raise Exception("dataset argument must either be 'simulation' or 'real_data'")

    if config['dataset'] == 'simulation':
        if config['model'] == 'CNN_full_annotations':
            labeled_dataset, val_dataset = dataset.load(dataset.train_img,
                                                        dataset.train_gt + dataset.val_gt,
                                                        batch_size=config['batch_size'],
                                                        unlabeled=False,
                                                        test=False,
                                                        split=True)
            unlabeled_dataset = None

        else:
            labeled_dataset = dataset.load(dataset.train_img,
                                           dataset.train_gt,
                                           batch_size=config['batch_size'],
                                           unlabeled=False,
                                           test=False)
            unlabeled_dataset = dataset.load(dataset.train_img,
                                             dataset.train_gt,
                                             batch_size=config['batch_size'],
                                             unlabeled=True,
                                             test=False)
            val_dataset = dataset.load(dataset.train_img,
                                       dataset.val_gt,
                                       batch_size=config['batch_size'],
                                       unlabeled=False,
                                       test=True)

        config['n_classes'] = len(np.unique(dataset.train_gt))-1

    elif config['dataset'] == 'real_data':
        if config['model'] == 'FG-Unet':
            labeled_dataset = dataset.patch_loader('labeled', batch_size=config['batch_size'])
            unlabeled_dataset = dataset.patch_loader('unlabeled', batch_size=config['batch_size'])
            val_dataset = dataset.patch_loader('unlabeled', batch_size=config['batch_size'])

        elif config['model'] == 'CNN_full_annotations':
            labeled_data = np.concatenate((dataset.labeled_train_data, dataset.unlabeled_train_data), axis=0)
            labels = np.concatenate((dataset.labeled_train_labels, dataset.unlabeled_train_labels), axis=0)
            labeled_dataset, val_dataset = dataset.load(labeled_data,
                                           labels,
                                           split=True,
                                           batch_size=config['batch_size'],
                                           test=False)
            unlabeled_dataset = None

        else:
            labeled_dataset = dataset.load(dataset.labeled_train_data,
                                           dataset.labeled_train_labels,
                                           batch_size=config['batch_size'],
                                           test=False)
            unlabeled_dataset = dataset.load(dataset.unlabeled_train_data,
                                             dataset.unlabeled_train_labels,
                                             batch_size=config['batch_size'],
                                             test=False)
            val_dataset = dataset.load(dataset.unlabeled_train_data,
                                       dataset.unlabeled_train_labels,
                                       batch_size=config['batch_size'],
                                       test=True)
        config['n_classes'] = len(dataset.classes)-1

    config['n_channels'] = dataset.n_channels
    model = load_model(dataset, config)

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        filename=model.path + 'logs.txt',
                        filemode='w')

    # ==================== Training ===================== #
    logs = model.optimize(labeled_dataset, unlabeled_dataset, val_dataset, config, logger)
    # =================================================== #
    logger.info('Best epoch: {}, best loss: {}'.format(model.best_epoch, model.best_loss))

    return logs


if __name__ == "__main__":
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
    training_options.add_argument('--beta_g', type=float)
    training_options.add_argument('--dropout', type=float, default=0.5)
    training_options.add_argument('--d_step', type=int, default=5, help='For GAN training, ...')
    training_options.add_argument('--noise_dim', type=int, default=30, help='For GAN training, ...')
    training_options.add_argument('--c_weight', type=float, default=1, help='For GAN training, ...')
    training_options.add_argument('--l1', type=float, default=0.25, help='For GAN training, ...')
    training_options.add_argument('--seed', type=int, default=0)

    config = parser.parse_args()
    config = vars(config)
    train(config)
