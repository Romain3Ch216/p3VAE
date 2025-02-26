# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
import errno
import torch
import torch.nn as nn
import numpy as np
import json
import os
from models.utils import data_loader_from_image


class Model(nn.Module):
    """
    Generic model
    """
    def __init__(self, config):
        self.path = './results/{}/{}/{}/'.format(config['dataset'], config['model'], config['seed'])
        self.best_loss = np.inf * config['direction']
        self.direction = config['direction']
        self.loss_ = config['loss_']
        self.n_classes = config['n_classes']

        try:
            os.makedirs(self.path, exist_ok=True)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        with open(self.path + 'config.json', 'w') as outfile:
            json.dump(config, outfile, indent=4)

    def init_logs(self):
        logs = {'train': {}, 'val': {}}
        for metric in self.logs['train']:
            logs['train'][metric] = []
        if 'val' in self.logs:
            for metric in self.logs['val']:
                logs['val'][metric] = []
        return logs

    def update_logs(self, logs, epoch):
        for metric in self.logs['train']:
            logs['train'][metric].append(sum(self.logs['train'][metric])/len(self.logs['train'][metric]))
            self.logs['train'][metric] = []

        if 'val' in self.logs:
            for metric in self.logs['val']:
                logs['val'][metric].append(sum(self.logs['val'][metric])/len(self.logs['val'][metric]))
                self.logs['val'][metric] = []

            if logs['val'][self.loss_][-1] < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = logs['val'][self.loss_][-1]
                torch.save({'epoch': len(logs['val']['F1-score']), 'best_loss': self.best_loss,\
                            'state_dict': self.state_dict()}, self.path + 'best_model.pth.tar')
        return logs

    def print_logs(self, logs, epoch):
        train_log = "[Train - epoch {}]\t ".format(epoch)
        for metric in logs['train']:
            train_log = train_log + "{}: {:.2e}, ".format(metric, logs['train'][metric][-1])

        val_log = "[Val - epoch {}]\t ".format(epoch)
        if 'val' in logs:
            for metric in logs['val']:
                val_log = val_log + "{}: {:.2e}, ".format(metric, logs['val'][metric][-1])
        return train_log, val_log

    def feed_logger(self, logs, epoch, logger):
        logs = self.update_logs(logs, epoch)
        train_log, val_log = self.print_logs(logs, epoch)
        logger.info(train_log)
        logger.info(val_log)

    def optimize(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def inference_on_image(self, image, config):
        data_loader = data_loader_from_image(image, config['batch_size'])
        pred = self.inference(data_loader)
        pred = pred.view(image.shape[0], image.shape[1])
        return pred

