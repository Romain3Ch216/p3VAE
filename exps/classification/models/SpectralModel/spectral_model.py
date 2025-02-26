import pdb

from models.models import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Dict
from models.utils import accuracy_metrics
from tqdm import tqdm


class SpectralModel(Model):
    """
    """
    def __init__(self,
                 encoder: nn.Module,
                 classifier: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 config: Dict):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.y_dim = self.classifier.y_dim
        self.loss_ = 'Lc'
        self.direction = 1
        self.device = config['device']
        config['loss_'] = self.loss_
        config['direction'] = self.direction
        super().__init__(config)

        self.logs = {
            'train': {
                'Lc': [],
                'Accuracy': [],
                'F1-score': []
            },

            'val': {
                'Lc': [],
                'Accuracy': [],
                'F1-score': []
            }
        }

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                if config['seed']:
                    torch.manual_seed(config['seed'])
                    torch.cuda.manual_seed(config['seed'])
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def optimize(self, labeled_data_loader, unlabeled_data_loader, val_data_loader, config, logger):
        self.train()
        logs = self.init_logs()
        for epoch in range(1, config['epochs']+1):
            for data in tqdm(labeled_data_loader, total=len(labeled_data_loader), desc='Training - epoch {}'.format(epoch)):
                x, y = data
                x, y = x.to(config['device']), y.to(config['device'])
                y = y - 1

                logits = self(x)
                pred = torch.argmax(logits, dim=-1)
                loss = F.cross_entropy(logits, y) + config['lambda_classifier'] * self.classifier.parent_module.regularization()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                acc, f1_score = accuracy_metrics(y, pred)
                self.logs['train']['Lc'].append(loss.item())
                self.logs['train']['Accuracy'].append(acc)
                self.logs['train']['F1-score'].append(f1_score)

            for data in tqdm(val_data_loader, total=len(val_data_loader), desc='Validation - epoch {}'.format(epoch)):
                x, y = data
                x, y = x.to(config['device']), y.to(config['device'])
                y = y - 1

                with torch.no_grad():
                    logits = self(x)
                    pred = torch.argmax(logits, dim=-1)
                    loss = F.cross_entropy(logits, y) + config['lambda_classifier'] * self.classifier.parent_module.regularization()

                acc, f1_score = accuracy_metrics(y, pred)
                self.logs['val']['Lc'].append(loss.item())
                self.logs['val']['Accuracy'].append(acc)
                self.logs['val']['F1-score'].append(f1_score)

            self.feed_logger(logs, epoch, logger)

        return logs

    def inference(self, data_loader):
        pred = []
        for data in tqdm(data_loader, total=len(data_loader), desc='Inference'):
            try:
                x, _ = data
            except:
                x = data[0]
            x = x.to(self.device)
            pred.append(self.inference_on_batch(x))
        pred = torch.cat(pred)
        return pred

    def inference_on_batch(self, data):
        with torch.no_grad():
            logits = self(data)
        pred = torch.argmax(logits, dim=-1)
        return pred

    def regularization(self, norm="L2"):
        return self.classifier.regularization(norm)

    def forward(self, x, reduction=True):
        latent = self.encoder(x)
        logits = self.classifier(latent, reduction=reduction)
        return logits
