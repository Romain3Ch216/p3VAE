import pdb

from models.models import Model
import torch
from torch.nn.modules.loss import _Loss
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from models.utils import cycle_loader, patch_data_loader_from_image


class SegmentationModel(Model):
    def __init__(self, net, losses: Dict[(str, _Loss)], optim: torch.optim.Optimizer, config):
        super(Model, self).__init__()

        self.net = net
        self.losses = losses
        self.optim = optim
        self.config = config
        self.device = config['device']

        config['loss_'] = 'F1-score'
        config['direction'] = -1
        super().__init__(config)

        self.logs = {
            'train': {
                'Lc': [],
                'Lr': []
            },
            'val': {
                'Lc': [],
                'Lr': [],
                'Accuracy': [],
                'F1-score': []
            }
        }

    def optimize(self, labeled_data, unlabeled_data, val_data, config, logger):
        logs = self.init_logs()
        for epoch in range(1, config['epochs']+1):
            data_loader = cycle_loader(labeled_data, unlabeled_data)
            for data in tqdm(data_loader):
                ((x, y), (u, y_u)) = data
                x = x.float()
                u = u.float()
                u = u.to(self.device)
                x, y = x.to(self.device), y.to(self.device)
                r, logits = self.net(x)
                u_r, _ = self.net(u)
                logits = logits.permute(0, 2, 3, 1)
                mask = y != 0
                u_mask = y_u != 0
                if mask.sum() > 0:
                    sup_loss = self.losses['supervised'](logits[mask], y[mask]-1)
                    sup_loss.backward(retain_graph=True)
                unsup_loss = 0.5 * (self.losses['unsupervised'](x[mask], r[mask]) + self.losses['unsupervised'](u[u_mask], u_r[u_mask]))
                unsup_loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                self.logs['train']['Lc'].append(sup_loss.item())
                self.logs['train']['Lr'].append(unsup_loss.item())


            val_sup_loss, val_unsup_loss = [], []
            y_pred = []
            y_true = []
            for x, y in tqdm(val_data):
                x = x.float()
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    r, logits = self.net(x)
                logits = logits.permute(0, 2, 3, 1)
                mask = y != 0
                sup_loss = self.losses['supervised'](logits[mask], y[mask]-1)
                unsup_loss = self.losses['unsupervised'](x[mask], r[mask])

                y_true.extend((y[mask]-1).cpu().numpy())
                y_pred.extend(torch.argmax(logits[mask], dim=1).cpu().numpy())
                val_sup_loss.append(sup_loss.item())
                val_unsup_loss.append(unsup_loss.item())


            self.logs['val']['Accuracy'].append(accuracy_score(y_true, y_pred))
            self.logs['val']['F1-score'].append(f1_score(y_true, y_pred, average='micro'))
            self.logs['val']['Lc'].append(sum(val_sup_loss)/len(val_sup_loss))
            self.logs['val']['Lr'].append(sum(val_unsup_loss)/len(val_unsup_loss))

            self.feed_logger(logs, epoch, logger)
        return logs

    def inference(self, loader, with_labels=True):
        y_pred, y_true = [], []
        for data in tqdm(loader):
            try:
                x, y = data
            except:
                x = data[0]
                y = torch.ones((x.shape[:-1])).long()
            x = x.float()
            x, y = x.to(self.device), y.to(self.device)
            logits = self.inference_on_patch(x)
            mask = y != 0
            pred = torch.argmax(logits[mask], dim=-1)
            y_pred.append(pred.cpu())
            y_true.append(y[mask].cpu())
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        if with_labels:
            return y_true, y_pred
        else:
            return y_pred

    def inference_on_patch(self, patch):
        with torch.no_grad():
            r, logits = self.net(patch)
        logits = logits.permute(0, 2, 3, 1)
        return logits

    def inference_on_image(self, image, config):
        patch_size = config['patch_size']
        image = image.to(self.device).float()
        logits = torch.zeros((image.shape[0], image.shape[1], config['n_classes']))
        mask_overlap = torch.zeros((image.shape[:-1]))
        for patch, top, left in patch_data_loader_from_image(image, patch_size):
            patch = patch.unsqueeze(0)
            logits[top: top + patch_size, left: left + patch_size] = self.inference_on_patch(patch).squeeze(0)
            mask_overlap[top: top + patch_size, left: left + patch_size] += 1

        logits = logits / mask_overlap.unsqueeze(-1)
        pred = torch.argmax(logits, dim=-1)
        return pred












