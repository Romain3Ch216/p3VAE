import pdb

from models.models import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sklearn.metrics import accuracy_score, f1_score
from models.utils import one_hot, sam_, Lr_, enumerate_discrete, cycle_loader, accuracy_metrics, data_loader_from_image
from tqdm import tqdm
from typing import Dict


class SemiSupervisedVAE(Model):
    """
    Generic model for a semi-supervised VAE architecture
    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 classifier: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 config: Dict):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.y_dim = self.classifier.y_dim
        self.optimizer = optimizer
        self.scheduler = scheduler

        if config['model'] == 'gaussian':
            self.loss_ = 'Lc'
        else:
            self.loss_ = 'Lr'
        self.direction = 1

        config['loss_'] = self.loss_
        config['direction'] = self.direction
        super().__init__(config)

        self.logs = {
            'train': {
                'Lr_l': [],
                'Lr_u': [],
                'Lc': [],
                'Entropy': []
            },
            'val': {
                'Lr': [],
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
        for epoch in range(1, config['epochs'] + 1):
            for data in tqdm(
                    cycle_loader(labeled_data_loader, unlabeled_data_loader),
                    total=len(unlabeled_data_loader),
                    desc='Training - epoch {}'.format(epoch)
            ):
                (x, y), (u, _) = data
                x, y, u = x.to(config['device']), y.to(config['device']), u.to(config['device'])
                Ll, Lu, reg, Lr_l, Lr_u, Lc, entropy = self.train_loss(x, y, u,
                                   lambda_entropy=config['lambda_entropy'],
                                   lambda_encoder=config['lambda_encoder'],
                                   lambda_sam=config['lambda_sam'],
                                   lambda_classifier=config['lambda_classifier'],
                                   beta=config['beta'])

                self.logs['train']['Lr_l'].append(Lr_l)
                self.logs['train']['Lr_u'].append(Lr_u)
                self.logs['train']['Lc'].append(Lc)
                self.logs['train']['Entropy'].append(entropy)

                if config['lambda_classifier'] > 0:
                    reg.backward(retain_graph=True)

                if config['model'] in ['p3VAE', 'p3VAE_no_gs', 'p3VAE_g', 'guided', 'guided_no_gs', 'gaussian']:
                    Lu.backward(retain_graph=True)
                    if config['model'] in ['p3VAE', 'p3VAE_g', 'guided']:
                        for param in self.decoder.parameters():
                            if param.requires_grad:
                                param.grad.zero_()

                Ll.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            for data in tqdm(
                    val_data_loader,
                    total=len(val_data_loader),
                    desc='Validation - epoch {}'.format(epoch)
            ):
                x, y = data
                x, y = x.to(config['device']), y.to(config['device'])
                Lr, Lc, pred = self.val_loss(x, y)

                acc, f1_score = accuracy_metrics(y - 1, pred)

                self.logs['val']['Lr'].append(Lr)
                self.logs['val']['Lc'].append(Lc)
                self.logs['val']['Accuracy'].append(acc)
                self.logs['val']['F1-score'].append(f1_score)

            self.feed_logger(logs, epoch, logger)

        return logs

    def train_loss(self, x, y, u,
             lambda_entropy=0,
             lambda_encoder=0,
             lambda_sam=0,
             lambda_classifier=0,
             beta=0):

        self.train()
        y = one_hot(y - 1, self.y_dim).to(x.device)
        # ========================= Supervised part ============================ #
        reconstruction_l = self(x, y)
        mse_l = F.mse_loss(x, reconstruction_l)  # log p(s|y,z)
        sam_l = sam_(x, reconstruction_l)
        Lr_l = mse_l + lambda_sam * sam_l

        logits_l = self.q_y_x_batch(x)
        Lc_l = F.cross_entropy(logits_l, torch.argmax(y, dim=-1))  # log q(y|s)

        # KL Divergence
        kld_l = self.encoder.kl_divergence()  # log p(z_phi) + log p(z_eta) - log q(z_phi,z_eta|s,y)

        # Partial supervised loss
        L_l = Lr_l + Lc_l + beta * kld_l + lambda_encoder * self.encoder.regularization()

        # Accuracy
        # accuracy = torch.mean((torch.max(logits_l, 1)[1].data == torch.max(y, 1)[1].data).float())

        # ======================= Unsupervised part ============================ #
        y_u = enumerate_discrete(u, self.y_dim)
        batch_size = u.shape[0]
        logits_u = self.q_y_x_batch(u)
        probs_u = F.softmax(logits_u, dim=-1)

        # Entropy of the classifier
        H = -torch.sum(torch.mul(probs_u, torch.log(probs_u + 1e-8)), dim=-1)

        # Posterior
        probs_u = probs_u.repeat(self.y_dim, 1)
        posterior = probs_u[torch.arange(len(probs_u)), torch.argmax(y_u, dim=-1)]

        # Reconstruction
        u = u.repeat(self.y_dim, 1)
        z_u = self.encoder(u, y_u)
        reconstruction_u = self.decoder(z_u, y_u)
        mse_u = F.mse_loss(u, reconstruction_u, reduction='none')
        mse_u = torch.mean(mse_u, dim=-1) * posterior
        mse_u = mse_u.sum() / batch_size
        sam_u = sam_(u, reconstruction_u, reduction='none') * posterior
        sam_u = sam_u.sum() / batch_size

        # KL Divergence
        kld_u = self.encoder.kl_divergence(reduction='none') * posterior
        kld_u = kld_u.sum() / batch_size

        Lr_u = mse_u + lambda_sam * sam_u
        L_u = Lr_u + beta * kld_u - lambda_entropy * H.mean()
        reg = lambda_classifier * self.classifier.regularization()

        return L_l, L_u, reg, Lr_l.item(), Lr_u.item(), Lc_l.item(), H.mean().item()

    def val_loss(self, x, y, lambda_sam=0):
        y = one_hot(y - 1, self.y_dim).to(x.device)
        with torch.no_grad():
            reconstruction_l = self(x, y)
        mse_l = F.mse_loss(x, reconstruction_l)
        sam_l = sam_(x, reconstruction_l)
        Lr_l = mse_l + lambda_sam * sam_l

        with torch.no_grad():
            logits_l = self.q_y_x_batch(x)
        Lc_l = F.cross_entropy(logits_l, torch.argmax(y, dim=-1))

        # Accuracy
        pred = torch.argmax(logits_l, dim=-1)

        return Lr_l.item(), Lc_l.item(), pred

    def inference(self, data_loader, config, mode='q_y_x'):
        if mode == 'argmax_p_y_x':
            LR, PRED, REC, Z_P_STD, R_Z_P, R_Z_A = [], [], [], [], [], []
            for data in data_loader:
                try:
                    x, _ = data
                except:
                    x = data[0]
                x = x.to(config['device'])
                with torch.no_grad():
                    Lr, pred, z_P_std, random_z_P, random_z_A = self.argmax_p_y_x_batch(x, config)
                    rec = self.reconstruction(x, pred)
                LR.append(Lr)
                PRED.append(pred)
                REC.append(rec)
                Z_P_STD.append(z_P_std)
                R_Z_P.append(random_z_P)
                R_Z_A.append(random_z_A)
            Lr = torch.cat(LR)
            pred = torch.cat(PRED)
            z_P_std = torch.cat(Z_P_STD)
            random_z_P = torch.cat(R_Z_P)
            random_z_A = torch.cat(R_Z_A)
            rec = torch.cat(REC)
            return pred, Lr, rec, z_P_std, random_z_P, random_z_A

        elif mode == 'q_y_x':
            PRED, REC = [], []
            for data in data_loader:
                try:
                    x, _ = data
                except:
                    x = data[0]
                x = x.to(config['device'])
                with torch.no_grad():
                    logits = self.q_y_x_batch(x)
                    pred = torch.argmax(logits, dim=-1)
                    rec = self.reconstruction(x, pred)
                PRED.append(pred)
                REC.append(rec)
            pred = torch.cat(PRED)
            rec = torch.cat(REC)
            return pred, rec

    def argmax_p_y_x_batch(self, batch, config, num_samples=5):
        self.train()
        likelihood = torch.zeros((batch.shape[0], self.y_dim))
        z_Ps = torch.zeros((batch.shape[0], num_samples))

        for k in range(num_samples):
            with torch.no_grad():
                logits = self.q_y_x_batch(batch)
            probs = torch.softmax(logits, dim=-1)
            categorical = torch.distributions.categorical.Categorical(probs=probs)
            y_tmp = categorical.sample()
            y_tmp = one_hot(y_tmp, self.y_dim)
            with torch.no_grad():
                z = self.encoder(batch, y_tmp)
            z_Ps[:, k] = z[:, 0]
            try:
                log_q_z = self.encoder.q_z_phi.log_prob(z[:, 0]) + self.encoder.q_z_eta.log_prob(z[:, 1:])
                log_p_z = self.encoder.p_z_phi.log_prob(z[:, 0]) + self.encoder.p_z_eta.log_prob(z[:, 1:])
            except:
                log_q_z = torch.sum(self.encoder.q_z.log_prob(z), dim=1)
                log_p_z = torch.sum(self.encoder.p_z.log_prob(z), dim=1)

            ratio = torch.exp(log_p_z - log_q_z)

            for class_id in range(self.y_dim):
                y = torch.zeros((batch.shape[0], self.y_dim))
                y[torch.arange(batch.shape[0]), class_id] = 1.

                with torch.no_grad():
                    rec = self.decoder(z, y)

                p_x_z_y = torch.exp(-(
                        torch.mean(F.mse_loss(batch, rec, reduction='none'), dim=-1) + config['lambda_sam'] * sam_(
                    batch, rec, reduction='none')))
                likelihood[:, class_id] += ratio * p_x_z_y

        Lr, pred = torch.max(likelihood, dim=-1)
        z_P_std = torch.std(z_Ps, dim=-1)
        random_z_P = z[:, 0]
        random_z_A = z[:, 1:]
        return Lr, pred, z_P_std, random_z_P, random_z_A

    def q_y_x_batch(self, batch, reduction=True):
        logits = self.classifier(batch, reduction=reduction)
        return logits

    def reconstruction(self, x, pred):
        y = one_hot(pred, self.y_dim)
        rec = self(x, y)
        return rec

    def empirical_z_std(self, batch, config, num_samples=10):
        self.train()
        batch = batch.to(config['device'])
        Z = torch.zeros((batch.shape[0], num_samples))

        for k in range(num_samples):
            logits = self.q_y_x_batch(batch, config)
            probs = torch.softmax(logits, dim=-1)
            categorical = torch.distributions.categorical.Categorical(probs=probs)
            y = categorical.sample()
            y = one_hot(y, self.y_dim)
            with torch.no_grad():
                Z[:, k] = self.encoder(batch, y)[:, 0]
        return torch.std(Z, dim=-1)

    def inference_on_image(self, image, config, mode='argmax_p_y_x'):
        data_loader = data_loader_from_image(image, config['batch_size'])
        pred, Lr, rec, z_P_std, random_z_P, random_z_A = self.inference(data_loader, config, mode=mode)
        pred = pred.view(image.shape[0], image.shape[1])
        Lr = Lr.view(image.shape[0], image.shape[1])
        rec = rec.view(image.shape[0], image.shape[1], -1)
        z_P_std = z_P_std.view(image.shape[0], image.shape[1])
        random_z_P = random_z_P.view(image.shape[0], image.shape[1])
        random_z_A = random_z_A.view(image.shape[0], image.shape[1], -1)
        return pred, Lr, rec, z_P_std, random_z_P, random_z_A

    def forward(self, x, y):
        z = self.encoder(x, y)
        x = self.decoder(z, y)
        return x
