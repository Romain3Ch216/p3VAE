# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.

import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import json 
import numpy as np  
import os  
import logging  
from torch.nn import init 
from models.utils import HyperspectralWrapper, SpectralConvolution, View, one_hot, sam_, Lr_, enumerate_discrete
from models.encoders import PhysicsGuidedEncoder, GaussianEncoder
from models.decoders import HybridDecoder, PhysicsGuidedDecoder, GaussianDecoder
from models.consistent_dropout import MCConsistentDropoutModule 
from models.classifiers import DenseNeuralNetwork
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Model(nn.Module):
    """
    Generic model
    """
    def __init__(self, config):
        self.logs = None
        self.best_loss = np.inf
        self.path = './results/{}/{}/'.format(config['model'], config['seed'])

        try:
            os.makedirs(self.path, exist_ok=True)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        with open(self.path + 'config.json', 'w') as outfile:
            json.dump(config, outfile, indent=4)

    def logs_(self):
        logs = {'train': {}, 'val': {}}
        for metric in self.logs['train']:
            logs['train'][metric] = []
        for metric in self.logs['val']:
            logs['val'][metric] = []
        return logs

    def update_logs(self, logs, y_true, y_pred):
        for metric in self.logs['train']:
            logs['train'][metric].append(sum(self.logs['train'][metric])/len(self.logs['train'][metric]))
            self.logs['train'][metric] = []
        for metric in self.logs['val']:
            if metric not in ['Accuracy', 'F1-score']:
                logs['val'][metric].append(sum(self.logs['val'][metric])/len(self.logs['val'][metric]))
                self.logs['val'][metric] = []
        acc = accuracy_score(y_true, y_pred)
        score = f1_score(y_true, y_pred, average='macro')
        logs['val']['Accuracy'].append(acc)
        logs['val']['F1-score'].append(score)
        if logs['val'][self.loss_][-1] < self.best_loss:
            self.best_loss = logs['val'][self.loss_][-1]
            torch.save({'epoch': len(logs['val']['F1-score']), 'best_loss': self.best_loss,\
                        'state_dict': self.state_dict()}, self.path + 'best_model.pth.tar')
        return logs

    def print(self, logs):
        train_log = "[Train]\t "
        for metric in logs['train']:
            train_log = train_log + "{}: {:.2e}, ".format(metric, logs['train'][metric][-1])
        val_log = "[Val]\t "
        for metric in logs['val']:
            val_log = val_log + "{}: {:.2e}, ".format(metric, logs['val'][metric][-1])
        return train_log, val_log

class SemiSupervisedVAE(Model):
    """
    Generic model for a semi-supervised VAE architecture
    """
    def __init__(self, encoder, decoder, classifier, config):
        super(Model, self).__init__()
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.y_dim = self.classifier.y_dim
        self.seed = config['seed']
        self.loss_ = 'Lr'

        self.logs = {
            'train': {
                'Lr_l': [],
                'Lr_u': [],
                'Lc': [],
                'Entropy': [],
                'Accuracy': []
            },
            'val': {
                'Lr': [],
                'Entropy': [],
                'Accuracy': [],
                'F1-score': []
            }
        }

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                if self.seed:
                    torch.manual_seed(self.seed)
                    torch.cuda.manual_seed(self.seed)
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def loss(self, x, y, u, 
             lambda_entropy=0, 
             lambda_encoder=0, 
             lambda_classifier=0,
             lambda_sam=0, 
             beta=0, 
             target_entropy=0, 
             debug=False):

        self.train()
        y = one_hot(y-1, self.y_dim)
        #========================= Supervised part ============================#
        reconstruction_l = self(x, y)

        mse_l = F.mse_loss(x, reconstruction_l) # log p(s|y,z)
        sam_l = sam_(x, reconstruction_l)

        logits_l = self.classify(x)

        Lc_l = F.cross_entropy(logits_l, torch.argmax(y, dim=-1)) # log q(y|s)

        # KL Divergence
        kld_l = self.encoder.kl_divergence() # log p(z_phi) + log p(z_eta) - log q(z_phi,z_eta|s,y)

        # Partial supervised loss
        L_l = mse_l + lambda_sam*sam_l + Lc_l + beta*kld_l + lambda_encoder*self.encoder.regularization()

        # Accuracy
        accuracy = torch.mean((torch.max(logits_l, 1)[1].data == torch.max(y, 1)[1].data).float())

        #======================= Unsupervised part ============================#
        y_u = enumerate_discrete(u, self.y_dim)
        batch_size = u.shape[0]
        logits_u = self.classifier(u, reduction=False)
        expected_entropy, entropy_expected_p, bald = self.classifier.BALD(torch.softmax(logits_u, dim=1))
        logits_u = torch.mean(logits_u, dim=-1)
        logits_u = F.softmax(logits_u, dim=-1)

        # Entropy of the classifier
        H = -torch.sum(torch.mul(logits_u, torch.log(logits_u + 1e-8)), dim=-1)

        # Posterior
        logits_u = logits_u.repeat(self.y_dim, 1)
        posterior = logits_u[torch.arange(len(logits_u)), torch.argmax(y_u, dim=-1)]

        # Reconstruction
        u = u.repeat(self.y_dim, 1)
        z_u = self.encoder(u, y_u)
        reconstruction_u = self.decoder(z_u, y_u)

        mse_u = F.mse_loss(u, reconstruction_u, reduction='none')
        mse_u = torch.mean(mse_u, dim=-1)*posterior
        mse_u = mse_u.sum()/batch_size

        sam_u = sam_(u, reconstruction_u, reduction='none')*posterior
        sam_u = sam_u.sum()/batch_size

        # KL Divergence
        kld_u = self.encoder.kl_divergence(reduction='none')*posterior
        kld_u = kld_u.sum()/batch_size

        Lr_u = mse_u + lambda_sam*sam_u
        L_u = mse_u + lambda_sam*sam_u + beta*kld_u - lambda_entropy*H.mean()
        L = L_l + L_u + lambda_classifier*self.classifier.regularization() 

        self.logs['train']['Lr_l'].append((mse_l+lambda_sam*sam_l).item())
        self.logs['train']['Lr_u'].append((mse_u+lambda_sam*sam_u).item())
        self.logs['train']['Lc'].append(Lc_l.item())
        self.logs['train']['Entropy'].append(H.mean().item())
        self.logs['train']['Accuracy'].append(accuracy.item())

        return L_l, L_u

    def validation(self, x, y_true, lambda_sam=0):
        y_true = y_true - 1
        y = one_hot(y_true, self.y_dim)
        with torch.no_grad():
            reconstruction = self(x, y)
            logits = self.classify(x)
        logits = F.softmax(logits, dim=-1)
        y_pred = torch.argmax(logits, dim=-1)
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1).mean()
        Lr = Lr_(x, reconstruction, alpha=lambda_sam)
        self.logs['val']['Lr'].append(Lr.item())
        self.logs['val']['Entropy'].append(H.item())
        return y_true, y_pred

    def argmax_q_z_x_batch(self, batch, config, num_samples=5):
        self.train()
        batch = batch.to(config['device'])
        L = torch.zeros((batch.shape[0], self.y_dim))
        Z = torch.zeros((batch.shape[0], num_samples))

        for k in range(num_samples):
            with torch.no_grad():
                logits = self.classify(batch)
            probs = torch.softmax(logits, dim=-1)
            categorical = torch.distributions.categorical.Categorical(probs=probs)
            y_tmp = categorical.sample()
            y_tmp = one_hot(y_tmp, self.y_dim)
            with torch.no_grad():
                z = self.encoder(batch, y_tmp)
            Z[:,k] = z[:,0]
            try:
                log_q_z = self.encoder.q_z_phi.log_prob(z[:,0]) + self.encoder.q_z_eta.log_prob(z[:,1:])
                log_p_z = self.encoder.p_z_phi.log_prob(z[:,0]) + self.encoder.p_z_eta.log_prob(z[:,1:])
            except:
                log_q_z = torch.sum(self.encoder.q_z.log_prob(z), dim=1)
                log_p_z = torch.sum(self.encoder.p_z.log_prob(z), dim=1)
            
            ratio = torch.exp(log_p_z - log_q_z)

            for class_id in range(self.y_dim):
                y = torch.zeros((batch.shape[0], self.y_dim))
                y[torch.arange(batch.shape[0]), class_id] = 1.
            
                with torch.no_grad():
                    rec = self.decoder(z, y)

                p_x_z_y = torch.exp(-(torch.mean(F.mse_loss(batch, rec, reduction='none'), dim=-1) + config['lambda_sam']*sam_(batch, rec, reduction='none')))
                L[:,class_id] += ratio*p_x_z_y

        Lr, pred = torch.max(L, dim=-1)
        z_std = torch.std(Z, dim=-1)
        return Lr, pred, z[:,0], z[:,1:], logits, z_std

    def empirical_z_std(self, batch, config, num_samples=10):
        self.train()
        batch = batch.to(config['device'])
        Z = torch.zeros((batch.shape[0], num_samples))

        for k in range(num_samples):
            logits = self.classify(batch)
            probs = torch.softmax(logits, dim=-1)
            categorical = torch.distributions.categorical.Categorical(probs=probs)
            y = categorical.sample()
            y = one_hot(y, self.y_dim)
            with torch.no_grad():
                Z[:,k] = self.encoder(batch, y)[:,0]
        return torch.std(Z, dim=-1)


    def map(self, img, config):
        self.train()
        pred, z_phi, z_eta, rec, rho, i, H = [], [], [], [], [], [], []
        for (x, _) in img:
            x = x.to(config['device'])
            with torch.no_grad():
                logits = self.classify(x, reduction=False)
                logits = torch.softmax(logits, dim=1)
                _, entropy, _ = self.classifier.BALD(logits)
                logits = torch.mean(logits, dim=-1)
                y_pred = torch.argmax(logits, dim=-1)
                y = one_hot(y_pred, self.y_dim)
                z = self.encoder(x, y)
                reconstruction = self.decoder(z, y)

            logits = F.softmax(logits, dim=-1)
            pred.append(y_pred)
            z_phi.append(z[:,0])
            z_eta.append(z[:,1:])
            rec.append(reconstruction)
            rho.append(self.decoder.rho)
            i.append(x)
            H.append(entropy)

        rec = torch.cat(rec)
        i = torch.cat(i)
        rho = torch.cat(rho)
        pred = torch.cat(pred)
        z_phi = torch.cat(z_phi)
        z_eta = torch.cat(z_eta)
        H = torch.cat(H)
        Lr = Lr_(rec, i, alpha=config['lambda_sam'])
        return rec, i, rho, pred, z_phi, z_eta, Lr.item(), H


    def classify(self, x, reduction=True):
        return self.classifier(x, reduction=reduction)

    def forward(self, x, y):
        z = self.encoder(x, y)
        x = self.decoder(z, y)
        return x


class Classifier(Model):
    """
    Generic model for a classifier architecture
    """
    def __init__(self, encoder: nn.Module, classifier: nn.Module, config: dict):
        super(Model, self).__init__()
        super().__init__(config)
        self.encoder = encoder
        self.classifier = classifier
        self.y_dim = self.classifier.y_dim
        self.seed = config['seed']
        self.loss_ = 'Lc'
        self.lambda_classifier = config['lambda_classifier']

        self.logs = {
            'train': {
                'Lc': [],
                'Accuracy': [],
            },

            'val': {
                'Lc': [],
                'Accuracy': [],
                'F1-score': []
            }
        }

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                if self.seed:
                    torch.manual_seed(self.seed)
                    torch.cuda.manual_seed(self.seed)
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def loss(self, x, y, u, 
             lambda_entropy=0, 
             lambda_encoder=0, 
             lambda_classifier=0,
             lambda_sam=0, 
             beta=0, 
             target_entropy=0, 
             debug=False):
        """
        """
        y = y -1
        logits = self(x)
        pred = torch.argmax(logits, dim=-1)
        # Classification loss is a regular cross entropy
        L = F.cross_entropy(logits, y)
        # Accuracy
        accuracy = accuracy_score(pred, y)

        self.logs['train']['Lc'].append(L.item())
        self.logs['train']['Accuracy'].append(accuracy)

        return L + lambda_classifier*self.classifier.parent_module.regularization(), None


    def validation(self, x, y, lambda_sam=None):
        y = y-1
        with torch.no_grad():
            logits = self(x)
        y_pred = torch.argmax(logits, dim=-1)
        L = F.cross_entropy(logits, y)
        self.logs['val']['Lc'].append(L.item())
        return y, y_pred

    def map(self, img, config):
        self.eval()
        pred = []
        for (x, i) in img:
            x = x.to(config['device'])
            with torch.no_grad():
                logits = self(x)
                y_pred = torch.argmax(logits, dim=-1)

            pred.append(y_pred)
        pred = torch.cat(pred)
        return pred

    def regularization(self, norm="L2"):
        return self.classifier.regularization(norm)

    def forward(self, x):
        if self.encoder is None:
            x = x
        else:
            x = self.encoder(x)
        y = self.classifier(x)
        return y

class BayesianClassifier(Classifier):
    """
    Generic model for a bayesian classifier
    """
    def __init__(self, encoder, classifier, config):
        super().__init__(encoder, classifier, config)

    def BALD(self, probs):
        expected_entropy = - torch.mean(torch.sum(probs * torch.log(probs + 1e-5), 1),
                                        dim=-1)  # [batch size, ...]
        expected_p = torch.mean(probs, dim=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-5),
                                         dim=1)  # [batch size, ...]
        bald_acq = entropy_expected_p - expected_entropy
        return expected_entropy, entropy_expected_p, bald_acq

    def predict(self, x):
        self.train()
        if self.encoder is None:
            x_ = x
        else:
            with torch.no_grad():
                x_ = self.encoder(x)
        with torch.no_grad():
            pred = self.classifier(x_)
        return pred

    def predict_batch(self, x, config):
        return None, self.predict(x)

    def forward(self, x, num_samples=10, reduction=True):
        self.train()
        y = torch.zeros((x.shape[0], self.y_dim, num_samples))
        for _ in range(num_samples):
            if self.encoder is None:
                x_ = x
            else:
                x_ = self.encoder(x)
            y[:,:,_] = self.classifier(x_)
        if reduction:
            return torch.mean(y, dim=-1)
        else:
            return y 
    

def load_model(dataset, config):
    clf_cnn = {}
    for i, n_channels in enumerate(dataset.n_bands_):
        params = n_channels, 1, 1, n_channels//5, n_channels//5, 2
        clf_cnn[f'conv-{i}'] = SpectralConvolution(params)

    classifier_encoder = HyperspectralWrapper(clf_cnn)
    classifier = DenseNeuralNetwork([classifier_encoder.out_channels, config['h_dim'], config['n_classes']], dropout=0.5)
    classifier = MCConsistentDropoutModule(classifier)
    classifier = BayesianClassifier(classifier_encoder, classifier, config)

    if config['model'] in ['p3VAE', 'p3VAE_no_gs']:
        encoder = PhysicsGuidedEncoder([config['n_channels'], config['n_classes'], config['z_eta_dim'], config['h_dim']], dataset.theta)
        decoder = HybridDecoder([config['n_channels'], config['n_classes'], config['z_eta_dim'], config['h_dim'], dataset.n_bands_],\
                                [dataset.E_dir, dataset.E_dif, dataset.theta])
        model = SemiSupervisedVAE(encoder, decoder, classifier, config)

    elif config['model'] in ['guided']:
        encoder = PhysicsGuidedEncoder([config['n_channels'], config['n_classes'], config['z_eta_dim'], config['h_dim']], 1e-2)
        decoder = PhysicsGuidedDecoder([config['n_channels'], config['n_classes'], config['z_eta_dim'], config['h_dim'], dataset.n_bands_])
        model = SemiSupervisedVAE(encoder, decoder, classifier, config)

    elif config['model'] == 'gaussian':
        encoder = GaussianEncoder([config['n_channels'], config['n_classes'], config['z_eta_dim'], config['h_dim']])
        decoder = GaussianDecoder([config['n_channels'], config['n_classes'], config['z_eta_dim'], config['h_dim']])
        model = SemiSupervisedVAE(encoder, decoder, classifier, config)

    elif config['model'] in ['CNN', 'CNN_full_annotations']:
        model = classifier 

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': config['lr'], 'betas': (0.9, 0.999)}])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config['lr'], max_lr=1e-3, step_size_up=660, mode="triangular2", cycle_momentum=False)

    return model, optimizer, scheduler
