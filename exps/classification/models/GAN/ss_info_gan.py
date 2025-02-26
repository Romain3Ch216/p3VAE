import pdb

from models.models import Model
from models.utils import NormalNLLLoss, cycle_loader, data_loader_from_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import errno


class ssInfoGAN(Model):
    def __init__(self, discriminator, generator, netQss, netQus, optimD, optimG, config):
        super(Model, self).__init__()

        self.netD = discriminator
        self.netG = generator
        self.netQss = netQss
        self.netQus = netQus

        self.optimD = optimD
        self.optimG = optimG

        self.config = config
        self.device = config['device']
        config['loss_'] = 'C_loss'
        config['direction'] = 1
        super().__init__(config)

        self.logs = {
            'train': {
                'D_loss': [],
                'G_loss': [],
                'C_loss': [],
                'con_loss': [],
                'train_c_loss': []
            },
            'val': {
                'val_C_loss': [],
                'Accuracy': [],
                'F1-score': [],
                'C_loss': [],
                'con_loss': []
            }
        }

        for m in self.modules():
            m.apply(self.weight_init)

    def update_logs(self, logs, epoch):
        for metric in self.logs['train']:
            logs['train'][metric].append(sum(self.logs['train'][metric])/len(self.logs['train'][metric]))
            self.logs['train'][metric] = []

        if 'val' in self.logs:
            for metric in self.logs['val']:
                if metric in ['C_loss', 'con_loss']:
                    logs['val'][metric].append(logs['train'][metric][-1])
                else:
                    logs['val'][metric].append(sum(self.logs['val'][metric])/len(self.logs['val'][metric]))
                self.logs['val'][metric] = []

            if logs['val'][self.loss_][-1] < self.best_loss:
                self.best_epoch = epoch
                self.best_loss = logs['val'][self.loss_][-1]
                torch.save({'epoch': epoch, 'best_loss': self.best_loss,\
                            'state_dict': self.state_dict()}, self.path + 'best_model.pth.tar')
        return logs

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
            torch.manual_seed(self.config['seed'])
            torch.cuda.manual_seed(self.config['seed'])
            torch.nn.init.kaiming_normal_(m.weight.data)

    def calc_gradient_penalty(self, netD, real_data, generated_data, penalty_weight=10):
        batch_size = real_data.size()[0]

        alpha = torch.rand(batch_size, 1) if real_data.dim() == 2 else torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)

        interpolated = alpha * real_data + (1 - alpha) * generated_data
        #interpolated = Variable(interpolated, requires_grad=True)
        interpolated.requires_grad_()
        interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        _, prob_interpolated = netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return penalty_weight * ((gradients_norm - 1) ** 2).mean()

    def train_discriminator_and_Q_dis_on_batch(self, x, y, u, c_weight=0.1):
        try:
            x, y, u = x.float(), y.long(), u.float()
            x, y, u = x.to(self.device), y.to(self.device), u.to(self.device)
            real_samples = torch.cat((x, u), dim=0)
        except:
            x, y = x.float(), y.long()
            x, y = x.to(self.device), y.to(self.device)
            real_samples = x

        z, _ = self.sample_noise(real_samples.size(0))
        self.netD.zero_grad()
        self.netQss.zero_grad()

        # Get discriminator prediction on real samples
        real_features, real_logits = self.netD(real_samples)
        D_real = real_logits.mean()
        # D_real = self.netD(real_samples).mean()
        # Get discriminator prediction on fake samples
        fake_samples = self.netG(z)
        _, fake_logits = self.netD(fake_samples)
        D_fake = fake_logits.mean()
        # Compute gradient penalty
        gradient_penalty = self.calc_gradient_penalty(self.netD, real_samples.data, fake_samples.data)
        # Compute loss and backpropagate
        D_loss = D_fake - D_real + gradient_penalty
        D_loss.backward(retain_graph=True)
        # _, classes = torch.max(y, dim=1)
        classes = y - 1
        # Semi-supervised part
        logits = self.netQss(real_features[:y.size(0)])
        C_loss = c_weight * F.cross_entropy(logits, classes) + self.config['lambda_classifier'] * self.netQss.regularization()
        C_loss.backward()
        self.optimD.step()
        return D_loss.item(), C_loss.item()/c_weight

    def train_generator_on_batch(self, x, y, u, c_weight=1, l1=0.25):
        self.netG.zero_grad()
        self.netQus.zero_grad()

        # Pas besoin de y

        try:
            x, y, u = x.float(), y.long(), u.float()
            x, y, u = x.to(self.device), y.to(self.device), u.to(self.device)
            real_samples = torch.cat((x, u), dim=0)
        except:
            x, y = x.float(), y.long()
            x, y = x.to(self.device), y.to(self.device)
            real_samples = x

        # Sample random noise
        z, classes = self.sample_noise(real_samples.size(0))
        classes = torch.LongTensor(classes).to(self.device)
        # Generate fake samples
        fake_samples = self.netG(z)
        fake_features, fake_logits = self.netD(fake_samples)
        pred = self.netQss(fake_features)
        C_loss = F.cross_entropy(pred, classes)
        G_loss = -torch.mean(fake_logits)
        # InfoGAN loss
        q_mu, q_var = self.netQus(fake_features)
        # con_loss = l1*NormalNLLLoss(z[:, self.config['noise_dim'] + self.config['n_classes']:], q_mu, q_var)
        con_loss = l1 * F.mse_loss(z[:, self.config['noise_dim'] + self.config['n_classes']:], q_mu)
        # Include the InfoGAN loss for continuous code
        loss = G_loss + l1 * con_loss + c_weight * C_loss
        # Backpropagate
        loss.backward()
        self.optimG.step()
        return G_loss.item(), con_loss.item() / l1, C_loss.item()

    def validation(self, data_loader):
        y_true, y_pred, C_loss = [], [], []
        for x, y in data_loader:
            x, y = x.float(), y.long()
            y = y - 1
            x, y = x.to(self.device), y.to(self.device)
            # Get predictions from C
            with torch.no_grad():
                real_features, _ = self.netD(x)
                logits = self.netQss(real_features)
            y_pred_ = torch.argmax(logits, dim=-1)
            C_loss.append(F.cross_entropy(logits, y).item())
            y_pred.extend(y_pred_.cpu().numpy())
            y_true.extend(y.cpu().numpy())
        f1score = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        C_loss = sum(C_loss)/len(C_loss)
        return C_loss, acc, f1score

    def optimize(self, labeled_data, unlabeled_data, val_dataset, config, logger, verbose=1000):
        logs = self.init_logs()
        n_it = len(unlabeled_data)*config['epochs']
        d_step = config['d_step']
        for it in tqdm(range(n_it)):
            ###########################
            # (1) Update C and D      #
            ###########################
            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in self.netQss.parameters():
                p.requires_grad = True
            for p in self.netG.parameters():
                p.requires_grad = False

            # D is trained d_step times for each iteration
            data_loader = cycle_loader(labeled_data, unlabeled_data)
            for _, data in zip(range(d_step), data_loader):
                try:
                    ((x, y), (u, _)) = data
                except:
                    x, y = data
                    u = None

                D_loss, train_c_loss = self.train_discriminator_and_Q_dis_on_batch(x, y, u, c_weight=config['c_weight'])

            for p in self.netD.parameters():
                p.requires_grad = False
            for p in self.netQss.parameters():
                p.requires_grad = False
            for p in self.netG.parameters():
                p.requires_grad = True

            G_loss, con_loss, C_loss = self.train_generator_on_batch(x, y, u, c_weight=config['c_weight'], l1=config['l1'])

            self.logs['train']['D_loss'].append(D_loss)
            self.logs['train']['G_loss'].append(G_loss)
            self.logs['train']['train_c_loss'].append(train_c_loss)
            self.logs['train']['C_loss'].append(C_loss)
            self.logs['train']['con_loss'].append(con_loss)

            if it % verbose == 0:
                # self.generate_samples(y.size(0), it)
                C_loss, acc, f1score = self.validation(val_dataset)
                self.logs['val']['val_C_loss'].append(C_loss)
                self.logs['val']['Accuracy'].append(acc)
                self.logs['val']['F1-score'].append(f1score)

                self.feed_logger(logs, it, logger)

        return logs

    def inference(self, data_loader):
        PRED = []
        Z = []
        for data in data_loader:
            try:
                x, _ = data
            except:
                x = data[0]
            x = x.to(self.device)
            pred, z = self.inference_on_batch(x)
            PRED.append(pred)
            Z.append(z)
        pred = torch.cat(PRED)
        z = torch.cat(Z)
        return pred, z

    def inference_on_batch(self, x):
        with torch.no_grad():
            real_features, _ = self.netD(x)
            logits = self.netQss(real_features)
            mu, var = self.netQus(real_features)
        z = mu + torch.randn(var.shape)*var**0.5
        pred = torch.argmax(logits, dim=-1)
        return pred, z

    def inference_on_image(self, image, config):
        data_loader = data_loader_from_image(image, config['batch_size'])
        pred, z = self.inference(data_loader)
        pred = pred.view(image.shape[0], image.shape[1])
        z = z.view(image.shape[0], image.shape[1], -1)
        return pred, z

    def generate_samples(self, b_size, it):
        try:
            os.makedirs(self.path + '/samples', exist_ok=True)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        with torch.no_grad():
            idx = np.random.randint(0, self.netQss.y_dim)
            z, _ = self.sample_noise(b_size, class_id=idx)
            samples = self.netG(z).data.cpu().numpy()[:16]
            mean_spectrum = np.mean(samples, axis=0)
            std_spectrum = np.std(samples, axis=0)
            fig = plt.figure()
            plt.plot(mean_spectrum - std_spectrum, linestyle='dotted', label='-std')
            plt.plot(mean_spectrum, label='mean')
            plt.plot(mean_spectrum + std_spectrum, linestyle='dotted', label='+std')
            plt.title("Samples for class {} at iteration {}".format(idx, it))
            plt.savefig(self.path + '/samples/sample_{}_it_{}.png'.format(idx, it))

    def sample_noise(self, batch_size, class_id=None, con_c=None, z=None):
        """
        Sample random noise vector for training.

        INPUT
        --------
        n_dis_c : Number of discrete latent code.
        dis_c_dim : Dimension of discrete latent code.
        n_con_c : Number of continuous latent code.
        n_z : Dimension of incompressible noise.
        batch_size : Batch Size
        device : GPU/CPU
        """

        dis_c_dim = self.config['n_classes']
        n_con_c = self.config['z_eta_dim']+1
        n_z = self.config['noise_dim']
        device = self.config['device']

        if z is None:
            z = torch.randn(batch_size, n_z, device=device)

        dis_c = torch.zeros(batch_size, dis_c_dim, device=device)
        if class_id is None:
            idx = np.random.randint(dis_c_dim, size=batch_size)
        else:
            idx = class_id*np.ones(batch_size)
        dis_c[torch.arange(0, batch_size), idx] = 1.0

        if con_c is None:
            # Random uniform between -1 and 1.
            con_c = torch.rand(batch_size, n_con_c, device=device) * 2 - 1

        noise = torch.cat((z, dis_c, con_c), dim=1)

        return noise, idx

    def classify(self, x):
        real_features, _ = self.netD(x)
        logits = self.netQss(real_features)
        return logits
