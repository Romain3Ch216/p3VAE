import torch
import torch.nn as nn
from models.SpectralModel.spectral_model import SpectralModel
from models.generic_nets import convolutional_latent_net, dense_classes_net
from models.VAE.encoders import PhysicsGuidedEncoder, GaussianEncoder
from models.VAE.decoders import HybridDecoder, GaussianDecoder, PhysicsGuidedDecoder
from models.VAE.ss_vae import SemiSupervisedVAE
from models.GAN.nets import Discriminator, Generator, QHeadUS
from models.GAN.ss_info_gan import ssInfoGAN
from models.DeepSegModel.fg_unet import FgUnet
from models.DeepSegModel.segmentation_model import SegmentationModel

def load_model(dataset, config):
    config.setdefault('dropout', 0.5)

    dims = [config['n_channels'], config['n_classes'], config['z_eta_dim'], config['h_dim']]
    conv_params = dict(
        (conv_id, (n_channels, 1, 1, n_channels//5, n_channels//5, 2))
            for (conv_id, n_channels) in enumerate(dataset.n_bands_)
    )

    if config['model'] in ['CNN', 'CNN_full_annotations']:
        latent_cnn = convolutional_latent_net(conv_params)
        classifier = dense_classes_net(latent_cnn.out_channels, config)

        optimizer = torch.optim.Adam(
            [{'params': latent_cnn.parameters()}, {'params': classifier.parameters()}],
            lr=config['lr'],
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config['lr'],
            max_lr=1e-3,
            step_size_up=660,
            mode="triangular2",
            cycle_momentum=False)

        model = SpectralModel(latent_cnn, classifier, optimizer, scheduler, config)

    elif config['model'] == 'ssInfoGAN':
        # latent_cnn = convolutional_latent_net(conv_params)
        netD = Discriminator(config['n_channels'], config['h_dim'])
        netQss = dense_classes_net(config['h_dim'], config)
        netG = Generator(config['noise_dim'], config['n_classes'], config['z_eta_dim']+1, config['h_dim'], config['n_channels']) #, dataset.n_bands_)
        netQus = QHeadUS(config['h_dim'], config['z_eta_dim']+1)

        optimD = torch.optim.RMSprop(
            [{'params': netD.parameters(), 'lr': config['lr']},
             {'params': netQss.parameters(), 'lr': config['lr'], 'weight_decay': 1E-2}]
        )
        optimG = torch.optim.RMSprop(
            [{'params': netG.parameters()},
             {'params': netQus.parameters()},
             {'params': netQss.parameters(), 'weight_decay': 1E-2}],
            lr=config['lr']
        )

        model = ssInfoGAN(netD, netG, netQss, netQus, optimD, optimG, config)

    elif config['model'] == 'FG-Unet':
        config.setdefault('patch_size', 32)
        fg_unet = FgUnet(config['n_channels'], config['n_classes'], config['patch_size'])
        model = SegmentationModel(fg_unet,
                                  {'supervised': nn.CrossEntropyLoss(), 'unsupervised': nn.L1Loss()},
                                  torch.optim.Adam(fg_unet.parameters(), lr=config['lr'], weight_decay=1E-2),
                                  config)


    elif config['model'] in ['p3VAE', 'p3VAE_no_gs', 'p3VAE_g', 'gaussian', 'guided', 'guided_no_gs']:
        if config['model'] in ['p3VAE', 'p3VAE_no_gs', 'p3VAE_g']:
            encoder = PhysicsGuidedEncoder(dims, dataset.theta)
            decoder = HybridDecoder(dims + [dataset.n_bands_],\
                                    [dataset.E_dir, dataset.E_dif, dataset.theta], config['beta_g'])

        elif config['model'] == 'gaussian':
            encoder = GaussianEncoder(dims)
            decoder = GaussianDecoder(dims)

        elif config['model'] in ['guided', 'guided_no_gs']:
            encoder = PhysicsGuidedEncoder(dims, 1e-2)
            decoder = PhysicsGuidedDecoder(dims +[dataset.n_bands_])

        latent_cnn = convolutional_latent_net(conv_params)
        classes_dense = dense_classes_net(latent_cnn.out_channels, config)
        classifier = SpectralModel(latent_cnn, classes_dense, None, None, config)

        optimizer = torch.optim.Adam(
            [{'params': encoder.parameters()}, {'params': decoder.parameters()}, {'params': classifier.parameters()}],
            lr=config['lr'],
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config['lr'],
            max_lr=1e-3,
            step_size_up=660,
            mode="triangular2",
            cycle_momentum=False)

        model = SemiSupervisedVAE(encoder, decoder, classifier, optimizer, scheduler, config)

    for module in model.modules():
        module.to(config['device'])

    return model
