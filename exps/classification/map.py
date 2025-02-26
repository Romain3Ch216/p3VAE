# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to produce land cover maps

import torch
from data import RealDataSet
from models.model_loader import load_model
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import matplotlib.patches as patches


def legend(dataset):
    class_patches = []
    for class_id in dataset.classes:
        if class_id != 0:
            class_patches.append(patches.Patch(color=dataset.palette[class_id], label=dataset.classes[class_id]['label']))
    legend = plt.legend(handles=class_patches, ncol=len(dataset.classes)-1)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('./data/real_data/legende.pdf', bbox_inches=bbox)


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.
    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)
    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.float)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


path_images = ['/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_3d_2021-06-15_11-10-12_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_1c_2021-06-15_10-41-20_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_3a_2021-06-15_11-10-12_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_5c_2021-06-15_11-40-13_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_9c_2021-06-15_12-56-29_reflectance_rect.bsq']

colors = ['#6320EE', '#E94974', '#F2A65A', '#909CC2', '#89A7A7', ]

for k in range(1, len(sys.argv)-2):
    print('Model ', k)
    print(sys.argv[k])

    results_path = sys.argv[k]

    with open(results_path + '/config.json') as f:
        config = json.load(f)

    config['device'] = 'cpu'
    dataset = RealDataSet()
    target_names = [dataset.classes[i]['label'] for i in range(len(dataset.classes))][1:]
    classes = [class_id for class_id in range(len(dataset.classes))][1:]
    E_dir = torch.from_numpy(dataset.E_dir)
    E_dif = torch.from_numpy(dataset.E_dif)
    theta = torch.tensor([dataset.theta])

    # legend(dataset)
    # pdb.set_trace()
    n_bands = n_bands_(dataset.bbl)
    filters = {}
    for i in range(len(n_bands)):
        filters[f'conv-{i}'] = GaussianConvolution(sigma=1.5, n_channels=n_bands[i])
    preprocessing_filter = HyperspectralWrapper(filters)

    labels = ['#{} - {}'.format(class_id, dataset.classes[class_id]['label']) for class_id in classes]
    ids = ['#{}'.format(class_id) for class_id in classes]

    model = load_model(dataset, config)
    checkpoint = torch.load(results_path + '/best_model.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    if sys.argv[-1] == 'Vegetation':
        img = np.load('./data/real_data/veg_test_img.npy')
        path = results_path + '/veg_test_pred.pdf'
    elif sys.argv[-1] == 'Asphalt':
        img = np.load('./data/real_data/asph_test_img.npy')
        path = results_path + '/asph_test_pred.pdf'
    elif sys.argv[-1] == 'Fiber':
        img = np.load('./data/real_data/fib_test_img.npy')
        path = results_path + '/fib_test_pred.pdf'

    if config['model'] == 'FG-Unet':
        H, W = img.shape[:-1]
        img = img.reshape(-1, img.shape[-1])
        img = preprocessing_filter(img)
        img = torch.from_numpy(img / 10 ** 4).float()
        img = img.reshape(H, W, -1)
        pred = model.inference_on_image(img, config)
        pred = pred.reshape(H, W) + 1
        pred = convert_to_color_(pred, dataset.palette)

        fig = plt.figure()
        plt.imshow(pred)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.05)

    elif config['model'] == 'CNN':
        H, W = img.shape[:-1]
        img = img.reshape(-1, img.shape[-1])
        img = preprocessing_filter(img)
        img = torch.from_numpy(img / 10 ** 4).float()
        with torch.no_grad():
            logits = model(img)
        pred = torch.argmax(logits, dim=1).numpy()+1
        pred = pred.reshape(H, W)
        pred = convert_to_color_(pred, dataset.palette)

        fig = plt.figure()
        plt.imshow(pred)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.05)


    elif config['model'] in ['gaussian', 'p3VAE', 'guided']:
        H, W = img.shape[:-1]
        img = img.reshape(-1, img.shape[-1])
        img = preprocessing_filter(img)
        img = torch.from_numpy(img / 10 ** 4).float()
        data = torch.utils.data.TensorDataset(img, torch.ones(img.shape[0]))
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

        if sys.argv[-2] == 'p':
            pred, _, _, _, _, _ = model.inference(loader, config, mode='argmax_p_y_x')
        elif sys.argv[-2] == 'q':
            pred, _ = model.inference(loader, config, mode='q_y_x')

        pred = pred.numpy().reshape(H, W)+1
        pred = convert_to_color_(pred, dataset.palette)

        fig = plt.figure()
        plt.imshow(pred)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.05)

    elif config['model'] in ['ssInfoGAN']:
        H, W = img.shape[:-1]
        img = img.reshape(-1, img.shape[-1])
        img = preprocessing_filter(img)
        img = torch.from_numpy(img / 10 ** 4).float()
        data = torch.utils.data.TensorDataset(img, torch.ones(img.shape[0]))
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

        pred, _ = model.inference(loader)
        pred = pred.numpy().reshape(H, W)+1
        pred = convert_to_color_(pred, dataset.palette)

        fig = plt.figure()
        plt.imshow(pred)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.05)


