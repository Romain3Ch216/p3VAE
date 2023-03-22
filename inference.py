# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Script to compute quantitative metrics on real data
import torch
from data import spectra_bbm, RealDataSet
from models.model_loader import load_model
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
import json
import sys
from utils import *


global_report = []

for k in range(1, len(sys.argv)):
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

    report = dict((class_id, {}) for class_id in classes)
    report['cm'] = {}
    report['avg_f1_score_q'] = 0
    report['avg_f1_score_p'] = 0

    reconstruction = {}
    pred = {}
    z_phi = {}
    z_eta = {}
    H = {}
    Lr = {}

    if config['model'] == 'FG-Unet':
        test_dataloader = dataset.patch_loader('test', batch_size=config['batch_size'])
    else:
        test_dataloader = dataset.load(dataset.test_data, None, batch_size=config['batch_size'], test=True)

    if config['model'] in ['p3VAE', 'p3VAE_no_gs', 'gaussian', 'guided', 'guided_no_sg']:
        pred['q'], _ = model.inference(test_dataloader, config, mode='q_y_x')
        pred['p'], _, _, _, _, _ = model.inference(test_dataloader, config, mode='argmax_p_y_x')
        pred['p'] = pred['p'].numpy()

        cm_p = confusion_matrix(dataset.test_labels - 1, pred['p'], normalize='true')
        report['cm']['p'] = cm_p.tolist()
        f1_score_p = f1_score(dataset.test_labels-1, pred['p'], average=None)

    elif config['model'] in ['CNN', 'CNN_full_annotations']:
        pred['q'] = model.inference(test_dataloader)

    elif config['model'] in ['ssInfoGAN']:
        pred['q'], _ = model.inference(test_dataloader)

    elif config['model'] == 'FG-Unet':
        dataset.test_labels, pred['q'] = model.inference(test_dataloader)


    pred['q'] = pred['q'].numpy()
    f1_score_q = f1_score(dataset.test_labels-1, pred['q'], average=None)
    cm_q = confusion_matrix(dataset.test_labels - 1, pred['q'], normalize='true')
    report['cm']['q'] = cm_q.tolist()

    for class_id in classes:
        report[class_id]['f1_score_q'] = f1_score_q[class_id - 1]
        report['avg_f1_score_q'] += f1_score_q[class_id - 1] / len(f1_score_q)
        if config['model'] in ['p3VAE', 'p3VAE_no_gs', 'gaussian', 'guided', 'guided_no_sg']:
            report[class_id]['f1_score_p'] = f1_score_p[class_id - 1]
            report['avg_f1_score_p'] += f1_score_p[class_id - 1] / len(f1_score_p)
        else:
            report[class_id]['f1_score_p'] = 0
            report['avg_f1_score_p'] = 0

    global_report.append(report)

avg_report = {}
for class_id in classes:
    avg_report[class_id] = {}
    for metric in report[class_id]:
        avg_report[class_id][metric] = 0


avg_report['avg_f1_score_q'] = 0
avg_report['avg_f1_score_p'] = 0
avg_report['f1_score_p'] = []
avg_report['f1_score_q'] = []
avg_report['cm_q'] = []
avg_report['cm_p'] = []

for report in global_report:
    for key in report:
        if key in ['avg_f1_score_q', 'avg_f1_score_p']:
            avg_report[key] += report[key] / len(global_report)
        elif key == 'cm':
            pass
        else:
            for metric in report[key]:
                avg_report[key][metric] += report[key][metric] / len(global_report)
    avg_report['f1_score_p'].append(report['avg_f1_score_p']);
    avg_report['f1_score_q'].append(report['avg_f1_score_q'])
    avg_report['cm_q'].append(report['cm']['q'])
    if config['model'] in ['p3VAE', 'gaussian', 'guided']:
        avg_report['cm_p'].append(report['cm']['p'])

with open('./results/real_data/{}/classification_report.json'.format(config['model']), 'w') as f:
    json.dump(avg_report, f, indent=4)
