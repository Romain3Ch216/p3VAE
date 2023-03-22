from utils import *
import os
from os import listdir
from os.path import isfile, join
import pickle as pkl
import numpy as np
import pdb

sets = ['labeled_train_ground_truth', 'unlabeled_train_ground_truth', 'test_ground_truth']
path_images = ['/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_3d_2021-06-15_11-10-12_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_1c_2021-06-15_10-41-20_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_3a_2021-06-15_11-10-12_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_5c_2021-06-15_11-40-13_reflectance_rect.bsq',
               '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_9c_2021-06-15_12-56-29_reflectance_rect.bsq']
path_shapefile = './data/real_data/ground_truth/'

bands, bbl = get_bands(path_images[0])
wv = wv(path_images[0])
classes = [1, 2, 3, 4, 5, 6, 7, 8]

np.save('./data/real_data/bbl.npy', bbl)
np.save('./data/real_data/wv.npy', wv)

for data_set in sets:
    path = join(path_shapefile, data_set, data_set) + '.shp'
    output_path = join(path_shapefile, data_set, data_set)
    rasterize_shapefile(path,
                        path_images,
                        attribute='Material',
                        output_path=output_path)

    folder = join(path_shapefile, data_set)
    path_gts = [join(folder, f) for f in listdir(folder)
                if join(folder, f)[-3:] == 'bsq']

    spectra = read_spectra_from_rasters(path_images, path_gts, bands, bbl, classes)

    with open(os.path.join(folder, data_set) + '.pkl', 'wb') as f:
        pkl.dump(spectra, f)


