# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Data sets definition
import pdb

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import concatenate_dict
import pickle as pkl
import geopandas as gpd
from rasterio.features import rasterize
import rasterio
from geopandas import GeoDataFrame
from typing import Dict, Tuple


class DataSet:
    """
    A generic class for data sets

    Attributes:
        - classes: a Dict with class ids as keys and (label, spectrum, rgb color) as values
        - wv: a 1D array (float) of the central wavelength of the bands
        - bbl: a 1D array (bool) of the bad bands (i.e. bands that are removed)
    """

    def __init__(self, classes: Dict, wv: np.ndarray, bbl: np.ndarray):
        self.classes = classes
        self.wv = wv
        self.bbl = bbl

    @property
    def palette(self):
        color = {}
        for class_id in self.classes:
            color[class_id] = tuple([color_ / 256 for color_ in self.classes[class_id]['rgb']])
        return color

    @property
    def labels(self):
        labels_ = []
        for class_id in self.classes:
            labels_.append(self.classes[class_id]['label'])
        return labels_

    @property
    def bands(self):
        bbl = np.array(list(map(int, self.bbl)), dtype=int)
        bands = tuple(np.where(bbl != 0)[0] + 1)
        bands = [int(b) for b in bands]
        return bands

    @property
    def n_bands_(self):
        n_bands = []
        good_bands = np.where(self.bbl == True)[0]
        s = 1
        for i in range(len(good_bands) - 1):
            if good_bands[i] == good_bands[i + 1] - 1:
                s += 1
            else:
                n_bands.append(s)
                s = 1

        n_bands.append(s)
        return n_bands

    def img_to_rgb(self, img):
        b = np.mean(img[:, :, self.rgb_bands[0]:self.rgb_bands[0] + 1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
        v = np.mean(img[:, :, self.rgb_bands[1]:self.rgb_bands[1] + 1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
        r = np.mean(img[:, :, self.rgb_bands[2]:self.rgb_bands[2] + 1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
        rgb = 3 * np.concatenate((r, v, b), axis=-1)
        return rgb

    def gt_to_rgb(self, gt):
        plot = np.zeros((gt.shape[0], gt.shape[1], 3))
        for class_id in np.unique(gt):
            plot[gt == class_id] = self.classes[class_id]['rgb']
        return plot / 255

    def plot_spectra(self):
        fig = plt.figure(figsize=(15, 10))
        style = ['solid', 'dotted', 'dashed']

        for class_id in self.classes:
            if class_id != 0:
                label = self.classes[class_id]['label']
                color = [self.classes[class_id]['rgb'][i] / 256 for i in range(3)]
                for i, sp in enumerate(self.classes[class_id]['spectrum']):
                    plt.plot(self.wv, spectra_bbm(sp.reshape(1, -1), self.bbl).reshape(-1),
                             label=label + '_' + str(i + 1), color=color, linestyle=style[i], lw=2)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel(r'Wavelenght ($\mu m$)', fontsize=15)
        plt.ylabel('Reflectance', fontsize=15)

        for class_id in self.classes:
            if class_id != 0:
                fig_ = plt.figure(figsize=(15, 10))
                style = ['solid', 'dotted', 'dashed']
                label = self.classes[class_id]['label']
                color = [self.classes[class_id]['rgb'][i] / 256 for i in range(3)]
                for i, sp in enumerate(self.classes[class_id]['spectrum']):
                    plt.plot(self.wv, spectra_bbm(sp.reshape(1, -1), self.bbl).reshape(-1),
                             label=label + '_' + str(i + 1), color=color, linestyle=style[i], lw=2)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.xlabel(r'Wavelenght ($\mu m$)', fontsize=15)
                plt.ylabel('Reflectance', fontsize=15)
                plt.ylim(0, 1)
                plt.savefig('./Figures/reflectance_spectra.pdf'.format(class_id), dpi=200, bbox_inches='tight',
                            pad_inches=0.05)
        return fig


class SimulatedDataSet(DataSet):
    def __init__(self):
        train_img = np.load('./data/simulation/train_img.npy')
        test_img = np.load('./data/simulation/test_img.npy')
        test_img_coupling = np.load('./data/simulation/test_img_coupling.npy')
        self.train_gt = np.load('./data/simulation/train_gt.npy')
        self.val_gt = np.load('./data/simulation/val_gt.npy')
        self.test_gt = np.load('./data/simulation/test_gt.npy')
        test_spectra = np.load('./data/simulation/test_spectra.npy')
        self.test_labels = np.load('./data/simulation/test_labels.npy')
        self.wv = np.load('./data/simulation/wv.npy')
        E_dir = np.load('./data/simulation/E_dir.npy').astype(np.float32)
        E_dif = np.load('./data/simulation/E_dif.npy').astype(np.float32)
        self.theta = 30 * np.pi / 180
        bb = [[45, 49], [66, 82], [98, 109], [131, 162], [201, 242], [289, 300]]
        self.B = train_img.shape[-1]
        self.bbl = bandNumbersToBbl(self.B, bb)
        self.rgb_bands = [0, 12, 38]
        self.train_img = train_img[:, :, self.bbl]
        self.test_img = test_img[:, :, self.bbl]
        self.test_img_coupling = test_img_coupling[:, :, self.bbl]
        self.E_dir = E_dir[self.bbl] / np.cos(self.theta)
        self.E_dif = E_dif[self.bbl]
        self.test_spectra = test_spectra[:, self.bbl]
        self.n_channels = sum(self.n_bands_)

        # Reference spectra
        grass = './data/simulation/reflectance/grass.npy'
        dry_grass = './data/simulation/reflectance/dry_grass.npy'
        tree = './data/simulation/reflectance/tree.npy'
        alu = './data/simulation/reflectance/alu.npy'
        loam_1 = './data/simulation/reflectance/loam_1.npy'
        loam_2 = './data/simulation/reflectance/loam_2.npy'
        tile_1 = './data/simulation/reflectance/tile_1.npy'
        tile_2 = './data/simulation/reflectance/tile_2.npy'
        tile_3 = './data/simulation/reflectance/tile_3.npy'
        asphalt = './data/simulation/reflectance/asphalt.npy'

        self.classes = {
            0: {
                'label': 'Unknown',
                'spectrum': [],
                'rgb': (255, 255, 255)
            },
            1: {
                'label': 'Vegetation',
                'spectrum': [np.load(grass), np.load(dry_grass), np.load(tree)],
                'rgb': (142, 196, 110)
            },
            2: {
                'label': 'Alu',
                'spectrum': [np.load(alu)],
                'rgb': (159, 194, 204)
            },
            3: {
                'label': 'Loam',
                'spectrum': [np.load(loam_1), np.load(loam_2)],
                'rgb': (254, 215, 102)
            },
            4: {
                'label': 'Tile',
                'spectrum': [np.load(tile_1), np.load(tile_2), np.load(tile_3)],
                'rgb': (193, 102, 107)
            },
            5: {
                'label': 'Asphalt',
                'spectrum': [np.load(asphalt)],
                'rgb': (65, 63, 80)
            }
        }

        for class_id in self.classes:
            for i in range(len(self.classes[class_id]['spectrum'])):
                self.classes[class_id]['spectrum'][i] = self.classes[class_id]['spectrum'][i][self.bbl]

    def load(self, img, gt, batch_size, unlabeled=False, test=False, split=False, split_rate=0.8):
        config = {'patch_size': 1, 'batch_size': batch_size, 'ignored_labels': [0]}
        if unlabeled:
            config['ignored_labels'] = list(np.arange(1, len(np.unique(gt))))
        if test:
            shuffle = False
        else:
            shuffle = True

        data = HyperX(img, gt, config)
        n_data = len(data)
        n_train = int(split_rate * n_data)
        n_val = n_data - n_train
        if split:
            train_data, val_data = torch.utils.data.random_split(data, [n_train, n_val])
            train_data = torch.utils.data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
            val_data = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
            return train_data, val_data
        else:
            data = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
            return data


class RealDataSet(DataSet):
    def __init__(self):
        with open('./data/real_data/ground_truth/labeled_train_ground_truth/labeled_train_ground_truth.pkl', 'rb') as f:
            labeled_train_gt = pkl.load(f)
        with open('./data/real_data/ground_truth/unlabeled_train_ground_truth/unlabeled_train_ground_truth.pkl',
                  'rb') as f:
            unlabeled_train_gt = pkl.load(f)
        with open('./data/real_data/ground_truth/test_ground_truth/test_ground_truth.pkl', 'rb') as f:
            test_gt = pkl.load(f)

        self.path_gts = {
            'labeled': './data/real_data/ground_truth/labeled_train_ground_truth/labeled_train_ground_truth.shp',
            'unlabeled': './data/real_data/ground_truth/unlabeled_train_ground_truth/unlabeled_train_ground_truth.shp',
            'test': './data/real_data/ground_truth/test_ground_truth/test_ground_truth.shp'}

        self.path_images = [
            '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_3d_2021-06-15_11-10-12_reflectance_rect.bsq',
            '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_1c_2021-06-15_10-41-20_reflectance_rect.bsq',
            '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_3a_2021-06-15_11-10-12_reflectance_rect.bsq',
            '/home/rothor/Documents/ONERA/Datasets/Toulouse/images/TLS_9c_2021-06-15_12-56-29_reflectance_rect.bsq']
        self.folder = './data/real_data/ground_truth/labeled_train_ground_truth/'

        self.loaded_patches = {
            'labeled': ['./data/real_data/labeled_patches.npy', './data/real_data/labeled_gt_patches.npy'],
            'unlabeled': ['./data/real_data/unlabeled_patches.npy', './data/real_data/unlabeled_gt_patches.npy'],
            'test': ['./data/real_data/test_patches.npy', './data/real_data/test_gt_patches.npy']
        }

        self.bbl = np.load('./data/real_data/bbl.npy')
        self.wv = np.load('./data/real_data/wv.npy')
        self.theta = 22.12 * np.pi / 180
        E_dir = np.load('./data/real_data/E_dir.npy').astype(np.float32)
        E_dif = np.load('./data/real_data/E_dif.npy').astype(np.float32)
        self.E_dir = E_dir[self.bbl] / np.cos(self.theta)
        self.E_dif = E_dif[self.bbl]
        self.rgb_bands = [6, 35, 65]
        self.B = len(self.wv)

        self.classes = {
            0: {'label': 'Untitled', 'rgb': (0, 0, 0)},
            1: {'label': 'Tile', 'rgb': (193, 102, 107)},
            2: {'label': 'Asphalt', 'rgb': (65, 63, 80)},
            3: {'label': 'Vegetation', 'rgb': (142, 196, 110)},
            4: {'label': 'Painted sheet metal', 'rgb': (232, 241, 242)},
            5: {'label': 'Water', 'rgb': (2, 128, 144)},
            6: {'label': 'Gravels', 'rgb': (225, 218, 189)},
            7: {'label': 'Metal', 'rgb': (159, 194, 204)},
            8: {'label': 'Fiber cement', 'rgb': (236, 157, 237)}
        }

        self.labeled_train_data, self.labeled_train_labels = concatenate_dict(labeled_train_gt,
                                                                    [class_id for class_id in range(1, len(self.classes))])
        self.unlabeled_train_data, self.unlabeled_train_labels = concatenate_dict(unlabeled_train_gt, [class_id for class_id in
                                                                                             range(1, len(self.classes))])

        self.test_data, self.test_labels = concatenate_dict(test_gt, [class_id for class_id in range(1, len(self.classes))])
        self.n_channels = self.labeled_train_data.shape[-1]

        test_coords = []
        for class_id in self.classes:
            if class_id > 0:
                for image_id in range(len(test_gt[class_id]['coords'])):
                    coords = test_gt[class_id]['coords'][image_id]
                    if coords is not None:
                        out = np.concatenate((
                            image_id * np.ones((len(coords[0]), 1)), coords[0].reshape(-1, 1), coords[1].reshape(-1, 1)
                        ), axis=1)
                        test_coords.append(out)
        self.test_coords = np.concatenate(test_coords, axis=0)

    def get_patches(self, set_, patch_size=32, top_left_padding=8):
        gt = gpd.read_file(self.path_gts[set_])
        self.patches_coordinates = []
        self.patches_images = []

        for path in self.path_images:
            raster = gdal.Open(path)
            transform = raster.GetGeoTransform()
            xOrigin = transform[0]
            yOrigin = transform[3]
            pixelWidth = transform[1]
            pixelHeight = -transform[5]
            # Xgeo = xOrigin + Xpixel*pixelWidth
            # Xpixel = (Xgeo - xOrigin) / pixelWidth
            # Ygeo = yOrigin - Yline * pixelHeight
            # Yline = (yOrigin - Ygeo) / pixelHeight
            image_bounds = (xOrigin,
                            yOrigin,
                            xOrigin + raster.RasterXSize * pixelWidth,
                            yOrigin - raster.RasterYSize * pixelHeight)

            for id, polygon in gt.iterrows():
                bounds = polygon['geometry'].bounds  # min x, min y, max x, max y
                if is_polygon_in_rectangle(bounds, image_bounds):
                    print('Polygon in image')

                    left_col = int((bounds[0] - xOrigin) / pixelWidth) - top_left_padding
                    right_col = int((bounds[2] - xOrigin) / pixelWidth)
                    top_row = int((yOrigin - bounds[3]) / pixelHeight) - top_left_padding
                    bottom_row = int((yOrigin - bounds[1]) / pixelHeight)

                    width = right_col - left_col
                    height = bottom_row - top_row
                    n_x_patches = int(np.ceil(width / patch_size))
                    n_y_patches = int(np.ceil(height / patch_size))
                    for i in range(n_x_patches):
                        for j in range(n_y_patches):
                            self.patches_images.append(self.path_images.index(path))
                            self.patches_coordinates.append(
                                tuple((left_col + i * patch_size, top_row + j * patch_size, patch_size, patch_size))
                            )
                else:
                    print('Polygon is not in image', polygon['Image'])

    def load(self, data, labels, batch_size, test=False, split=False, split_rate=0.8):
        shuffle = False if test else True
        data = torch.from_numpy(data / 10 ** 4).float()
        n_data = data.shape[0]
        if labels is None:
            data = torch.utils.data.TensorDataset(data)
        else:
            labels = torch.from_numpy(labels).long()
            data = torch.utils.data.TensorDataset(data, labels)

        if split:
            n_train = int(split_rate * n_data)
            n_val = n_data - n_train
            train_data, val_data = torch.utils.data.random_split(data, [n_train, n_val])
            train_data = torch.utils.data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
            val_data = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
            return train_data, val_data
        else:
            data = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
            return data

    def patch_loader(self, set_, batch_size):
        patches, gt = np.load(self.loaded_patches[set_][0]), np.load(self.loaded_patches[set_][1])
        patches, gt = torch.from_numpy(patches), torch.from_numpy(gt)
        data = torch.utils.data.TensorDataset(patches, gt)
        loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
        return loader


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene
        Credits to https://github.com/nshaud/DeepHyperX"""

    def __init__(self, data, gt, hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.shuffle = False
        self.height = data.shape[0]
        self.width = data.shape[1]
        self.augment = False

        mask = np.ones_like(gt)
        for l in self.ignored_labels:
            mask[gt == l] = 0

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        if p > 0:
            self.indices = np.array(
                [
                    (x, y)
                    for x, y in zip(x_pos, y_pos)
                    if x >= p and x < data.shape[0] - p and y >= p and y < data.shape[1] - p
                ]
            )
        else:
            self.indices = np.array(
                [
                    (x, y)
                    for x, y in zip(x_pos, y_pos)
                ]
            )
        self.labels = [self.label[x, y] for x, y in self.indices]
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        if self.augment:
            data = data * (1 + torch.randn((data.shape[-2], data.shape[-1])) * 0.05)

        return data, label


class GeoDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, raster_gt_path, patches_coordinates, patches_images, bands):
        super(GeoDataset, self).__init__()
        from osgeo import gdal
        self.images_path = images_path
        self.raster_gt_path = raster_gt_path
        self.patches_coordinates = patches_coordinates
        self.patches_images = patches_images
        self.bands = bands

    def __len__(self):
        return len(self.patches_coordinates)

    def __getitem__(self, i):
        img_path = self.images_path[self.patches_images[i]]
        raster = gdal.Open(img_path, gdal.GA_ReadOnly)
        raster_gt = gdal.Open(self.raster_gt_path[img_path], gdal.GA_ReadOnly)
        patch = raster.ReadAsArray(*self.patches_coordinates[i], band_list=self.bands)
        patch = patch.transpose(1, 2, 0) / 10 ** 4
        patch = torch.from_numpy(patch)
        gt = raster_gt.ReadAsArray(*self.patches_coordinates[i])
        return patch, gt


def is_polygon_in_rectangle(bounds: np.ndarray, rectangle: Tuple[int]) -> bool:
    """
    Calculates if a rectangle contains a polygon.

    :param bounds: (left, top, right, bottom) bounds of a polygon
    :param rectangle: (left, top, right, bottom) bounds of a rectangle
    :return: True if the polygon is in the rectangle
    """
    in_rectangle = (rectangle[0] < bounds[0]) * (bounds[2] < rectangle[2]) * \
                   (bounds[1] < rectangle[1]) * (rectangle[3] < bounds[3])

    return in_rectangle


def bandNumbersToBbl(n_bands, band_numbers):
    """
    :n_bands: total number of bands
    :band_numbers: list of bad band intervals (for instance [[3, 8]]
    means that bands 3 (included) to 8 (excluded) are bad.)

    :returns: boolean array of bad bands
    """

    bbl = np.ones(n_bands)
    for interval in band_numbers:
        for band_number in np.arange(interval[0], interval[1]):
            bbl[band_number] = 0
    return bbl.astype(np.bool)


def spectra_bbm(spectra, mask_bands):
    """
    Args:
        - spectra: npy array, HS cube
        - mask_bands: npy boolean array, masked bands
    Output:
        HS cube with NaN at masked band locations
    """
    mask_bands = np.array(mask_bands).astype(bool)
    res = np.zeros((spectra.shape[0], len(mask_bands)))
    res[:, mask_bands] = spectra
    res[:, mask_bands == False] = np.nan
    return res


def rasterize_gt(path_gt, path_images, attribute: str = 'Material', folder: str = None):
    """
    Rasterize the shapefile ground truth.
    """
    gt = gpd.read_file(path_gt)
    gt_paths = {}

    def shapes(gt: GeoDataFrame, attribute: str):
        indices = gt.index
        for i in range(len(gt)):
            if np.isnan(gt.loc[indices[i], attribute]):
                yield gt.loc[indices[i], 'geometry'], 0
            else:
                yield gt.loc[indices[i], 'geometry'], int(gt.loc[indices[i], attribute])

    groups = list(gt.groupby(by='Image').groups.keys())
    for id, path in enumerate(path_images):
        if (id + 1) in groups:
            img = rasterio.open(path)
            shape = img.shape
            data = rasterize(shapes(gt.groupby(by='Image').get_group(id + 1), attribute), shape[:2], dtype='uint8',
                             transform=img.transform)
            data = data.reshape(1, data.shape[0], data.shape[1]).astype(int)
            with rasterio.Env():
                profile = img.profile
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    compress='lzw')
                gt_path = folder + 'gt_{}_{}.bsq'.format(attribute, id)
                gt_paths[path] = gt_path
                with rasterio.open(gt_path, 'w', **profile) as dst:
                    dst.write(data)
    return gt_paths
