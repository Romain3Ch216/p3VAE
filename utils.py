# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
import pdb

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
import rasterio
from rasterio.features import rasterize
from typing import List, Dict
import torch
import torch.nn as nn
# from osgeo import gdal
from scipy.ndimage import gaussian_filter1d
import numpy as np


def triangle_scheduler(length, section, amplitude, exp=1/20):
    x = np.arange(length)
    scheduler = []
    for e in x:
        if (e//section)%2 == 1:
            scheduler.append(amplitude*(e%section)/section)
        else:
            scheduler.append(amplitude - amplitude*(e%section)/section)
    scheduler = np.array(scheduler)
    x = np.arange(scheduler.shape[0])
    x = [np.exp(-e*exp) for e in x]
    scheduler = x*scheduler
    return scheduler

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        # import pdb 
        # pdb.set_trace()
        m = arr_2d == c
        arr_3d[m] = [e*255 for e in i]

    return arr_3d

def average(L):
    avg = 0 
    c = 0
    for e in L:
        if np.isnan(e) == False:
            avg += e 
            c += 1
    return avg/c

def rasterize_shapefile(path_shapefile: str, path_images: List[str], attribute: str = 'Material', output_path: str = None):
    """
    Rasterize a shapefile.
    Args:
        path_shapefile: path of the shapefile
        path_images: paths to the images on which the shapefile is defined
        attribute: attribute of the polygons to extract
        output_path: path where to save the rasters (eg '/home/Documents/train_ground_truth'
    """

    gt = gpd.read_file(path_shapefile)
    img_gt = []

    def shapes(gt: GeoDataFrame, attribute: str):
        indices = gt.index
        for i in range(len(gt)):
            if np.isnan(gt.loc[indices[i], attribute]):
                yield gt.loc[indices[i], 'geometry'], 0
            else:
                yield gt.loc[indices[i], 'geometry'], int(gt.loc[indices[i], attribute])

    gt = gt.groupby(by='Image')
    for id, path in enumerate(path_images):
        img = rasterio.open(path)
        shape = img.shape
        if (id+1) in gt.groups:
            data = rasterize(shapes(gt.get_group(id + 1), attribute), shape[:2], dtype='uint8',
                             transform=img.transform)
            data = data.reshape(1, data.shape[0], data.shape[1]).astype(int)
            with rasterio.Env():
                profile = img.profile
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    compress='lzw')
                with rasterio.open(output_path + '_image_{}.bsq'.format(id), 'w', **profile) as dst:
                    dst.write(data)


def read_spectra_from_rasters(rasters: List[rasterio.DatasetReader],
                              gts: List[rasterio.DatasetReader],
                              bands: List,
                              bbl: List,
                              classes: List,
                              sigma: float = 1.5) -> Dict[(int, Dict)]:
    """
    Extract spectra from rasters based on ground truth.

    :param rasters: list of raster images
    :param gts: list of raster ground truth
    :param bands: list of bands to select
    :param bbl: bad band list
    :param classes: list of classes to select
    :param subsample: number of samples to select
    :param core_set: whether to apply a core set selection with a k-greedy center algorithm
    :return: a dict whose keys are class ids and values are spectra and image coordinates
    """
    n_bands = n_bands_(bbl)
    filters = {}
    for i in range(len(n_bands)):
        filters[f'conv-{i}'] = GaussianConvolution(sigma=sigma, n_channels=n_bands[i])
    preprocessing_filter = HyperspectralWrapper(filters)

    spectra = dict((int(class_id), {'data': [], 'coords': []}) for class_id in classes)

    gt_ids = [int(gt[-5]) for gt in gts]
    rasters = [rasters[i] for i in gt_ids]

    for i, (raster, gt) in enumerate(zip(rasters, gts)):
        print('---> Raster {}'.format(i))
        spectra_ = read_spectra_from_raster(raster, gt, bands, preprocessing_filter)
        for class_id in spectra:
            if class_id in spectra_:
                spectra[class_id]['data'].append(spectra_[class_id]['data'])
                spectra[class_id]['coords'].append(spectra_[class_id]['coords'])
            else:
                spectra[class_id]['data'].append(None)
                spectra[class_id]['coords'].append(None)

    return spectra


def read_spectra_from_raster(raster: rasterio.DatasetReader,
                             gt: rasterio.DatasetReader,
                             bands: List,
                             preprocessing: callable = None,
                             block_size: int = 500,
                             with_data: bool = True) -> Dict:
    """
    Extract spectra from a raster based on ground truth.

    :param raster: image raster
    :param gt: ground truth raster
    :param bands: bands to select
    :param classes: classes to select
    :param count: number of labeled pixels per class
    :param subsample: number of samples to randomly select per class
    :param block_size: size of the window that slides along the image
    :param with_data: whether to return data with coordinates
    :return: a dict whose keys are class ids and values are spectra (if with_data is True) and image coordinates
    """
    raster = gdal.Open(raster, gdal.GA_ReadOnly)
    gt = gdal.Open(gt, gdal.GA_ReadOnly)
    H, W = raster.RasterYSize, raster.RasterXSize
    n_blocks_h = H // block_size + 1
    n_blocks_w = W // block_size + 1
    bands = [int(x) for x in bands]
    classes = get_classes(gt)
    if with_data:
        spectra = dict((class_id, {'data': [], 'coords': []}) for class_id in classes)
    else:
        spectra = dict((class_id, {'coords': []}) for class_id in classes)

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            row_offset = block_size * i
            col_offset = block_size * j
            row_size = min(block_size, H - row_offset)
            col_size = min(block_size, W - col_offset)

            gt_ = gt.ReadAsArray(col_offset, row_offset, col_size, row_size)
            classes_in_block = sum([(gt_ == class_id).sum() for class_id in classes]) > 0

            if classes_in_block:
                if with_data:
                    block = raster.ReadAsArray(col_offset, row_offset, col_size, row_size, band_list=bands)
                    block = np.transpose(block, (1, 2, 0))

                for class_id in spectra:
                    coords = np.where(gt_ == class_id)
                    if with_data:
                        data = block[coords]
                        data = preprocessing(data)
                        spectra[class_id]['data'].append(data)

                    coords = tuple((coords[0] + row_offset, coords[1] + col_offset))
                    spectra[class_id]['coords'].append(coords)

    for class_id in spectra:
        if with_data:
            spectra[class_id]['data'] = np.concatenate(spectra[class_id]['data'], axis=0)
        spectra[class_id]['coords'] = tuple((np.concatenate([coords[0] for coords in spectra[class_id]['coords']]),
                                             np.concatenate([coords[1] for coords in spectra[class_id]['coords']])))
    return spectra

class HyperspectralWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models):
        super(HyperspectralWrapper, self).__init__()
        self.models = nn.ModuleDict(models)

    @property
    def out_channels(self):
        with torch.no_grad():
            n_channels = sum([model.n_channels for model in self.models.values()])
            x = torch.ones((2, n_channels))
            x = self.forward(x)
        return x.numel()//2

    def forward(self, x):
        z, B = {}, 0

        for model_id, model in self.models.items():
            z[model_id] = model(x[:, B:B+model.n_channels])
            B += model.n_channels

        keys = list(z.keys())
        out = np.concatenate([z[keys[i]] for i in range(len(z))], axis=1)

        return out


class GaussianConvolution(torch.nn.Module):
    def __init__(self, sigma, n_channels):
        super(GaussianConvolution, self).__init__()
        self.sigma = sigma
        self.n_channels = n_channels

    def forward(self, x):
        x = gaussian_filter1d(x, sigma=self.sigma)
        return x


def n_bands_(bbl):
    n_bands = []
    good_bands = np.where(bbl==True)[0]
    s = 1
    for i in range(len(good_bands)-1):
        if good_bands[i] == good_bands[i+1]-1:
            s += 1
        else:
            n_bands.append(s)
            s = 1

    n_bands.append(s)
    return n_bands


def smooth_feature_(data, n_bands):
    n_bands = np.cumsum([0]+n_bands)
    s = torch.zeros(data.shape[0])
    for i in range(len(n_bands)-1):
        x = data[:, n_bands[i]+1:n_bands[i+1]] - data[:, n_bands[i]: n_bands[i+1]-1]
        x = x[:, 1:] - x[:, :-1]
        s += torch.mean(x**2, dim=1)
    return s/(len(n_bands)-1)




def get_bands(path_image):
        """
        Extract the band numbers and the bad band list from the header of the first image.
        """
        src = rasterio.open(path_image)
        bbl = src.tags(ns=src.driver)['bbl'].replace(' ', '').replace('{', '').replace('}', '').split(',')
        bbl = np.array(list(map(int, bbl)), dtype=int)
        bands = tuple(np.where(bbl != 0)[0] + 1)
        return bands, bbl.astype(bool)


def wv(path_image):
    src = rasterio.open(path_image)
    wv = src.tags(ns=src.driver)['wavelength'].replace(' ', '').replace('{', '').replace('}', '').split(',')
    wv = np.array(wv).astype(np.float32)
    return wv


def get_classes(gt):
    gt = gt.ReadAsArray()
    classes = np.unique(gt)
    classes = classes[classes != 0]
    return classes


def concatenate_dict(spectra, classes):
    data = np.concatenate([
                np.concatenate([x for x in spectra[class_id]['data'] if x is not None])
            for class_id in classes])
    labels = np.concatenate([
                np.concatenate([int(class_id) * np.ones(x.shape[0]) for x in spectra[class_id]['data'] if x is not None])
            for class_id in classes])
    return data, labels


