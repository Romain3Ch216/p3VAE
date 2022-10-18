# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.
# Data set definition 

import numpy as np 
import torch  
import matplotlib.pyplot as plt 
import spectral.io.envi as envi 
import pdb 

def load_dataset(config):
    if config['dataset'] == 'simulation':
        train_img = './data/simulation/train_img.npy'
        test_img = './data/simulation/test_img.npy'
        test_img_coupling = './data/simulation/test_img_coupling.npy'
        train_gt = './data/simulation/train_gt.npy'
        val_gt = './data/simulation/val_gt.npy'
        test_gt = './data/simulation/test_gt.npy'
        E_dir = './data/simulation/E_dir.npy'
        E_dif = './data/simulation/E_dif.npy'
        theta = 30*np.pi/180
        wv = './data/simulation/wv.npy'
        rgb_bands = [0, 12, 38]
        bb = [[45, 49], [66, 82], [98, 109], [131, 162], [201, 242], [289,300]]

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

        test_spectra = './data/simulation/test_spectra.npy'
        test_labels = './data/simulation/test_labels.npy'

        classes = {
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

        dataset = Dataset(train_img, test_img, test_img_coupling, train_gt, val_gt, test_gt,
                   test_spectra, test_labels,
                   E_dir, E_dif, theta, rgb_bands, bb, wv, classes)

    return dataset

class Dataset:
    def __init__(self, train_img, test_img, test_img_coupling, train_gt, val_gt, test_gt,
                       test_spectra, test_labels,
                       E_dir, E_dif, theta, rgb_bands, bb, wv, classes):

        self.train_img = np.load(train_img)
        self.test_img = np.load(test_img)
        self.test_img_coupling = np.load(test_img_coupling)
        self.train_gt = np.load(train_gt)
        self.val_gt = np.load(val_gt)
        self.test_gt = np.load(test_gt)
        self.test_spectra = np.load(test_spectra)
        self.test_labels = np.load(test_labels)
        self.wv = np.load(wv)
        self.E_dir = np.load(E_dir).astype(np.float32)
        self.E_dif = np.load(E_dif).astype(np.float32)
        self.theta = theta
        self.B = self.train_img.shape[-1]
        self.bbl = bandNumbersToBbl(self.B, bb)
        self.train_img = self.train_img[:,:,self.bbl]
        self.test_img = self.test_img[:,:,self.bbl]
        self.test_img_coupling = self.test_img_coupling[:,:,self.bbl]
        self.E_dir = self.E_dir[self.bbl]/np.cos(self.theta)
        self.E_dif = self.E_dif[self.bbl]
        self.test_spectra = self.test_spectra[:,self.bbl]
        self.rgb_bands = rgb_bands
        self.n_channels = sum(self.n_bands_)
        self.classes = classes
        for class_id in self.classes:
            for i in range(len(self.classes[class_id]['spectrum'])):
                self.classes[class_id]['spectrum'][i] = self.classes[class_id]['spectrum'][i][self.bbl]


    @property
    def palette(self):
        color = {}
        for class_id in self.classes:
            color[class_id] = tuple([color_/256 for color_ in self.classes[class_id]['rgb']])
        return color

    @property
    def labels(self):
        labels_ = []
        for class_id in self.classes:
            labels_.append(self.classes[class_id]['label'])
        return labels_

    @property
    def n_bands_(self):
        n_bands = []
        good_bands = np.where(self.bbl==True)[0]
        s = 1
        for i in range(len(good_bands)-1):
            if good_bands[i] == good_bands[i+1]-1:
                s += 1
            else:
                n_bands.append(s)
                s = 1

        n_bands.append(s)
        return n_bands

    def gt_to_rgb(self, gt):
        plot = np.zeros((gt.shape[0], gt.shape[1], 3))
        for class_id in np.unique(gt):
            plot[gt == class_id] = self.classes[class_id]['rgb']
        return plot/256

    def img_to_rgb(self, img):
        fig = plt.figure(figsize=(10, 10))
        b = np.mean(img[:,:,self.rgb_bands[0]:self.rgb_bands[0]+1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
        v = np.mean(img[:,:,self.rgb_bands[1]:self.rgb_bands[1]+1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
        r = np.mean(img[:,:,self.rgb_bands[2]:self.rgb_bands[2]+1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
        rgb = 3*np.concatenate((r,v,b), axis=-1)
        plt.imshow(rgb)
        plt.xticks([])
        plt.yticks([])
        return fig

    def plot_spectra(self):
        fig = plt.figure(figsize=(15,10))
        style = ['solid', 'dotted', 'dashed']

        for class_id in self.classes:
            if class_id != 0:
                label = self.classes[class_id]['label']
                color = [self.classes[class_id]['rgb'][i]/256 for i in range(3)]
                for i, sp in enumerate(self.classes[class_id]['spectrum']):
                    plt.plot(self.wv, spectra_bbm(sp.reshape(1,-1), self.bbl).reshape(-1), label=label + '_' + str(i+1) , color=color, linestyle=style[i], lw=2)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel(r'Wavelenght ($\mu m$)', fontsize=15)
        plt.ylabel('Reflectance', fontsize=15)

        for class_id in self.classes:
            if class_id != 0:
                fig_ = plt.figure(figsize=(15,10))
                style = ['solid', 'dotted', 'dashed']
                label = self.classes[class_id]['label']
                color = [self.classes[class_id]['rgb'][i]/256 for i in range(3)]
                for i, sp in enumerate(self.classes[class_id]['spectrum']):
                    plt.plot(self.wv, spectra_bbm(sp.reshape(1,-1), self.bbl).reshape(-1), label=label + '_' + str(i+1) , color=color, linestyle=style[i], lw=2)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.xlabel(r'Wavelenght ($\mu m$)', fontsize=15)
                plt.ylabel('Reflectance', fontsize=15)
                plt.ylim(0,1)
                plt.savefig('./Figures/reflectance_spectra.pdf'.format(class_id), dpi=200, bbox_inches = 'tight', pad_inches = 0.05)
        return fig

    def load(self, img, gt, batch_size, unlabeled=False, test=False, split=False):
        config = {'patch_size': 1, 'batch_size': batch_size, 'ignored_labels': [0]}
        if unlabeled:
            config['ignored_labels'] = list(np.arange(1, len(np.unique(gt))))
        if test:
            shuffle =  False
        else:
            shuffle = True

        data = HyperX(img, gt, config)
        n_data = len(data)
        n_train = int(0.8*n_data)
        n_val = n_data - n_train
        if split:
            train_data, val_data = torch.utils.data.random_split(data, [n_train, n_val])
            train_data = torch.utils.data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
            val_data = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
            return train_data, val_data 
        else:
            data = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
            return data

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
            data = data*(1+torch.randn((data.shape[-2], data.shape[-1]))*0.05)

        return data, label


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
    res = np.zeros((spectra.shape[0],len(mask_bands)))
    res[:, mask_bands] = spectra
    res[:, mask_bands==False] = np.nan
    return res
