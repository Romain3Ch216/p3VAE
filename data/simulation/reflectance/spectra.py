import numpy as np 
import matplotlib.pyplot as plt  
import sys 
sys.path.insert(1, '/home/rothor/Documents/ONERA/code/2A/RTB-VAE')
from data import spectra_bbm 

wv = np.load('../wv.npy')
bbl = np.load('../bbl.npy')

grass = 'grass.npy'
dry_grass = 'dry_grass.npy'
tree = 'tree.npy'
alu = 'alu.npy'
loam_1 = 'loam_1.npy'
loam_2 = 'loam_2.npy'
tile_1 = 'tile_1.npy'
tile_2 = 'tile_2.npy'
tile_3 = 'tile_3.npy'
asphalt = 'asphalt.npy'

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

fig = plt.figure(figsize=(15,10))
style = ['solid', 'dotted', 'dashed']

# for class_id in classes:
#     if class_id != 0:
#         label = classes[class_id]['label']
#         color = [classes[class_id]['rgb'][i]/256 for i in range(3)]
#         for i, sp in enumerate(classes[class_id]['spectrum']):
#             plt.plot(wv, spectra_bbm(sp.reshape(1,-1), bbl).reshape(-1), label=label + '_' + str(i+1) , color=color, linestyle=style[i], lw=2)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel(r'Wavelenght ($\mu m$)', fontsize=15)
# plt.ylabel('Reflectance', fontsize=15)

for class_id in classes:
    if class_id != 0:
        fig_, ax = plt.subplots(figsize=(15,10))
        style = ['solid', 'dotted', 'dashed']
        label = classes[class_id]['label']
        color = [classes[class_id]['rgb'][i]/256 for i in range(3)]
        for i, sp in enumerate(classes[class_id]['spectrum']):
            sp = sp[bbl]
            plt.plot(wv, spectra_bbm(sp.reshape(1,-1), bbl).reshape(-1), label=label + '_' + str(i+1) , color=color, linestyle=style[i], lw=5)
        #plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel(r'Wavelenght ($\mu m$)', fontsize=25)
        plt.ylabel('Reflectance', fontsize=25)
        plt.ylim(0, 0.7)
        plt.savefig('spectra_{}.pdf'.format(class_id), dpi=200, bbox_inches='tight', pad_inches=0.05)
