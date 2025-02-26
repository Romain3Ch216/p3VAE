import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

img = np.load(sys.argv[1])
#img = img[:,:-1,:]
rgb_bands = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])) # 0 15 38 

fig, ax = plt.subplots(figsize=(10, 10))
b = np.mean(img[:,:,rgb_bands[0]:rgb_bands[0]+1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
v = np.mean(img[:,:,rgb_bands[1]:rgb_bands[1]+1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
r = np.mean(img[:,:,rgb_bands[2]:rgb_bands[2]+1], axis=-1).reshape(img.shape[0], img.shape[1], 1)
rgb = 3*np.concatenate((r,v,b), axis=-1)
plt.imshow(rgb[:-1,:,:])
plt.xticks([])
plt.yticks([])


# colors = ['#F2A65A', '#909CC2', '#89A7A7', '#6320EE', '#E94974']

# ax.add_patch(patches.Rectangle((13.5,8.5), 1, 1, lw=3, edgecolor=colors[4], fill=False))
# ax.add_patch(patches.Rectangle((16.5,8.5), 1, 1, lw=3, linestyle='dashed', edgecolor=colors[4], fill=False))

# ax.add_patch(patches.Rectangle((5.5,3.5), 1, 1, lw=3, edgecolor=colors[3], fill=False))
# ax.add_patch(patches.Rectangle((5.5,8.5), 1, 1, lw=3, linestyle='dashed', edgecolor=colors[3], fill=False))

plt.savefig('rgb.pdf', dpi=200, bbox_inches = 'tight', pad_inches = 0.05)
