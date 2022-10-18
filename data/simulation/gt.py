import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

path = sys.argv[1]
gt = np.load(path)

colors = {
	0: (255, 255, 255),
	1: (142, 196, 110),
	2: (159, 194, 204),
	3: (254, 215, 102),
	4: (193, 102, 107),
	5: (65, 63, 80)
}


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
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

rgb = convert_to_color_(gt, colors)
fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(rgb[:-1,:,:])
plt.xticks([])
plt.yticks([])
plt.savefig(path[:-3] + 'pdf', dpi=200, bbox_inches = 'tight', pad_inches = 0.05)
