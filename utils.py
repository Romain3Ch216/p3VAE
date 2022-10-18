# Copyright (c) 2022 ONERA, Magellium and IMT, Romain Thoreau, Laurent Risser, Véronique Achard, Béatrice Berthelot, Xavier Briottet.

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