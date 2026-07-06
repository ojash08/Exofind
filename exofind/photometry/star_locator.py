import numpy as np


def average_image(tpf):
    """
    Compute the average image over all frames.
    """
    return np.nanmean(tpf.flux.value, axis=0)

def brightest_pixel(image):

    return np.unravel_index(
        np.nanargmax(image),
        image.shape
    )