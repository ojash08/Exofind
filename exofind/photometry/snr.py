import numpy as np


def signal(flux):
    """
    Median signal level.
    """
    return np.median(flux)


def noise(flux):
    """
    Noise estimated as the standard deviation.
    """
    return np.std(flux)


def photometric_snr(flux):
    """
    Signal-to-noise ratio.
    """

    if len(flux) == 0:
        return 0.0

    s = signal(flux)
    n = noise(flux)

    if n == 0 or np.isnan(n):
        return 0.0

    return s / n