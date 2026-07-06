import numpy as np
from photometry.snr import photometric_snr
from photometry.snr import signal, noise, photometric_snr


def rms(flux):
    """
    Root Mean Square scatter of the light curve.
    Lower is better.
    """
    flux = np.asarray(flux)

    if len(flux) == 0:
        return 0.0

    median = np.median(flux)
    
    if np.isnan(median):
        return 0.0

    return np.sqrt(np.mean((flux - median) ** 2))

def aperture_size(mask):
    """
    Number of pixels inside the aperture.
    """

    return np.sum(mask)

def quality_score(flux, mask):
    """
    Simple quality metric.

    Higher is better.
    """

    size = aperture_size(mask)
    if size == 0 or len(flux) == 0:
        return 0.0

    return photometric_snr(flux) / np.sqrt(size)