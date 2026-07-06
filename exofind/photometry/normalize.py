import numpy as np


def normalize_flux(flux):
    """
    Normalize the light curve by its median.
    """

    if len(flux) == 0:
        return flux

    median_flux = np.median(flux)

    # Defensive check: if median is zero or NaN, fallback to array of ones
    if median_flux == 0 or np.isnan(median_flux):
        return np.ones_like(flux, dtype=float)

    return flux / median_flux