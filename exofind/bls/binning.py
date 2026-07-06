import numpy as np


def bin_lightcurve(phase, flux, bins=100):

    edges = np.linspace(0, 1, bins + 1)

    centers = (edges[:-1] + edges[1:]) / 2

    binned_flux = np.full(bins, np.nan)

    for i in range(bins):

        mask = (phase >= edges[i]) & (phase < edges[i + 1])

        if np.any(mask):
            binned_flux[i] = np.mean(flux[mask])

    return centers, binned_flux