import numpy as np


def fold_lightcurve(time, flux, period):
    """
    Fold a light curve using a trial period.
    """

    phase = (time % period) / period
    order = np.argsort(phase)

    phase = phase[order]
    flux = flux[order]

    return phase, flux