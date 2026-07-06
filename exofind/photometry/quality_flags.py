import numpy as np


def filter_quality(time, flux, quality):
    """
    Keep only cadences with quality == 0.
    """

    mask = quality == 0

    return (
        time[mask],
        flux[mask],
        quality[mask]
    )