import numpy as np


def transit_score(flux):
    """
    Simple score for a folded light curve.

    Larger score means a deeper,
    cleaner transit.
    """

    median = np.median(flux)

    minimum = np.min(flux)

    depth = median - minimum

    scatter = np.std(flux)

    if scatter == 0:
        return 0

    return depth / scatter