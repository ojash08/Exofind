import numpy as np


def make_transit_box(
        phase,
        center,
        duration
):
    """
    Returns a boolean mask indicating
    which points are inside the transit.
    """

    half = duration / 2

    # Circular distance on a folded phase [0,1]
    distance = np.abs(phase - center)

    distance = np.minimum(
        distance,
        1.0 - distance
    )

    inside = distance <= half

    return inside