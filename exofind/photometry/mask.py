import numpy as np


def mask_from_growth_order(shape, growth_order, n_pixels):
    """
    Create an aperture mask using the first n_pixels
    from the growth order.
    """

    mask = np.zeros(shape, dtype=bool)

    for r, c in growth_order[:n_pixels]:
        mask[r, c] = True

    return mask