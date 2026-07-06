import numpy as np


def estimate_global_background(image):
    """
    Estimate the background using the whole image.
    """
    return np.nanmedian(image)


def estimate_local_background(
        image,
        center,
        inner_radius=3,
        outer_radius=5,
):
    """
    Estimate the background using an annulus around the target.
    """

    rows, cols = image.shape

    y, x = np.indices((rows, cols))

    cy, cx = center

    distance = np.sqrt(
        (y - cy) ** 2 +
        (x - cx) ** 2
    )

    annulus = (
            (distance >= inner_radius)
            &
            (distance <= outer_radius)
    )

    values = image[annulus]

    return np.nanmedian(values)