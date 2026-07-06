import numpy as np
from astropy.stats import sigma_clip


def remove_outliers(time, flux, sigma=5.0):
    """
    Remove outliers iteratively using astropy's sigma_clip.
    """

    # We only clip positive outliers (flares/cosmic rays) so we
    # don't accidentally clip out deep transits!
    filtered_flux = sigma_clip(
        flux,
        sigma_lower=float('inf'),  # Do not clip negative values (transits)
        sigma_upper=sigma,         # Clip positive spikes
        maxiters=5                 # Iterative
    )

    # sigma_clip returns a MaskedArray. We want the unmasked values.
    mask = ~filtered_flux.mask

    return (
        time[mask],
        flux[mask]
    )