"""
weights.py — Data Normalization

Equations [A1], [A2] from Kovács, Zucker & Mazeh (2002)

[A1] Normalize weights:
    w_tilde_i = w_i / sum(w_j)

    so that sum(w_tilde_i) = 1.

[A2] Subtract weighted mean:
    x_bar = sum(w_tilde_i * x_i)
    x_tilde_i = x_i - x_bar

After this transformation:
    sum(w_tilde_i * x_tilde_i) = 0
"""

import numpy as np


def normalize_weights(flux, errors=None):
    """
    Normalize observation weights and subtract the
    weighted mean from the flux.

    Parameters
    ----------
    flux : ndarray
        Raw flux measurements.

    errors : ndarray or None
        Per-point measurement uncertainties (sigma_i).
        If None, uniform weights are assumed.

    Returns
    -------
    x_tilde : ndarray
        Mean-subtracted flux. Satisfies:
        sum(w_tilde * x_tilde) = 0

    w_tilde : ndarray
        Normalized weights. Satisfies:
        sum(w_tilde) = 1.0
    """

    n = len(flux)

    # ----- [A1] Compute and normalize weights -----

    if errors is not None:

        # Inverse-variance weighting
        w = 1.0 / (np.asarray(errors) ** 2)

    else:

        # Uniform weights
        w = np.ones(n, dtype=np.float64)

    w_sum = np.sum(w)

    w_tilde = w / w_sum

    # ----- [A2] Subtract weighted mean -----

    x_bar = np.sum(w_tilde * flux)

    x_tilde = flux - x_bar

    return x_tilde, w_tilde
