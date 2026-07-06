"""
fold.py — Phase Folding

Equations [B1], [B2] from Kovács, Zucker & Mazeh (2002)

[B1] Phase calculation:
    phi_i = (t_i mod P) / P,    phi in [0, 1)

[B2] Sort all arrays by phase.
"""

import numpy as np


def fold_lightcurve(time, flux, weights, period):
    """
    Fold a light curve onto a trial period and
    sort all arrays by phase.

    Parameters
    ----------
    time : ndarray
        Observation timestamps (days).

    flux : ndarray
        Mean-subtracted flux (x_tilde).

    weights : ndarray
        Normalized weights (w_tilde).

    period : float
        Trial period (days).

    Returns
    -------
    phase : ndarray
        Phase values in [0, 1), sorted.

    flux : ndarray
        Flux sorted by phase.

    weights : ndarray
        Weights sorted by phase.
    """

    # ----- [B1] Compute phase -----

    phase = np.mod(time, period) / period

    # ----- [B2] Sort by phase -----

    order = np.argsort(phase)

    return (
        phase[order],
        flux[order],
        weights[order]
    )