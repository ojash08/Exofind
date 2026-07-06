"""
sde.py — Signal Detection Efficiency

Equations [I1], [I2] from Kovács, Zucker & Mazeh (2002)

The SDE normalizes the SR periodogram to identify
statistically significant peaks.

[I1] Mean and standard deviation of SR:
    <SR> = (1/N_P) * sum(SR_max(P_k))
    sigma_SR = sqrt( (1/N_P) * sum( (SR_max - <SR>)^2 ) )

[I2] Signal Detection Efficiency:
    SDE(P_k) = (SR_max(P_k) - <SR>) / sigma_SR

KZM 2002 show that SDE > 6 indicates a significant
detection when the effective signal-to-noise alpha >= 6.
"""

import numpy as np


def compute_sde(sr_array):
    """
    Compute the Signal Detection Efficiency from an
    array of SR values (one per trial period).

    Parameters
    ----------
    sr_array : ndarray
        Array of SR_max values, one per trial period.

    Returns
    -------
    sde : ndarray
        SDE values, same shape as sr_array.

    mean_sr : float
        Mean of the SR distribution.

    std_sr : float
        Standard deviation of the SR distribution.
    """

    # ----- [I1] Statistics of the SR distribution -----

    mean_sr = np.mean(sr_array)
    std_sr = np.std(sr_array)

    # ----- [I2] SDE -----

    if std_sr == 0.0:
        # All SR values identical (pure noise, no signal)
        sde = np.zeros_like(sr_array)
    else:
        sde = (sr_array - mean_sr) / std_sr

    return sde, mean_sr, std_sr
