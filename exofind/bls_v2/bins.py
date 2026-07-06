"""
bins.py — Weighted Phase Binning

Equations [C1], [C2] from Kovács, Zucker & Mazeh (2002)

Divide the phase interval [0, 1) into n_bins equal bins.

[C1] Binned weighted flux:
    S_j = sum(w_tilde_i * x_tilde_i)   for i in bin j

[C2] Binned weight sum:
    R_j = sum(w_tilde_i)                for i in bin j

These are the inputs to the prefix sum stage.
"""

import numpy as np


def bin_folded(phase, flux, weights, n_bins=400):
    """
    Bin a folded light curve into weighted sums.

    Parameters
    ----------
    phase : ndarray
        Sorted phase values in [0, 1).

    flux : ndarray
        Mean-subtracted flux (x_tilde), sorted by phase.

    weights : ndarray
        Normalized weights (w_tilde), sorted by phase.

    n_bins : int
        Number of phase bins.

    Returns
    -------
    S : ndarray, shape (n_bins,)
        Weighted flux sum in each bin.
        S_j = sum(w_tilde_i * x_tilde_i) for i in bin j.

    R : ndarray, shape (n_bins,)
        Weight sum in each bin.
        R_j = sum(w_tilde_i) for i in bin j.
    """

    S = np.zeros(n_bins, dtype=np.float64)
    R = np.zeros(n_bins, dtype=np.float64)

    # Assign each point to a bin
    bin_indices = np.floor(phase * n_bins).astype(int)

    # Clamp to valid range (phase = 1.0 edge case)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # ----- [C1] Accumulate weighted flux -----
    # ----- [C2] Accumulate weights -----

    np.add.at(S, bin_indices, weights * flux)
    np.add.at(R, bin_indices, weights)

    return S, R