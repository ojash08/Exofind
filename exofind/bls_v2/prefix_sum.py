"""
prefix_sum.py — Weighted Prefix Sums

Equations [D1], [D2] from Kovács, Zucker & Mazeh (2002)

Build cumulative sums over the binned data to enable
O(1) range queries for any contiguous bin interval.

[D1] Prefix sum of weighted flux:
    PF[k] = sum(S_j, j=0..k-1),    PF[0] = 0

[D2] Prefix sum of weights:
    PR[k] = sum(R_j, j=0..k-1),    PR[0] = 0

For any bin range [i1, i2):
    s(i1, i2) = PF[i2] - PF[i1]
    r(i1, i2) = PR[i2] - PR[i1]

Mathematical invariants:
    PF[n_bins] = 0    (because sum(w_tilde * x_tilde) = 0)
    PR[n_bins] = 1    (because sum(w_tilde) = 1)
"""

import numpy as np


def build_prefix_sums(S, R):
    """
    Build prefix sums from binned weighted data.

    Parameters
    ----------
    S : ndarray, shape (n_bins,)
        Weighted flux sum per bin (from bins.py).

    R : ndarray, shape (n_bins,)
        Weight sum per bin (from bins.py).

    Returns
    -------
    PF : ndarray, shape (n_bins + 1,)
        Prefix sum of weighted flux.
        PF[0] = 0, PF[k] = sum(S[0:k]).

    PR : ndarray, shape (n_bins + 1,)
        Prefix sum of weights.
        PR[0] = 0, PR[k] = sum(R[0:k]).
    """

    PF = np.concatenate(([0.0], np.cumsum(S)))

    PR = np.concatenate(([0.0], np.cumsum(R)))

    return PF, PR