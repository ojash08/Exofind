"""
vetting/odd_even.py — Odd/Even Transit Mismatch Test

Eclipsing binaries often have primary and secondary eclipses
of different depths. When folded at half the true orbital period,
the odd-numbered transits are primary and the even-numbered
transits are secondary (or vice-versa).
This test checks if odd and even transits have significantly
different depths.
"""

import numpy as np
from bls_v2.fold import fold_lightcurve
from bls_v2.weights import normalize_weights
from bls_v2.bins import bin_folded
from bls_v2.prefix_sum import build_prefix_sums
from bls_v2.sr import transit_parameters


def check_odd_even_mismatch(time, flux, period, phase_start, phase_width, n_bins=400):
    """
    Compare the depth of odd and even transits.

    Returns a dictionary with the depths and the statistical significance
    of the mismatch.
    """
    # 1. Separate time into odd and even epochs
    epochs = np.floor(time / period)
    odd_mask = (epochs % 2) != 0
    even_mask = (epochs % 2) == 0

    if np.sum(odd_mask) == 0 or np.sum(even_mask) == 0:
        return {"error": "Not enough transits to split odd/even"}

    x_odd, w_odd = normalize_weights(flux[odd_mask])
    x_even, w_even = normalize_weights(flux[even_mask])

    # 2. Extract depth for odd transits
    depth_odd = _extract_depth_at_phase(
        time[odd_mask], x_odd, w_odd, period, phase_start, phase_width, n_bins
    )

    # 3. Extract depth for even transits
    depth_even = _extract_depth_at_phase(
        time[even_mask], x_even, w_even, period, phase_start, phase_width, n_bins
    )

    if depth_odd is None or depth_even is None:
         return {"error": "Degenerate transit fit during odd/even split"}

    # 4. Compare
    # Simple metric: fractional difference
    mean_depth = (depth_odd + depth_even) / 2.0
    if mean_depth == 0:
        diff_sigma = 0
    else:
        diff_frac = abs(depth_odd - depth_even) / abs(mean_depth)
        
        # A rough heuristic for significance: if depths differ by > 20%, flag it.
        # A more rigorous test would use the errors on the depth.
        diff_sigma = diff_frac / 0.1  # Suppose 10% is 1-sigma

    return {
        "depth_odd": depth_odd,
        "depth_even": depth_even,
        "diff_frac": diff_frac if mean_depth != 0 else 0,
        "is_false_positive": diff_sigma > 3.0,
        "diff_sigma": diff_sigma
    }


def _extract_depth_at_phase(time, x_tilde, w_tilde, period, phase_start, phase_width, n_bins):
    """Helper to force a transit fit at a specific phase window."""
    phase, f, w = fold_lightcurve(time, x_tilde, w_tilde, period)
    S, R = bin_folded(phase, f, w, n_bins)
    PF, PR = build_prefix_sums(S, R)

    start_bin = int(phase_start * n_bins)
    width_bins = int(phase_width * n_bins)
    
    end_bin = start_bin + width_bins

    if end_bin <= n_bins:
        s = PF[end_bin] - PF[start_bin]
        r = PR[end_bin] - PR[start_bin]
    else:
        end_wrapped = end_bin - n_bins
        s = (PF[n_bins] - PF[start_bin]) + PF[end_wrapped]
        r = (PR[n_bins] - PR[start_bin]) + PR[end_wrapped]

    params = transit_parameters(s, r)
    if params is None:
        return None
    return abs(params["depth"])
