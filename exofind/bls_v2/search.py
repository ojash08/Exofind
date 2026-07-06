"""
search.py — BLS Period Search Orchestrator

Equation Group [H] from Kovács, Zucker & Mazeh (2002)

For each trial period P:
    1. Fold the light curve at P           → fold.py      [B1, B2]
    2. Bin into n_bins phase bins           → bins.py      [C1, C2]
    3. Build prefix sums                    → prefix_sum.py [D1, D2]
    4. Slide the transit window             → window.py    [F1–F3]
    5. Record SR_max(P) and best-fit params

This module ties together the complete BLS pipeline.
"""

import numpy as np

from bls_v2.weights import normalize_weights
from bls_v2.fold import fold_lightcurve
from bls_v2.bins import bin_folded
from bls_v2.prefix_sum import build_prefix_sums
from bls_v2.window import slide_window
from bls_v2.periodogram import (
    generate_period_grid,
    duration_grid
)


def search_single_period(
        time,
        flux,
        weights,
        period,
        n_bins=400,
        min_duration_frac=0.01,
        max_duration_frac=0.15
):
    """
    Run the full BLS pipeline for one trial period.

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

    n_bins : int
        Number of phase bins.

    min_duration_frac : float
        Minimum transit duration as fraction of period.

    max_duration_frac : float
        Maximum transit duration as fraction of period.

    Returns
    -------
    result : dict or None
        Best transit candidate for this period, with keys:
            period, start, width, sr, depth, L, H,
            s, r, phase_start, phase_width,
            duration_days
    """

    # Step 1: Fold
    phase, folded_flux, folded_weights = fold_lightcurve(
        time, flux, weights, period
    )

    # Step 2: Bin
    S, R = bin_folded(
        phase, folded_flux, folded_weights,
        n_bins=n_bins
    )

    # Step 3: Prefix sums
    PF, PR = build_prefix_sums(S, R)

    # Step 4: Slide window
    min_width, max_width = duration_grid(
        period, n_bins,
        min_fraction=min_duration_frac,
        max_fraction=max_duration_frac
    )

    best = slide_window(
        PF, PR, n_bins,
        min_width, max_width
    )

    # Step 5: Record result
    if best is None:
        return None

    best["period"] = period
    best["duration_days"] = best["phase_width"] * period

    return best


def search_all_periods(
        time,
        flux,
        errors=None,
        periods=None,
        n_bins=400,
        min_period=None,
        max_period=None,
        min_duration=None,
        max_duration=None,
        min_n_transits=3,
        frequency_factor=1.0
):
    """
    Run the complete BLS search over all trial periods.

    Parameters
    ----------
    time : ndarray
        Observation timestamps (days).

    flux : ndarray
        Raw flux measurements.

    errors : ndarray or None
        Per-point measurement uncertainties.

    periods : ndarray or None
        Explicit period grid. If None, one is generated
        automatically using the Hartman & Bakos heuristic.

    n_bins : int
        Number of phase bins.

    min_period : float or None
        Minimum period to search (days).

    max_period : float or None
        Maximum period to search (days).

    min_duration : float or None
        Minimum transit duration (days).
        Default: 0.05 days (~1.2 hours).

    max_duration : float or None
        Maximum transit duration (days).
        Default: max_period * 0.15.

    min_n_transits : int
        Minimum number of transits required.

    frequency_factor : float
        Controls period grid density.

    Returns
    -------
    result : dict
        periods     : ndarray of trial periods
        sr_array    : ndarray of SR_max at each period
        best_params : list of dicts (best params per period)
        best        : dict of the overall best candidate
    """

    # ===== Normalize data (once) =====

    x_tilde, w_tilde = normalize_weights(flux, errors)

    # ===== Generate period grid if needed =====

    baseline = np.max(time) - np.min(time)

    if min_duration is None:
        min_duration = 0.05  # ~1.2 hours

    if max_duration is None:
        max_duration = 0.25  # ~6 hours

    if periods is None:

        periods = generate_period_grid(
            baseline=baseline,
            min_duration=min_duration,
            max_duration=max_duration,
            min_period=min_period,
            max_period=max_period,
            min_n_transits=min_n_transits,
            frequency_factor=frequency_factor
        )

    # ===== Search each period =====

    n_periods = len(periods)
    sr_array = np.zeros(n_periods)
    best_params = [None] * n_periods

    overall_best = None
    overall_best_sr = -np.inf

    for k in range(n_periods):

        period = periods[k]

        # Convert duration limits to phase fractions
        min_frac = min_duration / period
        max_frac = max_duration / period

        # Clamp fractions
        min_frac = max(0.005, min(min_frac, 0.5))
        max_frac = max(min_frac + 0.005, min(max_frac, 0.5))

        result = search_single_period(
            time, x_tilde, w_tilde,
            period,
            n_bins=n_bins,
            min_duration_frac=min_frac,
            max_duration_frac=max_frac
        )

        if result is not None:

            sr_array[k] = result["sr"]
            best_params[k] = result

            if result["sr"] > overall_best_sr:
                overall_best_sr = result["sr"]
                overall_best = result

    return {
        "periods": periods,
        "sr_array": sr_array,
        "best_params": best_params,
        "best": overall_best
    }