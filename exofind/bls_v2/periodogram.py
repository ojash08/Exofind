"""
periodogram.py — Period Grid Generation

Equations [G1]–[G3] from Kovács, Zucker & Mazeh (2002)
and Hartman & Bakos (2016)

The BLS periodogram is more sensitive to period grid
spacing than the Lomb-Scargle periodogram, because
transit signals are sharper than sinusoids.

[G1] Frequency step (Hartman & Bakos 2016):
    df = frequency_factor * min_duration / baseline^2

[G2] Period range:
    P_min = 2 * max_duration (or user-specified)
    P_max = baseline / (min_n_transits - 1)

[G3] Generate periods from a uniform frequency grid.
"""

import numpy as np


def generate_period_grid(
        baseline,
        min_duration,
        max_duration,
        min_period=None,
        max_period=None,
        min_n_transits=3,
        frequency_factor=1.0
):
    """
    Generate a period grid uniform in frequency.

    This follows the Astropy heuristic from
    Hartman & Bakos (2016), which scales the
    frequency resolution proportionally to the
    transit duration.

    Parameters
    ----------
    baseline : float
        Total observation baseline (days).
        baseline = max(time) - min(time)

    min_duration : float
        Minimum transit duration to search (days).

    max_duration : float
        Maximum transit duration to search (days).

    min_period : float or None
        Minimum period to search (days).
        Default: 2 * max_duration.

    max_period : float or None
        Maximum period to search (days).
        Default: baseline / (min_n_transits - 1).

    min_n_transits : int
        Minimum number of transits required.
        Used only when max_period is None.

    frequency_factor : float
        Controls frequency grid density.
        Smaller = finer grid.

    Returns
    -------
    periods : ndarray
        Trial periods (days), sorted ascending.
    """

    # ----- [G2] Period range -----

    if min_period is None:
        min_period = 2.0 * max_duration

    if max_period is None:

        if min_n_transits <= 1:
            raise ValueError(
                "min_n_transits must be > 1"
            )

        max_period = baseline / (min_n_transits - 1)

    # Convert to frequency range
    max_freq = 1.0 / min_period
    min_freq = 1.0 / max_period

    # ----- [G1] Frequency step -----

    df = frequency_factor * min_duration / (baseline ** 2)

    # ----- [G3] Generate frequency grid -----

    n_freqs = int(np.ceil((max_freq - min_freq) / df)) + 1

    frequencies = min_freq + df * np.arange(n_freqs)

    # Convert to periods (sorted ascending = low freq first)
    periods = 1.0 / frequencies

    # Clip to range and sort ascending
    mask = (periods >= min_period) & (periods <= max_period)

    periods = np.sort(periods[mask])

    return periods


def duration_grid(period, n_bins, min_fraction=0.01, max_fraction=0.15):
    """
    Convert fractional transit durations into bin widths
    for a given period.

    Parameters
    ----------
    period : float
        Trial period (days).

    n_bins : int
        Number of phase bins.

    min_fraction : float
        Minimum transit duration as fraction of period.

    max_fraction : float
        Maximum transit duration as fraction of period.

    Returns
    -------
    min_width : int
        Minimum transit width in bins.

    max_width : int
        Maximum transit width in bins.
    """

    min_width = max(1, int(np.floor(min_fraction * n_bins)))
    max_width = max(min_width + 1, int(np.ceil(max_fraction * n_bins)))

    # Don't let the window exceed half the phase
    max_width = min(max_width, n_bins // 2)

    return min_width, max_width
