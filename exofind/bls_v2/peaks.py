"""
peaks.py — Peak Detection and Harmonic Filtering

Equations [J1]–[J3] — Standard practice derived from
the SDE periodogram.

[J1] Identify local maxima above a threshold.
[J2] Filter harmonics (P/2, 2P, P/3, 3P, etc.).
[J3] Rank candidates by SDE.
"""

import numpy as np


def find_peaks(periods, sde, threshold=6.0):
    """
    Identify significant peaks in the SDE periodogram.

    Parameters
    ----------
    periods : ndarray
        Trial periods (days).

    sde : ndarray
        Signal Detection Efficiency values.

    threshold : float
        Minimum SDE to consider a peak significant.
        KZM 2002 recommends 6.0.

    Returns
    -------
    candidates : list of dict
        Each candidate has keys:
            period      : peak period (days)
            sde         : SDE at peak
            index       : index into periods array
            is_harmonic : whether this is likely a harmonic
    """

    n = len(sde)

    if n < 3:
        return []

    candidates = []

    # ----- [J1] Find local maxima above threshold -----

    for i in range(1, n - 1):

        if sde[i] <= threshold:
            continue

        if sde[i] > sde[i - 1] and sde[i] > sde[i + 1]:

            candidates.append({
                "period": periods[i],
                "sde": sde[i],
                "index": i,
                "is_harmonic": False
            })

    # Also check endpoints
    if n > 0 and sde[0] > threshold and sde[0] > sde[1]:
        candidates.append({
            "period": periods[0],
            "sde": sde[0],
            "index": 0,
            "is_harmonic": False
        })

    if n > 1 and sde[-1] > threshold and sde[-1] > sde[-2]:
        candidates.append({
            "period": periods[-1],
            "sde": sde[-1],
            "index": n - 1,
            "is_harmonic": False
        })

    # ----- [J3] Sort by SDE descending -----

    candidates = sorted(
        candidates,
        key=lambda c: c["sde"],
        reverse=True
    )

    # ----- [J2] Flag harmonics -----

    candidates = _flag_harmonics(candidates)

    return candidates


def _flag_harmonics(candidates, tolerance=0.02):
    """
    Flag candidates that are likely harmonics of
    the strongest peak.

    A candidate at period P2 is flagged as a harmonic
    of a candidate at period P1 (P1 being stronger) if:

        |P2 / P1 - n/m| < tolerance

    for small integer ratios n/m in {1/2, 1/3, 2/1, 3/1,
    2/3, 3/2}.

    Parameters
    ----------
    candidates : list of dict
        Sorted by SDE descending.

    tolerance : float
        Fractional tolerance for harmonic matching.

    Returns
    -------
    candidates : list of dict
        Same list, with is_harmonic flags set.
    """

    harmonic_ratios = [
        0.5, 2.0,
        1.0 / 3.0, 3.0,
        2.0 / 3.0, 3.0 / 2.0,
        0.25, 4.0
    ]

    for i, cand in enumerate(candidates):

        if cand["is_harmonic"]:
            continue

        # Check all weaker candidates
        for j in range(i + 1, len(candidates)):

            if candidates[j]["is_harmonic"]:
                continue

            ratio = candidates[j]["period"] / cand["period"]

            for hr in harmonic_ratios:

                if abs(ratio - hr) < tolerance * hr:
                    candidates[j]["is_harmonic"] = True
                    break

    return candidates
