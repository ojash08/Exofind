import numpy as np

from bls.phase import fold_lightcurve
from bls.bls_core import analyze_period


def search_periods(
        time,
        flux,
        minimum_period,
        maximum_period,
        step,
):
    """
    Search many trial periods using a frequency grid.
    """

    best_period = None
    best_score = -np.inf

    results = []

    observation_length = np.max(time) - np.min(time)

    # Search uniformly in frequency instead of period
    minimum_frequency = 1 / maximum_period
    maximum_frequency = 1 / minimum_period

    frequencies = np.arange(
        minimum_frequency,
        maximum_frequency,
        step
    )

    for frequency in frequencies:

        period = 1.0 / frequency

        phase, folded_flux = fold_lightcurve(
            time,
            flux,
            period
        )

        candidate = analyze_period(
            phase,
            folded_flux
        )

        if candidate is None:
            continue

        # Expected number of observed transits
        transit_count = observation_length / period

        # Require at least 3 transits
        if transit_count < 3:
            continue

        # Final detection score
        score = candidate["power"]

        results.append(
            {
                "period": period,
                "frequency": frequency,
                "score": score,
                "depth": candidate["depth"],
                "duration": candidate["duration"],
                "center": candidate["center"],
                "snr": candidate["snr"],
                "power": candidate["power"],
                "transits": transit_count,
            }
        )

        if score > best_score:
            best_score = score
            best_period = period

    return (
        best_period,
        best_score,
        results
    )