import numpy as np

from bls.transit_box import make_transit_box
from bls.statistics import transit_statistics


def analyze_period(phase, flux):
    """
    Analyze one folded light curve for a single trial period.
    """

    best = None
    best_score = -np.inf

    # Use actual observed phases
    centers = np.linspace(0, 1, 300)

    # Trial transit durations (fraction of orbital phase)
    durations = np.linspace(
        0.01,
        0.05,
        15
    )

    for center in centers:

        for duration in durations:

            inside = make_transit_box(
                phase,
                center,
                duration
            )

            stats = transit_statistics(
                flux,
                inside
            )

            if stats is None:
                continue

            if stats["bls_score"] > best_score:

                best_score = stats["bls_score"]

                best = {
                    "center": center,
                    "duration": duration,
                    "snr": stats["snr"],
                    "power": stats["power"],
                    "depth": stats["depth"],
                    "ssr": stats["ssr"],
                    "bls_score": stats["bls_score"],
                    "n_inside": stats["n_inside"],
                }

    return best