"""
confidence.py — False Alarm Probability

Equations [K1], [K2] from Kovács, Zucker & Mazeh (2002)

[K1] Analytical estimate:
    FAP ≈ 1 - (1 - erfc(SDE / sqrt(2)))^N_eff

    where N_eff is the effective number of independent
    frequencies.

[K2] Bootstrap method:
    1. Shuffle flux values (keep timestamps fixed)
    2. Run full BLS on each shuffled dataset
    3. FAP = fraction with SR_max >= SR_observed
"""

import numpy as np
from math import erfc


def false_alarm_probability_analytical(sde_peak, n_eff):
    """
    Analytical FAP estimate from the SDE peak value.

    Based on the extreme-value distribution of the
    maximum of N_eff independent Gaussian variables.

    Parameters
    ----------
    sde_peak : float
        SDE value of the detected peak.

    n_eff : float
        Effective number of independent frequencies
        in the periodogram. A reasonable estimate is
        the number of trial periods, though the true
        value may be lower due to correlations.

    Returns
    -------
    fap : float
        False Alarm Probability in [0, 1].
    """

    # Probability of one random trial exceeding sde_peak
    p_single = erfc(sde_peak / np.sqrt(2.0))

    # Probability that at least one of N_eff trials exceeds
    fap = 1.0 - (1.0 - p_single) ** n_eff

    # Clamp to [0, 1]
    fap = np.clip(fap, 0.0, 1.0)

    return float(fap)


def false_alarm_probability_bootstrap(
        time,
        flux,
        errors,
        observed_sr,
        n_bootstrap=1000,
        n_bins=400,
        min_period=None,
        max_period=None,
        min_duration=None,
        max_duration=None
):
    """
    Bootstrap FAP by shuffling flux values.

    This preserves the time sampling but destroys any
    periodic signal. The fraction of shuffled datasets
    that produce SR_max >= observed_sr gives the FAP.

    Parameters
    ----------
    time : ndarray
        Observation timestamps.

    flux : ndarray
        Original flux values (NOT mean-subtracted).

    errors : ndarray or None
        Measurement uncertainties.

    observed_sr : float
        The SR_max from the real data.

    n_bootstrap : int
        Number of bootstrap iterations.

    n_bins : int
        Number of phase bins.

    min_period, max_period : float or None
        Period range for the bootstrap search.

    min_duration, max_duration : float or None
        Duration range for the bootstrap search.

    Returns
    -------
    fap : float
        Bootstrap False Alarm Probability.

    sr_distribution : ndarray
        SR_max from each bootstrap iteration.
    """

    # Import here to avoid circular dependency
    from bls_v2.search import search_all_periods

    rng = np.random.default_rng()

    count_exceeding = 0
    sr_distribution = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):

        # Shuffle flux (keep time fixed)
        shuffled_flux = rng.permutation(flux)

        result = search_all_periods(
            time,
            shuffled_flux,
            errors=errors,
            n_bins=n_bins,
            min_period=min_period,
            max_period=max_period,
            min_duration=min_duration,
            max_duration=max_duration
        )

        if result["best"] is not None:
            sr_max = result["best"]["sr"]
        else:
            sr_max = 0.0

        sr_distribution[i] = sr_max

        if sr_max >= observed_sr:
            count_exceeding += 1

    fap = count_exceeding / n_bootstrap

    return fap, sr_distribution
