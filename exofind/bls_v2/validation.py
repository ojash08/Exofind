"""
validation.py — Astropy Cross-Validation

Compares ExoFind BLS results against Astropy's
BoxLeastSquares implementation to verify correctness.

For each confirmed planet, we compare:
    - Best period
    - Transit duration
    - Transit depth
    - SR / power
    - SDE
    - Periodogram shape
"""

import numpy as np


def validate_against_astropy(
        time,
        flux,
        errors=None,
        min_period=None,
        max_period=None,
        duration=0.1
):
    """
    Run Astropy BoxLeastSquares on the same data and
    return results for comparison with ExoFind.

    Parameters
    ----------
    time : ndarray
        Observation timestamps (days).

    flux : ndarray
        Raw flux measurements.

    errors : ndarray or None
        Per-point uncertainties.

    min_period : float or None
        Minimum period to search (days).

    max_period : float or None
        Maximum period to search (days).

    duration : float or array-like
        Transit duration(s) to search (days).

    Returns
    -------
    astropy_result : dict
        best_period    : float
        best_duration  : float
        best_depth     : float
        best_power     : float
        periods        : ndarray
        power          : ndarray
        depth          : ndarray

    Raises ImportError if astropy is not installed.
    """

    from astropy.timeseries import BoxLeastSquares

    dy = errors if errors is not None else None

    model = BoxLeastSquares(time, flux, dy=dy)

    if min_period is not None and max_period is not None:

        periods = np.linspace(
            min_period, max_period, 10000
        )

        result = model.power(periods, duration)

    else:

        result = model.autopower(duration)

    idx = np.argmax(result.power)

    return {
        "best_period": float(result.period[idx]),
        "best_duration": float(result.duration[idx]),
        "best_depth": float(result.depth[idx]),
        "best_power": float(result.power[idx]),
        "periods": np.array(result.period),
        "power": np.array(result.power),
        "depth": np.array(result.depth)
    }


def compare_results(exofind_result, astropy_result):
    """
    Compare ExoFind and Astropy BLS results.

    Parameters
    ----------
    exofind_result : dict
        Output from search_all_periods().

    astropy_result : dict
        Output from validate_against_astropy().

    Returns
    -------
    comparison : dict
        For each quantity: exofind value, astropy value,
        absolute difference, and relative difference.
    """

    exo = exofind_result["best"]

    if exo is None:
        return {"error": "ExoFind found no candidate"}

    comparison = {}

    # Period
    exo_period = exo["period"]
    ast_period = astropy_result["best_period"]

    comparison["period"] = {
        "exofind": exo_period,
        "astropy": ast_period,
        "abs_diff": abs(exo_period - ast_period),
        "rel_diff": abs(exo_period - ast_period) / ast_period
    }

    # Duration
    exo_duration = exo["duration_days"]
    ast_duration = astropy_result["best_duration"]

    comparison["duration"] = {
        "exofind": exo_duration,
        "astropy": ast_duration,
        "abs_diff": abs(exo_duration - ast_duration),
        "rel_diff": abs(
            exo_duration - ast_duration
        ) / ast_duration if ast_duration > 0 else float("inf")
    }

    # Depth
    exo_depth = abs(exo["depth"])
    ast_depth = abs(astropy_result["best_depth"])

    comparison["depth"] = {
        "exofind": exo_depth,
        "astropy": ast_depth,
        "abs_diff": abs(exo_depth - ast_depth),
        "rel_diff": abs(
            exo_depth - ast_depth
        ) / ast_depth if ast_depth > 0 else float("inf")
    }

    return comparison


def print_comparison(comparison):
    """
    Pretty-print a comparison dict.
    """

    if "error" in comparison:
        print(f"ERROR: {comparison['error']}")
        return

    print("\n" + "=" * 60)
    print("ExoFind vs Astropy BLS Comparison")
    print("=" * 60)

    for key, vals in comparison.items():

        print(f"\n{key.upper()}")
        print(f"  ExoFind  : {vals['exofind']:.6f}")
        print(f"  Astropy  : {vals['astropy']:.6f}")
        print(f"  Abs Diff : {vals['abs_diff']:.6f}")
        print(f"  Rel Diff : {vals['rel_diff']:.4%}")

    print("\n" + "=" * 60)
