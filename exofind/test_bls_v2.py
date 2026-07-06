"""
test_bls_v2.py — Validation tests for Scientific BLS

Tests every module against the mathematical invariants
from Kovács, Zucker & Mazeh (2002).

Invariants checked:
    - sum(w_tilde) = 1
    - sum(w_tilde * x_tilde) = 0
    - PF[n_bins] ≈ 0
    - PR[n_bins] ≈ 1
    - SR >= 0 always
    - depth > 0 for a valid transit
    - Correct period recovery on synthetic data
"""

import numpy as np
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_weights():
    """Test data normalization [A1, A2]."""

    from bls_v2.weights import normalize_weights

    np.random.seed(42)
    flux = np.random.normal(1.0, 0.01, 1000)

    # --- Uniform weights ---
    x_tilde, w_tilde = normalize_weights(flux)

    assert abs(np.sum(w_tilde) - 1.0) < 1e-12, \
        f"Weights don't sum to 1: {np.sum(w_tilde)}"

    assert abs(np.sum(w_tilde * x_tilde)) < 1e-12, \
        f"Weighted mean not zero: {np.sum(w_tilde * x_tilde)}"

    # --- Non-uniform weights ---
    errors = np.random.uniform(0.005, 0.02, 1000)
    x_tilde, w_tilde = normalize_weights(flux, errors)

    assert abs(np.sum(w_tilde) - 1.0) < 1e-12, \
        f"Weighted sum != 1: {np.sum(w_tilde)}"

    assert abs(np.sum(w_tilde * x_tilde)) < 1e-12, \
        f"Weighted mean != 0: {np.sum(w_tilde * x_tilde)}"

    print("[PASS] weights.py — invariants verified")


def test_fold():
    """Test phase folding [B1, B2]."""

    from bls_v2.fold import fold_lightcurve

    time = np.linspace(0, 20, 500)
    flux = np.ones(500)
    weights = np.ones(500) / 500.0
    period = 3.0

    phase, f, w = fold_lightcurve(time, flux, weights, period)

    # Phase must be in [0, 1)
    assert np.all(phase >= 0.0), "Phase < 0 found"
    assert np.all(phase < 1.0), "Phase >= 1 found"

    # Phase must be sorted
    assert np.all(np.diff(phase) >= 0), "Phase not sorted"

    # Weights must be preserved
    assert abs(np.sum(w) - 1.0) < 1e-12, \
        "Weight sum changed during fold"

    print("[PASS] fold.py — phase in [0,1), sorted, weights preserved")


def test_bins():
    """Test weighted binning [C1, C2]."""

    from bls_v2.weights import normalize_weights
    from bls_v2.fold import fold_lightcurve
    from bls_v2.bins import bin_folded

    np.random.seed(42)
    time = np.linspace(0, 20, 1000)
    flux = np.random.normal(1.0, 0.01, 1000)
    x_tilde, w_tilde = normalize_weights(flux)

    phase, f, w = fold_lightcurve(time, x_tilde, w_tilde, 3.0)

    S, R = bin_folded(phase, f, w, n_bins=400)

    # Sum of S should be ~0 (weighted mean was subtracted)
    assert abs(np.sum(S)) < 1e-10, \
        f"sum(S) != 0: {np.sum(S)}"

    # Sum of R should be ~1 (weights are normalized)
    assert abs(np.sum(R) - 1.0) < 1e-10, \
        f"sum(R) != 1: {np.sum(R)}"

    print("[PASS] bins.py — sum(S)=0, sum(R)=1")


def test_prefix_sums():
    """Test prefix sums [D1, D2]."""

    from bls_v2.weights import normalize_weights
    from bls_v2.fold import fold_lightcurve
    from bls_v2.bins import bin_folded
    from bls_v2.prefix_sum import build_prefix_sums

    np.random.seed(42)
    time = np.linspace(0, 20, 1000)
    flux = np.random.normal(1.0, 0.01, 1000)
    x_tilde, w_tilde = normalize_weights(flux)

    phase, f, w = fold_lightcurve(time, x_tilde, w_tilde, 3.0)
    S, R = bin_folded(phase, f, w, n_bins=400)
    PF, PR = build_prefix_sums(S, R)

    # PF[0] = 0
    assert PF[0] == 0.0, f"PF[0] != 0: {PF[0]}"

    # PR[0] = 0
    assert PR[0] == 0.0, f"PR[0] != 0: {PR[0]}"

    # PF[-1] should be ~0 (total weighted flux = 0)
    assert abs(PF[-1]) < 1e-10, \
        f"PF[-1] != 0: {PF[-1]}"

    # PR[-1] should be ~1 (total weight = 1)
    assert abs(PR[-1] - 1.0) < 1e-10, \
        f"PR[-1] != 1: {PR[-1]}"

    # Length should be n_bins + 1
    assert len(PF) == 401, f"PF length wrong: {len(PF)}"
    assert len(PR) == 401, f"PR length wrong: {len(PR)}"

    print("[PASS] prefix_sum.py — PF[-1]=0, PR[-1]=1, correct length")


def test_sr():
    """Test Signal Residue [E1–E4]."""

    from bls_v2.sr import signal_residue, transit_parameters

    # Known values
    s = -0.05   # transit dip (negative)
    r = 0.1     # 10% of weight in transit

    sr = signal_residue(s, r)

    # SR = s^2 / (r * (1-r)) = 0.0025 / 0.09 = 0.02778
    expected = (0.05 ** 2) / (0.1 * 0.9)
    assert abs(sr - expected) < 1e-10, \
        f"SR wrong: {sr} != {expected}"

    # SR must be non-negative
    assert sr >= 0, f"SR negative: {sr}"

    # Transit parameters
    params = transit_parameters(s, r)
    assert params is not None

    # L = s/r = -0.5
    assert abs(params["L"] - (-0.5)) < 1e-10

    # H = -s/(1-r) = 0.0556
    assert abs(params["H"] - (0.05 / 0.9)) < 1e-10

    # depth = H - L > 0
    assert params["depth"] > 0, f"Depth not positive: {params['depth']}"

    # Degenerate cases
    assert signal_residue(0.1, 0.0) == 0.0
    assert signal_residue(0.1, 1.0) == 0.0
    assert transit_parameters(0.1, 0.0) is None

    print("[PASS] sr.py — SR formula correct, depth > 0, degenerates handled")


def test_synthetic_transit():
    """
    End-to-end test: inject a known transit and
    verify that the BLS recovers the correct period.
    """

    from bls_v2.search import search_all_periods
    from bls_v2.sde import compute_sde

    np.random.seed(42)

    # Generate synthetic data
    n_points = 2000
    time = np.sort(np.random.uniform(0, 30, n_points))

    true_period = 5.0       # days
    true_duration = 0.15    # days
    true_depth = 0.01       # 1% dip
    noise_level = 0.002

    # Baseline flux = 1.0
    flux = np.ones(n_points)

    # Inject transits
    phase = np.mod(time, true_period) / true_period
    in_transit = phase < (true_duration / true_period)
    flux[in_transit] -= true_depth

    # Add noise
    flux += np.random.normal(0, noise_level, n_points)

    # Run BLS search
    result = search_all_periods(
        time, flux,
        min_period=2.0,
        max_period=10.0,
        min_duration=0.05,
        max_duration=0.3,
        n_bins=300
    )

    best = result["best"]

    assert best is not None, "BLS found no candidate!"

    # Check period recovery
    period_error = abs(best["period"] - true_period)
    assert period_error < 0.1, \
        f"Period wrong: {best['period']:.4f} vs {true_period:.4f}"

    # Compute SDE
    sde, mean_sr, std_sr = compute_sde(result["sr_array"])
    peak_sde = sde[np.argmax(result["sr_array"])]

    assert peak_sde > 6.0, \
        f"SDE too low: {peak_sde:.2f} (need > 6)"

    print(f"[PASS] Synthetic transit test")
    print(f"       True period  : {true_period:.4f} d")
    print(f"       Found period : {best['period']:.4f} d")
    print(f"       Error        : {period_error:.4f} d")
    print(f"       Depth        : {abs(best['depth']):.6f}")
    print(f"       SDE          : {peak_sde:.2f}")


def test_sde():
    """Test SDE computation [I1, I2]."""

    from bls_v2.sde import compute_sde

    # Uniform SR → SDE should be all zero
    sr_uniform = np.ones(100) * 0.5
    sde, mean_sr, std_sr = compute_sde(sr_uniform)

    assert np.all(sde == 0.0), "Uniform SR should give SDE=0"
    assert mean_sr == 0.5

    # One outlier
    sr = np.zeros(100)
    sr[50] = 10.0

    sde, mean_sr, std_sr = compute_sde(sr)

    # The peak should have the highest SDE
    assert np.argmax(sde) == 50
    assert sde[50] > 0

    print("[PASS] sde.py — correct normalization")


def test_peaks():
    """Test peak detection [J1–J3]."""

    from bls_v2.peaks import find_peaks

    periods = np.linspace(1, 10, 1000)
    sde = np.zeros(1000)

    # Inject a clear peak at period ~5
    idx = 444  # ~5 days
    sde[idx] = 10.0
    sde[idx - 1] = 5.0
    sde[idx + 1] = 5.0

    # Inject a harmonic at period ~2.5
    idx_h = 167  # ~2.5 days
    sde[idx_h] = 7.0
    sde[idx_h - 1] = 3.0
    sde[idx_h + 1] = 3.0

    candidates = find_peaks(periods, sde, threshold=6.0)

    assert len(candidates) >= 1, "No peaks found"
    assert abs(candidates[0]["period"] - 5.0) < 0.1, \
        f"Wrong peak period: {candidates[0]['period']}"

    print("[PASS] peaks.py — correct peak at ~5 days")


def test_confidence_analytical():
    """Test analytical FAP [K1]."""

    from bls_v2.confidence import false_alarm_probability_analytical

    # Very high SDE → very low FAP
    fap_high = false_alarm_probability_analytical(10.0, 1000)
    assert fap_high < 0.01, f"FAP too high for SDE=10: {fap_high}"

    # Very low SDE → high FAP
    fap_low = false_alarm_probability_analytical(1.0, 1000)
    assert fap_low > 0.5, f"FAP too low for SDE=1: {fap_low}"

    print("[PASS] confidence.py — analytical FAP correct direction")


def run_all_tests():
    """Run all validation tests."""

    print("=" * 60)
    print("ExoFind BLS v2 — Scientific Validation")
    print("Kovács, Zucker & Mazeh (2002)")
    print("=" * 60)
    print()

    test_weights()
    test_fold()
    test_bins()
    test_prefix_sums()
    test_sr()
    test_sde()
    test_peaks()
    test_confidence_analytical()

    print()
    print("-" * 60)
    print("Running end-to-end synthetic transit test...")
    print("-" * 60)

    test_synthetic_transit()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
