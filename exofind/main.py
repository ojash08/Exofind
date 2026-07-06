"""
ExoFind — Exoplanet Detection Pipeline
=======================================

Custom Photometry → Scientific BLS → Detection Report

BLS implementation: Kovács, Zucker & Mazeh (2002)
    "A box-fitting algorithm in the search for periodic transits"
    A&A 391, 369–377

Target: WASP-18b (TIC 100100827)
    Published period: 0.94145 days
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

# ==========================================
# Photometry (untouched)
# ==========================================

from photometry.loader import load_tpf
from photometry.star_locator import average_image
from photometry.star_locator import brightest_pixel
from photometry.optimizer import optimize_aperture
from photometry.quality_flags import filter_quality
from photometry.detrend import detrend
from photometry.outliers import remove_outliers
from photometry.background import (
    estimate_global_background,
    estimate_local_background
)

# ==========================================
# Scientific BLS v2
# ==========================================

from bls_v2.search import search_all_periods
from bls_v2.sde import compute_sde
from bls_v2.peaks import find_peaks
from bls_v2.confidence import false_alarm_probability_analytical
from bls_v2.fold import fold_lightcurve
from bls_v2.weights import normalize_weights


# ==============================================================
# STAGE 1: Custom Photometry
# ==============================================================

print("=" * 65)
print("  ExoFind — Exoplanet Detection Pipeline")
print("  BLS: Kovacs, Zucker & Mazeh (2002)")
print("=" * 65)

parser = argparse.ArgumentParser(description="ExoFind — Exoplanet Detection Pipeline")
parser.add_argument("tpf_path", help="Path to the TESS Target Pixel File (.fits)")
args = parser.parse_args()

FILE = args.tpf_path
tpf = load_tpf(FILE)

# Extract target name from header
header = tpf.hdu[0].header
target_name = header.get("OBJECT", "Unknown Target")
tic_id = header.get("TICID", "Unknown TIC")
target_display = f"{target_name} (TIC {tic_id})"

print("\n--- STAGE 1: Custom Photometry ---\n")
print(f"Target       : {target_display}")
print(f"Frames       : {len(tpf.time)}")
print(f"Flux Shape   : {tpf.flux.shape}")

avg = average_image(tpf)
row, col = brightest_pixel(avg)

global_background = estimate_global_background(avg)
local_background = estimate_local_background(avg, (row, col))

print(f"Global Bkg   : {global_background:.2f}")
print(f"Local Bkg    : {local_background:.2f}")

background = local_background

mask, best_growth_order, time, flux, score, results = optimize_aperture(
    tpf, avg, background, (row, col)
)

time, flux, quality = filter_quality(time, flux, tpf.quality)

nan_mask = ~np.isnan(time) & ~np.isnan(flux)
time = time[nan_mask]
flux = flux[nan_mask]

# Use a window of ~5 hours (151 cadences at 2-min) to track spots
flat_flux, trend = detrend(time, flux, window_length=151)
time_clean, flux_clean = remove_outliers(time, flat_flux)

print(f"\nAperture Optimization:")
print(f"  Best Score : {score:.2f}")
print(f"  Cadences   : {len(time_clean)} (after quality + outlier filtering)")

baseline = np.max(time_clean) - np.min(time_clean)
print(f"  Baseline   : {baseline:.2f} days")

print("\n  Photometry COMPLETE")


# ==============================================================
# STAGE 2: Scientific BLS Search
# ==============================================================

print("\n--- STAGE 2: Scientific BLS (KZM 2002) ---\n")

# Search parameters
MIN_PERIOD = 0.5          # days
MAX_PERIOD = baseline / 3  # at least 3 transits
MIN_DURATION = 0.05       # ~1.2 hours
MAX_DURATION = 0.3        # ~7.2 hours
N_BINS = 400

print(f"Period range  : {MIN_PERIOD:.2f} – {MAX_PERIOD:.2f} days")
print(f"Duration range: {MIN_DURATION:.2f} – {MAX_DURATION:.2f} days")
print(f"Phase bins    : {N_BINS}")

print("\nSearching... ", end="", flush=True)

bls_result = search_all_periods(
    time_clean,
    flux_clean,
    errors=None,             # Uniform weights
    n_bins=N_BINS,
    min_period=MIN_PERIOD,
    max_period=MAX_PERIOD,
    min_duration=MIN_DURATION,
    max_duration=MAX_DURATION,
    min_n_transits=3,
    frequency_factor=1.0
)

n_periods = len(bls_result["periods"])
print(f"DONE ({n_periods} trial periods)")


# ==============================================================
# STAGE 3: Signal Detection Efficiency
# ==============================================================

print("\n--- STAGE 3: Signal Detection Efficiency ---\n")

sde, mean_sr, std_sr = compute_sde(bls_result["sr_array"])

peak_idx = np.argmax(sde)
peak_sde = sde[peak_idx]
peak_period = bls_result["periods"][peak_idx]

print(f"SR mean      : {mean_sr:.8f}")
print(f"SR std       : {std_sr:.8f}")
print(f"Peak SDE     : {peak_sde:.2f}")
print(f"Peak period  : {peak_period:.4f} days")
print(f"Threshold    : 6.0 (KZM 2002)")
print(f"Detection    : {'YES' if peak_sde > 6.0 else 'NO'}")


# ==============================================================
# STAGE 4: Peak Detection & Candidate Ranking
# ==============================================================

print("\n--- STAGE 4: Peak Detection ---\n")

candidates = find_peaks(
    bls_result["periods"],
    sde,
    threshold=5.0
)

if len(candidates) == 0:
    print("No significant peaks found.")
else:
    print(f"Found {len(candidates)} significant peak(s)\n")

    print(f"{'Rank':<6}{'Period (d)':<14}{'SDE':<10}{'Harmonic':<10}")
    print("-" * 40)

    for i, c in enumerate(candidates[:10]):
        print(
            f"{i+1:<6}"
            f"{c['period']:<14.4f}"
            f"{c['sde']:<10.2f}"
            f"{'yes' if c['is_harmonic'] else '':<10}"
        )


# ==============================================================
# STAGE 5: Best Candidate Analysis
# ==============================================================

print("\n--- STAGE 5: Best Candidate ---\n")

best = bls_result["best"]

if best is None:
    print("No candidate found.")
else:
    # False alarm probability
    fap = false_alarm_probability_analytical(peak_sde, n_periods)

    print(f"Period        : {best['period']:.5f} days")
    print(f"Duration      : {best['duration_days']:.4f} days ({best['duration_days']*24:.1f} hours)")
    print(f"Depth         : {abs(best['depth']):.8f}")
    print(f"SR            : {best['sr']:.8f}")
    print(f"SDE           : {peak_sde:.2f}")
    print(f"FAP           : {fap:.2e}")
    print(f"In-transit    : {best['L']:.8f}")
    print(f"Out-of-transit: {best['H']:.8f}")
    print(f"Phase start   : {best['phase_start']:.4f}")
    print(f"Phase width   : {best['phase_width']:.4f}")


# ==============================================================
# STAGE 6: Plots
# ==============================================================

print("\n--- STAGE 6: Generating Plots ---\n")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# --- Plot 1: SDE Periodogram ---
ax1 = axes[0]
ax1.plot(
    bls_result["periods"], sde,
    linewidth=0.8, color="#2196F3", alpha=0.8
)
ax1.axhline(6.0, color="red", linestyle="--", alpha=0.5, label="SDE = 6 threshold")

if best is not None:
    ax1.axvline(
        best["period"], color="red", linestyle="-",
        alpha=0.7, label=f"Best: {best['period']:.4f} d"
    )

ax1.set_xlabel("Period (days)")
ax1.set_ylabel("SDE")
ax1.set_title("ExoFind BLS v2 — Signal Detection Efficiency Periodogram")
ax1.legend()
ax1.grid(alpha=0.3)

# --- Plot 2: SR Periodogram ---
ax2 = axes[1]
ax2.plot(
    bls_result["periods"], bls_result["sr_array"],
    linewidth=0.8, color="#FF9800", alpha=0.8
)

if best is not None:
    ax2.scatter(
        [best["period"]], [best["sr"]],
        color="red", s=80, zorder=5,
        label=f"Best SR: {best['sr']:.6f}"
    )

ax2.set_xlabel("Period (days)")
ax2.set_ylabel("Signal Residue (SR)")
ax2.set_title("ExoFind BLS v2 — Signal Residue Periodogram")
ax2.legend()
ax2.grid(alpha=0.3)

# --- Plot 3: Folded Light Curve at Best Period ---
if best is not None:

    x_tilde, w_tilde = normalize_weights(flux_clean)
    phase, folded_flux, folded_weights = fold_lightcurve(
        time_clean, x_tilde, w_tilde, best["period"]
    )

    # Add the mean back for plotting
    mean_flux = np.mean(flux_clean)
    display_flux = folded_flux + mean_flux

    ax3 = axes[2]
    ax3.scatter(
        phase, display_flux,
        s=1, alpha=0.3, color="#607D8B", label="Unbinned Data"
    )

    # --- Dynamic Phase Binning ---
    from scipy.stats import binned_statistic
    
    # To perfectly resolve the U-shape, exactly 3 bins should fall inside the transit
    phase_duration = best["duration_days"] / best["period"]
    optimal_bins = int(3 / phase_duration)
    
    # Clamp between 50 and 500 for safety against anomalous BLS durations
    n_plot_bins = max(50, min(500, optimal_bins))
    
    # Use binned_statistic for brutally fast, purely functional binning
    binned_flux, bin_edges, _ = binned_statistic(
        phase, display_flux, statistic='median', bins=n_plot_bins
    )
    binned_phase = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
    ax3.plot(
        binned_phase, binned_flux,
        'o', markersize=4, color="#E91E63", 
        markeredgecolor="black", markeredgewidth=0.5,
        label=f"Binned Median ({n_plot_bins} bins)", zorder=10
    )

    # Mark the transit window
    transit_start = best["phase_start"]
    transit_end = transit_start + best["phase_width"]

    ax3.axvspan(
        transit_start, min(transit_end, 1.0),
        alpha=0.15, color="red",
        label=f"Transit ({best['duration_days']*24:.1f} h)"
    )

    # If wrap-around
    if transit_end > 1.0:
        ax3.axvspan(0, transit_end - 1.0, alpha=0.15, color="red")

    ax3.set_xlabel("Phase")
    ax3.set_ylabel("Flux")
    ax3.set_title(
        f"Folded Light Curve — P = {best['period']:.4f} d, "
        f"depth = {abs(best['depth']):.6f}"
    )
    ax3.legend()
    ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("exofind_bls_v2_results.png", dpi=150, bbox_inches="tight")
print("Saved: exofind_bls_v2_results.png")


# ==============================================================
# SUMMARY
# ==============================================================

print("\n" + "=" * 65)
print("  ExoFind Detection Summary")
print("=" * 65)

if best is not None:
    print(f"""
  Target         : {target_display}
  Cadences       : {len(time_clean)}
  Baseline       : {baseline:.2f} days
  Periods tested : {n_periods}

  DETECTION:
    Period       : {best['period']:.5f} days
    Duration     : {best['duration_days']:.4f} days
    Depth        : {abs(best['depth']):.8f}
    SDE          : {peak_sde:.2f}
    FAP          : {fap:.2e}
    SR           : {best['sr']:.8f}
""")
else:
    print("\n  No transit detected.\n")

print("=" * 65)