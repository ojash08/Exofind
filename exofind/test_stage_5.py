"""
test_stage_5.py — Validation for ExoFind Stage 5 (Vetting)

Injects synthetic transit signals to test:
1. True Planet (passes vetting)
2. Eclipsing Binary with odd/even depth mismatch (rejected)
3. Eclipsing Binary with secondary eclipse (rejected)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.target import ExoFindTarget
from pipeline.stages.vetting import VettingStage

def create_synthetic_target(name, period, duration, primary_depth, secondary_depth=0.0, odd_even_ratio=1.0):
    
    target = ExoFindTarget("synthetic.fits")
    target.filename = name
    
    n_points = 5000
    time = np.sort(np.random.uniform(0, 30, n_points))
    flux = np.ones(n_points)
    
    phase = np.mod(time, period) / period
    dur_frac = duration / period
    
    # Primary Eclipse (Odd)
    in_primary_odd = (phase < dur_frac) & (np.floor(time / period) % 2 != 0)
    flux[in_primary_odd] -= primary_depth
    
    # Primary Eclipse (Even)
    in_primary_even = (phase < dur_frac) & (np.floor(time / period) % 2 == 0)
    flux[in_primary_even] -= (primary_depth * odd_even_ratio)
    
    # Secondary Eclipse
    in_secondary = (phase > 0.5) & (phase < 0.5 + dur_frac)
    flux[in_secondary] -= secondary_depth
    
    # Add noise
    flux += np.random.normal(0, 0.001, n_points)
    
    target.clean_time = time
    target.clean_flux = flux
    
    # Mock a BLS search candidate
    target.candidates = [{
        "period": period,
        "phase_start": 0.0,
        "phase_width": dur_frac,
        "depth": primary_depth,
        "sde": 15.0,
        "duration_days": duration,
        "is_harmonic": False
    }]
    
    return target


def run_vetting_test():
    print("=" * 60)
    print("ExoFind Stage 5 — Vetting Tests")
    print("=" * 60)
    
    vetting = VettingStage()
    
    # --- Test 1: True Planet ---
    print("\n--- Test 1: True Planet ---")
    planet = create_synthetic_target(
        "True Planet", period=4.0, duration=0.1, primary_depth=0.02
    )
    vetting.execute(planet)
    print(f"Status: {planet.candidates[0]['status']}")
    assert planet.candidates[0]['status'] == "PASSED"
    
    # --- Test 2: Odd/Even Mismatch (Eclipsing Binary) ---
    print("\n--- Test 2: Odd/Even Mismatch (Eclipsing Binary) ---")
    binary_oe = create_synthetic_target(
        "Odd/Even Binary", period=4.0, duration=0.1, primary_depth=0.02, odd_even_ratio=0.5
    )
    vetting.execute(binary_oe)
    print(f"Status: {binary_oe.candidates[0]['status']}")
    assert "Odd/Even Mismatch" in binary_oe.candidates[0]['status']
    
    # --- Test 3: Secondary Eclipse (Eclipsing Binary) ---
    print("\n--- Test 3: Secondary Eclipse (Eclipsing Binary) ---")
    binary_se = create_synthetic_target(
        "Secondary Eclipse Binary", period=4.0, duration=0.1, primary_depth=0.02, secondary_depth=0.01
    )
    vetting.execute(binary_se)
    print(f"Status: {binary_se.candidates[0]['status']}")
    assert "Secondary Eclipse" in binary_se.candidates[0]['status']
    
    print("\n" + "=" * 60)
    print("ALL STAGE 5 VETTING TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_vetting_test()
