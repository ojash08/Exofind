"""
vetting/secondary_eclipse.py — Secondary Eclipse Search

Eclipsing binaries have a primary and secondary eclipse.
Planets (usually) only have a primary transit visible.
If we detect a deep secondary eclipse at phase ~0.5 (opposite
the primary transit), the candidate is likely a false positive.
"""

from vetting.odd_even import _extract_depth_at_phase
from bls_v2.weights import normalize_weights

def check_secondary_eclipse(time, flux, period, phase_start, phase_width, primary_depth, n_bins=400):
    """
    Look for a secondary eclipse at phase + 0.5.
    
    Returns a dictionary with the secondary depth and false positive flag.
    """
    
    x_tilde, w_tilde = normalize_weights(flux)
    
    # Phase 0.5 away from primary
    sec_phase_start = (phase_start + 0.5) % 1.0
    
    # Force a fit at the secondary phase
    sec_depth = _extract_depth_at_phase(
        time, x_tilde, w_tilde, period, sec_phase_start, phase_width, n_bins
    )
    
    if sec_depth is None:
        return {"error": "Degenerate transit fit during secondary search"}
        
    # If the secondary eclipse is > 10% of the primary depth, it's suspicious.
    # Note: Hot Jupiters *can* have secondary eclipses, but they are usually < 1% of primary.
    ratio = sec_depth / primary_depth
    
    return {
        "secondary_depth": sec_depth,
        "ratio_to_primary": ratio,
        "is_false_positive": ratio > 0.1
    }
