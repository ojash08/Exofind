import numpy as np
from bls_v2.search import search_all_periods
from photometry.detrend import detrend

def iterative_transit_masking(time, raw_flux, window_length=151):
    """
    Iterative Transit Masking Architecture.
    
    Prevents 'transit attenuation' (signal eating) by:
    1. Running a rough detrend and BLS search to find the transit.
    2. Masking out the transit data entirely.
    3. Recalculating the pristine trendline using ONLY out-of-transit data.
    4. Interpolating the pristine trendline straight across the transit gap.
    5. Flattening the raw flux against this unattenuated trend.
    """
    
    # 1. Rough Detrend
    rough_flat, rough_trend = detrend(time, raw_flux, window_length=window_length)
    
    # 2. Rough BLS Search (Fast mode)
    baseline = np.max(time) - np.min(time)
    bls_res = search_all_periods(
        time, rough_flat,
        n_bins=200,                # Coarse bins for speed
        min_period=0.5,
        max_period=baseline / 3.0,
        min_duration=0.05,
        max_duration=0.3
    )
    
    best = bls_res["best"]
    
    # If no real transit found, return the rough detrend
    if best is None or best["sr"] < 1e-6:
        return rough_flat, rough_trend
        
    period = best["period"]
    phase_start = best["phase_start"]
    phase_width = best["phase_width"]
    
    # Pad the mask slightly to ensure the transit wings aren't eaten
    padding = 0.01 
    p_start = phase_start - padding
    p_end = phase_start + phase_width + padding
    
    # 3. Generate In-Transit Mask
    phase = np.mod(time, period) / period
    
    # Handle phase wrap-around
    if p_end > 1.0:
        in_transit = (phase >= p_start) | (phase < (p_end - 1.0))
    elif p_start < 0.0:
        in_transit = (phase >= (p_start + 1.0)) | (phase < p_end)
    else:
        in_transit = (phase >= p_start) & (phase < p_end)
        
    # We want to use OUT-OF-TRANSIT data to build the true trend
    out_of_transit = ~in_transit
    
    # If the BLS search went wild and masked the whole array, fallback
    if np.sum(out_of_transit) < window_length:
        return rough_flat, rough_trend
        
    # 4. Recalculate Pristine Trend
    safe_time = time[out_of_transit]
    safe_flux = raw_flux[out_of_transit]
    
    # Piecewise detrend the safe flux to get a smooth, unattenuated baseline
    _, safe_trend = detrend(safe_time, safe_flux, window_length=window_length)
    
    # Interpolate this safe trend perfectly straight across the transit gaps
    pristine_trend = np.interp(time, safe_time, safe_trend)
    
    # 5. Final Flattening
    final_flat = raw_flux / pristine_trend
    
    return final_flat, pristine_trend
