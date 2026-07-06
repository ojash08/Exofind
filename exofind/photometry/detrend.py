from scipy.signal import savgol_filter
import numpy as np


def detrend(time, flux, window_length=151, polyorder=3, iters=2, sigma_lower=3.0):
    """
    Remove slow trends using a Piecewise Iterative Savitzky-Golay filter.
    Handles TESS mid-sector downlink gaps by splitting continuous segments.
    """
    # Find gaps larger than 0.5 days
    gaps = np.where(np.diff(time) > 0.5)[0] + 1
    flux_chunks = np.split(flux, gaps)
    
    flat_flux_all = []
    trend_all = []
    
    for f_chunk in flux_chunks:
        if len(f_chunk) == 0:
            continue
            
        wl = window_length
        # Window length must be odd
        if wl % 2 == 0:
            wl += 1

        # Window cannot exceed data length
        if wl >= len(f_chunk):
            wl = len(f_chunk) - 1
            if wl % 2 == 0:
                wl -= 1

        # Ensure window is large enough for polyorder
        if wl <= polyorder:
            wl = polyorder + 2
            if wl % 2 == 0:
                wl += 1

        mask = np.ones(len(f_chunk), dtype=bool)
        trend = np.ones(len(f_chunk))

        for _ in range(iters):
            valid_flux = f_chunk[mask]
            
            if len(valid_flux) < wl:
                valid_flux = f_chunk
                mask = np.ones(len(f_chunk), dtype=bool)

            trend_valid = savgol_filter(
                valid_flux,
                window_length=wl,
                polyorder=polyorder
            )

            trend[mask] = trend_valid
            
            if not np.all(mask):
                trend[~mask] = np.interp(
                    np.where(~mask)[0],
                    np.where(mask)[0],
                    trend_valid
                )

            residuals = f_chunk - trend
            std = np.std(residuals[mask])
            mask = residuals > (-sigma_lower * std)

        flattened = f_chunk / trend
        
        flat_flux_all.append(flattened)
        trend_all.append(trend)

    return np.concatenate(flat_flux_all), np.concatenate(trend_all)