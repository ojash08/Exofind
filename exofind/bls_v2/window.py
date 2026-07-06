"""
window.py — Sliding Transit Window

Equations [F1]–[F3] from Kovács, Zucker & Mazeh (2002)

Slides a transit window across all phase bins to find
the (start, width) that maximizes the Signal Residue.

[F1] For each starting bin i1 and width w:
    i2 = i1 + w
    s = PF[i2] - PF[i1]
    r = PR[i2] - PR[i1]

[F2] Handle wrap-around (transit crossing phase 0):
    When i2 > n_bins, the transit wraps around.
    Since PF[n_bins] = 0 (total weighted flux = 0):
        s = PF[n_bins] - PF[i1] + PF[i2 - n_bins]
          = -PF[i1] + PF[i2 - n_bins]
    Similarly for r, using PR[n_bins] = 1.

[F3] Record the maximum SR and its parameters.
"""

import numpy as np
from bls_v2.sr import signal_residue, transit_parameters


def slide_window(PF, PR, n_bins, min_width, max_width):
    """
    Slide a transit window across all phase bins
    and find the position that maximizes Signal Residue.

    Parameters
    ----------
    PF : ndarray, shape (n_bins + 1,)
        Prefix sum of weighted flux.

    PR : ndarray, shape (n_bins + 1,)
        Prefix sum of weights.

    n_bins : int
        Number of phase bins.

    min_width : int
        Minimum transit width in bins.

    max_width : int
        Maximum transit width in bins.

    Returns
    -------
    best : dict or None
        Best transit candidate with keys:
            start       : starting bin index
            width       : transit width in bins
            sr          : Signal Residue
            depth       : transit depth
            L           : in-transit level
            H           : out-of-transit level
            s           : in-transit weighted flux sum
            r           : in-transit weight sum
            phase_start : phase of transit start
            phase_width : fractional transit duration

        Returns None if no valid transit window found.
    """

    # To handle wrap-around efficiently, we extend the prefix sums
    # over two full cycles (2 * n_bins).
    
    # Extract the underlying bin values from the prefix sums
    s_bins = np.diff(PF)
    r_bins = np.diff(PR)
    
    # Extend to two cycles
    s_ext = np.concatenate((s_bins, s_bins))
    r_ext = np.concatenate((r_bins, r_bins))
    
    # Build extended prefix sums
    PF_ext = np.concatenate(([0.0], np.cumsum(s_ext)))
    PR_ext = np.concatenate(([0.0], np.cumsum(r_ext)))
    
    best_sr = -np.inf
    best = None
    
    # Vectorized loop over widths
    for width in range(min_width, max_width + 1):
        # We can evaluate all start positions simultaneously!
        starts = np.arange(n_bins)
        ends = starts + width
        
        # O(1) vectorized prefix sum query for all windows of this width
        s = PF_ext[ends] - PF_ext[starts]
        r = PR_ext[ends] - PR_ext[starts]
        
        # Find valid windows (avoid divide by zero)
        valid = (r > 0.0) & (r < 1.0)
        if not np.any(valid):
            continue
            
        # Calculate SR for all valid windows simultaneously!
        sr = np.zeros(n_bins)
        s_val = s[valid]
        r_val = r[valid]
        sr[valid] = (s_val * s_val) / (r_val * (1.0 - r_val))
        
        # Find the max SR for this width
        max_idx = np.argmax(sr)
        max_sr = sr[max_idx]
        
        if max_sr > best_sr:
            best_sr = max_sr
            best_s = s[max_idx]
            best_r = r[max_idx]
            
            params = transit_parameters(best_s, best_r)
            if params is not None:
                best = {
                    "start": int(max_idx),
                    "width": int(width),
                    "phase_start": max_idx / n_bins,
                    "phase_width": width / n_bins,
                    **params
                }
                
    return best
