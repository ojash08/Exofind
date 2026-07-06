"""
pipeline/stages/search.py — Stage 4

Handles the Transit Search via the BLS v2 algorithm.
Generates the periodogram, calculates SDE, and finds initial peaks.
"""

import numpy as np
from pipeline.stage import PipelineStage
from bls_v2.sde import compute_sde
from bls_v2.peaks import find_peaks
from astropy.timeseries import BoxLeastSquares

class BLSSearchStage(PipelineStage):

    def __init__(self, min_period=0.5, max_period_frac=1/3, n_bins=400):
        # We keep n_bins parameter for interface compatibility but Astropy handles binning internally
        self.min_period = min_period
        self.max_period_frac = max_period_frac
        self.n_bins = n_bins

    def execute(self, target):
        
        if target.clean_time is None or target.clean_flux is None:
            raise ValueError("BLSSearchStage requires clean_time and clean_flux.")
            
        max_period = target.baseline * self.max_period_frac
        
        print(f"  Searching periods from {self.min_period:.2f} to {max_period:.2f} days (Astropy Backend)...")
        
        # --- 1. Run Astropy BLS ---
        model = BoxLeastSquares(target.clean_time, target.clean_flux)
        
        # Search over a range of typical planetary transit durations (1.2 to 7.2 hours)
        durations = np.linspace(0.05, 0.3, 30)
        
        # Generate optimal frequency grid
        periods = model.autoperiod(durations, minimum_period=self.min_period, maximum_period=max_period)
        
        # Run C-optimized search using 'snr' objective to match KZM 2002 Signal Residue
        results = model.power(periods, durations, objective='snr')
        
        # Package results to match ExoFind internal format
        bls_result = {
            "periods": np.array(results.period),
            "sr_array": np.array(results.power),
        }
        target.bls_results = bls_result
        
        # --- 2. Compute SDE ---
        print("  Computing Signal Detection Efficiency (SDE)...")
        sde, mean_sr, std_sr = compute_sde(bls_result["sr_array"])
        
        bls_result["sde_array"] = sde
        bls_result["mean_sr"] = mean_sr
        bls_result["std_sr"] = std_sr
        
        # --- 3. Find Peaks ---
        print("  Identifying peaks...")
        candidates = find_peaks(
            bls_result["periods"],
            sde,
            threshold=3.0  # Lowered so true planets survive Astropy's unpenalized long-period noise
        )
        
        # Merge best fit params from astropy into candidate dicts
        for cand in candidates:
            idx = cand["index"]
            
            period = results.period[idx]
            duration_days = results.duration[idx]
            transit_time = results.transit_time[idx]
            depth = results.depth[idx]
            
            # Astropy transit_time is mid-transit. Convert to start phase for vetting
            transit_start_time = transit_time - (duration_days / 2.0)
            phase_start = (transit_start_time % period) / period
            phase_width = duration_days / period
            
            cand.update({
                "duration_days": duration_days,
                "depth": depth,
                "phase_start": phase_start,
                "phase_width": phase_width,
                "sr": results.power[idx]
            })
            
        target.candidates = candidates
        
        print(f"  Search complete. Found {len(candidates)} raw peaks.")
