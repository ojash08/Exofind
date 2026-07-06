"""
pipeline/stages/cleaning.py — Stage 3

Handles Data Cleaning.
Detrends the light curve to remove long-term stellar/instrumental
variations, and removes strong outliers.
"""

import numpy as np
from pipeline.stage import PipelineStage
from photometry.detrend import detrend
from photometry.outliers import remove_outliers

class CleaningStage(PipelineStage):

    def execute(self, target):
        
        if target.raw_time is None or target.raw_flux is None:
            raise ValueError("CleaningStage requires raw_time and raw_flux from PhotometryStage.")
            
        # Filter out any NaNs before math operations to prevent savgol failure
        nan_mask = ~np.isnan(target.raw_time) & ~np.isnan(target.raw_flux)
        valid_time = target.raw_time[nan_mask]
        valid_flux = target.raw_flux[nan_mask]

        print("  Detrending light curve (Aggressive Iterative Savgol)...")
        # Use a window of ~5 hours (151 cadences) to better track spots
        flat_flux, trend = detrend(valid_time, valid_flux, window_length=151)
        
        print("  Removing outliers...")
        time_clean, flux_clean = remove_outliers(
            valid_time, flat_flux
        )
        
        target.clean_time = time_clean
        target.clean_flux = flux_clean
        
        baseline = np.max(time_clean) - np.min(time_clean)
        target.baseline = baseline
        
        print(f"  Cleaning complete. Clean cadences: {len(time_clean)}")
        print(f"  Observation baseline: {baseline:.2f} days")
