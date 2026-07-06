"""
pipeline/stages/photometry.py — Stage 1 & 2

Handles Data Acquisition and Aperture Photometry.
Loads the FITS file, locates the star, estimates background,
optimizes the aperture, and extracts the raw flux.
"""

from pipeline.stage import PipelineStage
from photometry.loader import load_tpf
from photometry.star_locator import average_image, brightest_pixel
from photometry.background import estimate_local_background
from photometry.optimizer import optimize_aperture
from photometry.quality_flags import filter_quality

class PhotometryStage(PipelineStage):

    def execute(self, target):
        
        # --- 1. Load Data ---
        print(f"  Loading {target.filename}...")
        tpf = load_tpf(target.tpf_path)
        target.tpf = tpf
        
        # Try to extract TIC ID from header if available
        if hasattr(tpf, 'get_header'):
            try:
                hdr = tpf.get_header()
                if 'TICID' in hdr:
                    target.tic_id = hdr['TICID']
                elif 'OBJECT' in hdr:
                    target.tic_id = hdr['OBJECT'].replace('TIC ', '')
            except:
                pass
                
        print(f"  Frames: {len(tpf.time)}")
        
        # --- 2. Build Image ---
        avg = average_image(tpf)
        row, col = brightest_pixel(avg)
        
        # --- 3. Background ---
        bkg = estimate_local_background(avg, (row, col))
        
        # --- 4. Optimize Aperture ---
        print("  Optimizing aperture...")
        mask, best_order, time, flux, score, _ = optimize_aperture(
            tpf, avg, bkg, (row, col)
        )
        
        target.aperture_mask = mask
        target.aperture_score = score
        
        # --- 5. Quality Filter ---
        print("  Filtering by quality flags...")
        f_time, f_flux, f_quality = filter_quality(
            time, flux, tpf.quality
        )
        
        target.raw_time = f_time
        target.raw_flux = f_flux
        target.raw_quality = f_quality
        
        print(f"  Photometry complete. Final cadences: {len(f_time)}")
