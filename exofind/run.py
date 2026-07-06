"""
ExoFind run.py — Primary Entry Point

Runs the generalized ExoFind Pipeline on a single
Target Pixel File.
"""

import sys
import argparse

from pipeline.pipeline import ExoFindPipeline
from pipeline.stages.photometry import PhotometryStage
from pipeline.stages.cleaning import CleaningStage
from pipeline.stages.search import BLSSearchStage
from pipeline.stages.vetting import VettingStage


def main():
    parser = argparse.ArgumentParser(description="ExoFind Pipeline (Stage 1-5)")
    parser.add_argument("tpf_path", nargs="?", default="data/mastDownload/TESS/tess2018234235059-s0002-0000000100100827-0121-s/tess2018234235059-s0002-0000000100100827-0121-s_tp.fits", help="Path to the Target Pixel File (.fits)")
    # For testing, we allow coarse search to run faster
    parser.add_argument("--fast", action="store_true", help="Run a faster, coarser period search")
    
    args = parser.parse_args()
    
    # 1. Initialize Pipeline
    pipeline = ExoFindPipeline(args.tpf_path)
    
    # 2. Configure Stages
    pipeline.add_stage(PhotometryStage())
    pipeline.add_stage(CleaningStage())
    
    # Configure search resolution
    n_bins = 200 if args.fast else 400
    pipeline.add_stage(BLSSearchStage(n_bins=n_bins))
    
    pipeline.add_stage(VettingStage())
    
    # 3. Execute
    target = pipeline.run()
    
    # 4. Summarize
    print("\n" + "=" * 65)
    print("  ExoFind Vetting Summary (Stage 5)")
    print("=" * 65)
    
    if not target.candidates:
        print("  No initial candidates found.")
        return
        
    print(f"{'Rank':<6}{'Period (d)':<14}{'SDE':<10}{'Status'}")
    print("-" * 65)
    
    for i, c in enumerate(target.candidates[:10]):
        print(
            f"{i+1:<6}"
            f"{c['period']:<14.4f}"
            f"{c['sde']:<10.2f}"
            f"{c.get('status', 'UNKNOWN')}"
        )
        
    print("\n" + "=" * 65)
    
    if target.best_candidate:
        best = target.best_candidate
        print(f"  VETTED EXOPLANET CANDIDATE FOUND")
        print(f"  Period   : {best['period']:.5f} days")
        print(f"  Duration : {best['duration_days'] * 24:.2f} hours")
        print(f"  Depth    : {abs(best['depth']) * 1e6:.0f} ppm")
        print(f"  SDE      : {best['sde']:.2f}")
    else:
        print("  NO VETTED CANDIDATES (All false positives or no signals)")
    
    print("=" * 65)


if __name__ == "__main__":
    main()
