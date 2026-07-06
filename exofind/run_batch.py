"""
ExoFind run_batch.py — Batch Processing

Runs the generalized ExoFind Pipeline on a directory of
Target Pixel Files and saves a summary to a CSV file.
"""

import sys
import os
import glob
import argparse

from pipeline.pipeline import ExoFindPipeline
from pipeline.stages.photometry import PhotometryStage
from pipeline.stages.cleaning import CleaningStage
from pipeline.stages.search import BLSSearchStage
from pipeline.stages.vetting import VettingStage


def main():
    parser = argparse.ArgumentParser(description="ExoFind Batch Processing")
    parser.add_argument("data_dir", help="Directory containing .fits files")
    parser.add_argument("--out", default="exofind_batch_results.csv", help="Output CSV file")
    parser.add_argument("--fast", action="store_true", help="Run a faster, coarser period search")
    
    args = parser.parse_args()
    
    fits_files = glob.glob(os.path.join(args.data_dir, "*.fits"))
    if not fits_files:
        print(f"No .fits files found in {args.data_dir}")
        return
        
    print(f"Found {len(fits_files)} files in {args.data_dir}. Beginning batch run...\n")
    
    results = []
    
    for i, file_path in enumerate(fits_files):
        print("\n" + "=" * 65)
        print(f"  Processing File {i+1}/{len(fits_files)}: {os.path.basename(file_path)}")
        print("=" * 65)
        
        # 1. Initialize Pipeline
        pipeline = ExoFindPipeline(file_path)
        
        # 2. Configure Stages
        pipeline.add_stage(PhotometryStage())
        pipeline.add_stage(CleaningStage())
        
        n_bins = 200 if args.fast else 400
        pipeline.add_stage(BLSSearchStage(n_bins=n_bins))
        pipeline.add_stage(VettingStage())
        
        # 3. Execute
        target = pipeline.run()
        
        # 4. Record best candidate
        if target.best_candidate:
            best = target.best_candidate
            results.append({
                "tic_id": target.tic_id if target.tic_id else target.filename,
                "status": "VETTED CANDIDATE",
                "period": best['period'],
                "depth_ppm": abs(best['depth']) * 1e6,
                "duration_hrs": best['duration_days'] * 24,
                "sde": best['sde'],
                "fap": target.bls_results.get("fap", "N/A"),
                "odd_even_sigma": best.get("odd_even_diff_sigma", 0),
                "secondary_ratio": best.get("secondary_ratio", 0)
            })
        else:
             results.append({
                "tic_id": target.tic_id if target.tic_id else target.filename,
                "status": "NO CANDIDATE / FALSE POSITIVE",
                "period": 0,
                "depth_ppm": 0,
                "duration_hrs": 0,
                "sde": 0,
                "fap": "N/A",
                "odd_even_sigma": 0,
                "secondary_ratio": 0
            })
            
    # 5. Save Summary
    print("\n" + "=" * 65)
    print(f"  Batch Run Complete. Saving results to {args.out}")
    print("=" * 65)
    
    with open(args.out, "w") as f:
        # Header
        f.write("TIC_ID,Status,Period_days,Depth_ppm,Duration_hrs,SDE,FAP,OddEvenSigma,SecondaryRatio\n")
        
        for r in results:
            f.write(f"{r['tic_id']},{r['status']},{r['period']:.5f},{r['depth_ppm']:.1f},"
                    f"{r['duration_hrs']:.2f},{r['sde']:.2f},{r['fap']},{r['odd_even_sigma']:.2f},"
                    f"{r['secondary_ratio']:.3f}\n")
                    

if __name__ == "__main__":
    main()
