import sys
import os
import json
import numpy as np

# Add the exofind root directory to the python path so imports work
EXOFIND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'exofind'))
sys.path.insert(0, EXOFIND_ROOT)

from core.target import ExoFindTarget
from pipeline.stages.photometry import PhotometryStage
from pipeline.stages.cleaning import CleaningStage
from pipeline.stages.search import BLSSearchStage
from pipeline.stages.vetting import VettingStage
from bls_v2.fold import fold_lightcurve
from bls_v2.weights import normalize_weights

# Environment variables to limit threads and avoid OpenBLAS crashes
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def nan_to_null(obj):
    """Recursively convert float('nan'), float('inf') to None for JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: nan_to_null(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        return [nan_to_null(v) for v in obj]
    return obj


def run_exofind_pipeline(tpf_path: str, progress_callback=None):
    """
    Runs the full ExoFind pipeline on a given TPF file and returns the serialized results.
    
    Args:
        tpf_path: Path to the FITS file.
        progress_callback: A function taking a dict like {"stage": "Name", "status": "running/done", ...}
    """
    if progress_callback:
        progress_callback({"stage": "Initialization", "status": "running"})
        
    target = ExoFindTarget(tpf_path)
    
    stages = [
        PhotometryStage(),
        CleaningStage(),
        BLSSearchStage(n_bins=400),
        VettingStage()
    ]
    
    for stage in stages:
        if progress_callback:
            progress_callback({"stage": stage.name, "status": "running"})
            
        try:
            stage.execute(target)
        except Exception as e:
            if progress_callback:
                progress_callback({"stage": stage.name, "status": "error", "message": str(e)})
            raise e
            
        if progress_callback:
            progress_callback({"stage": stage.name, "status": "done"})

    if progress_callback:
        progress_callback({"stage": "Result Serialization", "status": "running"})

    # Package the results
    result = {
        "target_name": f"TIC {target.tic_id}" if target.tic_id else target.filename,
        "tic_id": str(target.tic_id) if target.tic_id else None,
        "baseline": target.baseline,
        "cadences": len(target.clean_time) if target.clean_time is not None else 0,
        "light_curve": None,
        "periodogram": None,
        "folded_curve": None,
        "candidates": [],
        "best_candidate": target.best_candidate,
        "detection": target.best_candidate is not None
    }
    
    if target.clean_time is not None and target.clean_flux is not None:
        result["light_curve"] = {
            "time": target.clean_time.tolist(),
            "flux": target.clean_flux.tolist()
        }
        
    if target.bls_results is not None:
        result["periodogram"] = {
            "periods": target.bls_results["periods"].tolist(),
            "sde": target.bls_results["sde_array"].tolist(),
            "sr": target.bls_results["sr_array"].tolist()
        }
        
    if target.candidates:
        result["candidates"] = target.candidates
        
    # Generate folded curve for the best candidate
    if target.best_candidate is not None and target.clean_time is not None:
        best_period = target.best_candidate["period"]
        
        # Calculate folded curve using the bls_v2 utils
        x_tilde, w_tilde = normalize_weights(target.clean_flux)
        phase, folded_flux, folded_weights = fold_lightcurve(
            target.clean_time, x_tilde, w_tilde, best_period
        )
        
        # Add the mean back to make it ~1.0 normalized
        mean_flux = np.mean(target.clean_flux)
        display_flux = folded_flux + mean_flux
        
        result["folded_curve"] = {
            "phase": phase.tolist(),
            "flux": display_flux.tolist()
        }
        
    # --- Generate and Save Plot Image (ALWAYS) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # SDE
    ax1 = axes[0]
    ax1.plot(target.bls_results["periods"], target.bls_results["sde_array"], linewidth=0.8, color="#2196F3", alpha=0.8)
    ax1.axhline(6.0, color="red", linestyle="--", alpha=0.5, label="SDE = 6 threshold")
    if target.best_candidate is not None:
        best_period = target.best_candidate["period"]
        ax1.axvline(best_period, color="red", linestyle="-", alpha=0.7, label=f"Best: {best_period:.4f} d")
    ax1.set_xlabel("Period (days)")
    ax1.set_ylabel("SDE")
    ax1.set_title("Signal Detection Efficiency")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # SR
    ax2 = axes[1]
    ax2.plot(target.bls_results["periods"], target.bls_results["sr_array"], linewidth=0.8, color="#FF9800", alpha=0.8)
    if target.best_candidate is not None:
        ax2.scatter([best_period], [target.best_candidate["sr"]], color="red", s=80, zorder=5)
    ax2.set_xlabel("Period (days)")
    ax2.set_ylabel("Signal Residue")
    ax2.set_title("Signal Residue Periodogram")
    ax2.grid(alpha=0.3)
    
    # Folded
    ax3 = axes[2]
    if target.best_candidate is not None and target.clean_time is not None:
        ax3.scatter(phase, display_flux, s=1, alpha=0.3, color="#607D8B", label="Unbinned Data")
        
        from scipy.stats import binned_statistic
        phase_duration = target.best_candidate["duration_days"] / best_period
        optimal_bins = int(3 / phase_duration)
        n_plot_bins = max(50, min(500, optimal_bins))
        
        binned_flux, bin_edges, _ = binned_statistic(phase, display_flux, statistic='median', bins=n_plot_bins)
        binned_phase = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax3.plot(binned_phase, binned_flux, 'o', markersize=4, color="#E91E63", markeredgecolor="black", markeredgewidth=0.5)
        
        transit_start = target.best_candidate["phase_start"]
        transit_end = transit_start + target.best_candidate["phase_width"]
        ax3.axvspan(transit_start, min(transit_end, 1.0), alpha=0.15, color="red")
        if transit_end > 1.0:
            ax3.axvspan(0, transit_end - 1.0, alpha=0.15, color="red")
            
        ax3.set_title("Folded Light Curve")
    else:
        ax3.set_title("No Candidate Found (No Folded Curve)")
        
    ax3.set_xlabel("Phase")
    ax3.set_ylabel("Flux")
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    img_path = os.path.join(EXOFIND_ROOT, "exofind_bls_v2_results.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
        
    # Sanitize NaN/Inf for JSON serialization
    sanitized_result = nan_to_null(result)
    
    if progress_callback:
        progress_callback({"stage": "Complete", "status": "done"})
        
    return sanitized_result
