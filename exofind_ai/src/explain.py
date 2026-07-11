"""
==============================================================
ExoFind AI — SHAP Explainability Module
==============================================================

Responsibility:
  Generate SHAP (SHapley Additive exPlanations) visualisations
  that explain WHY the model predicted a given class.

SHAP is OPTIONAL.  The predict.py module does not depend on
this module.  Pass ``run_shap=False`` to any pipeline entry
point to skip explainability entirely (e.g. during batch
inference where speed matters).

Outputs (all auto-saved to data/shap/)
---------------------------------------
- shap_summary.png        — Global feature importance / direction
- shap_force_<id>.png     — Local explanation for one prediction
- shap_waterfall_<id>.png — Waterfall breakdown for one prediction
- shap_beeswarm.png       — Beeswarm / violin detail plot

Author: Team ExoFind
==============================================================
"""

import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_NAMES_FILE,
    LABEL_ENCODER_FILE,
    MODEL_FILE,
    SHAP_DIR,
)
from src.utils import get_logger, load_artifact, load_json

logger = get_logger(__name__)

# Lazy import — SHAP is only imported when actually needed.
try:
    import shap as _shap_lib
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning(
        "shap package not installed.  "
        "Install it with: pip install shap"
    )


class SHAPExplainer:
    """
    Generate SHAP explanations for the ExoFind AI XGBoost model.

    Parameters
    ----------
    shap_dir : Path, optional
        Directory where SHAP plots are saved.  Defaults to
        ``config.SHAP_DIR``.
    model_file : Path, optional
        Override the default model artifact path.
    encoder_file : Path, optional
        Override the default label encoder path.
    feature_names_file : Path, optional
        Override the default feature names JSON path.
    """

    def __init__(
        self,
        shap_dir: Optional[Path] = None,
        model_file: Optional[Path] = None,
        encoder_file: Optional[Path] = None,
        feature_names_file: Optional[Path] = None,
    ) -> None:
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "The 'shap' package is required for explainability. "
                "Install it with: pip install shap"
            )

        self.shap_dir = Path(shap_dir) if shap_dir else SHAP_DIR
        self.shap_dir.mkdir(parents=True, exist_ok=True)

        self._model_file = Path(model_file) if model_file else MODEL_FILE
        self._encoder_file = Path(encoder_file) if encoder_file else LABEL_ENCODER_FILE
        self._feature_names_file = (
            Path(feature_names_file) if feature_names_file else FEATURE_NAMES_FILE
        )

        # Lazy-loaded
        self._model = None
        self._encoder = None
        self._feature_names: Optional[list[str]] = None
        self._explainer: Optional[object] = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def explain_global(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
    ) -> None:
        """
        Generate global SHAP explanations across the full feature matrix.
        """
        self._ensure_loaded()
        logger.info("Computing global SHAP values for %d samples…", len(X))

        expl = self._compute_explanation(X)
        expl.feature_names = self._feature_names

        # For multiclass XGBoost, expl.values shape is (n_samples, n_features, n_classes)
        if len(expl.shape) == 3:
            mean_shap = np.abs(expl.values).mean(axis=(0, 2))
        else:
            mean_shap = np.abs(expl.values).mean(axis=0)

        self._plot_summary(mean_shap, max_display)
        self._plot_beeswarm(expl, max_display)

    def explain_local(
        self,
        X_single: pd.DataFrame,
        sample_id: str = "sample",
    ) -> None:
        """
        Generate local SHAP explanation for a single prediction.
        """
        self._ensure_loaded()

        if len(X_single) != 1:
            raise ValueError(
                f"explain_local expects exactly 1 row, got {len(X_single)}."
            )

        logger.info("Computing local SHAP explanation for '%s'…", sample_id)
        expl = self._compute_explanation(X_single)

        prob = self._model.predict_proba(X_single)[0]
        top_class_idx = int(np.argmax(prob))
        top_class_name = self._encoder.inverse_transform([top_class_idx])[0]
        logger.info(
            "Explaining predicted class: %s (prob=%.4f)",
            top_class_name,
            float(prob[top_class_idx]),
        )

        # Slice Explanation object to select the row and predicted class
        if len(expl.shape) == 3:
            expl_sliced = expl[0, :, top_class_idx]
        else:
            expl_sliced = expl[0]
            
        expl_sliced.feature_names = self._feature_names

        self._plot_force(expl_sliced, sample_id, top_class_name)
        self._plot_waterfall(expl_sliced, sample_id, top_class_name)

    # ----------------------------------------------------------
    # SHAP value computation
    # ----------------------------------------------------------

    def _compute_explanation(self, X: pd.DataFrame):
        """Run the TreeExplainer and return an Explanation object."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expl = self._explainer(X)
        return expl

    # ----------------------------------------------------------
    # Lazy loading
    # ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model artifacts and build the TreeExplainer."""
        if self._explainer is not None:
            return

        logger.info("Loading model artifacts for SHAP explainer…")
        self._model = load_artifact(self._model_file)
        self._encoder = load_artifact(self._encoder_file)
        feature_data = load_json(self._feature_names_file)
        raw_feature_names = feature_data["feature_names"]
        
        bls_keys = {
            "Period", "Transit Duration", "Transit Depth", 
            "Signal Detection Efficiency", "Signal Residue", 
            "False Alarm Probability", "Number of Significant Peaks", 
            "Number of Harmonic Peaks", "Peak Ratio", "Baseline", 
            "Number of Cadences", "Phase Start", "Phase Width", 
            "Detection Threshold", "Detection Flag"
        }
        self._feature_names = [
            f"[BLS] {f}" if f in bls_keys else f 
            for f in raw_feature_names
        ]

        logger.info("Building SHAP TreeExplainer…")
        self._explainer = _shap_lib.TreeExplainer(self._model)
        logger.info("SHAP TreeExplainer ready.")

    # ----------------------------------------------------------
    # Plot helpers
    # ----------------------------------------------------------

    def _plot_summary(
        self,
        mean_shap: np.ndarray,
        max_display: int,
    ) -> None:
        """Save a global summary bar plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        top_idx = np.argsort(mean_shap)[::-1][:max_display]
        top_names = [self._feature_names[i] for i in top_idx]
        top_vals = mean_shap[top_idx]

        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(top_names)))
        ax.barh(range(len(top_names)), top_vals[::-1], color=colors)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(
            [n[:45] + "…" if len(n) > 45 else n for n in top_names[::-1]],
            fontsize=8,
        )
        ax.set_xlabel("Mean |SHAP value| (impact on model output)")
        ax.set_title(
            f"Global Feature Importance — Top {max_display} Features",
            fontsize=13,
        )
        plt.tight_layout()
        path = self.shap_dir / "shap_summary.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP summary -> %s", path)

    def _plot_beeswarm(
        self,
        expl,
        max_display: int,
    ) -> None:
        """Save a beeswarm plot using the native shap library."""
        try:
            # Beeswarm plot typically expects a 2D Explanation object.
            # For multiclass, we can visualize the impact on the first class
            # or sum across classes. We'll visualize class 0 here.
            if len(expl.shape) == 3:
                expl_to_plot = expl[:, :, 0]
            else:
                expl_to_plot = expl
                
            fig, ax = plt.subplots(figsize=(10, 7))
            _shap_lib.plots.beeswarm(expl_to_plot, max_display=max_display, show=False)
            path = self.shap_dir / "shap_beeswarm.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close("all")
            logger.info("Saved SHAP beeswarm -> %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate beeswarm plot: %s", exc)

    def _plot_force(
        self,
        expl_sliced,
        sample_id: str,
        class_name: str,
    ) -> None:
        """Save a force plot for a single prediction."""
        try:
            # shap.plots.force returns a figure when matplotlib=True
            fig = _shap_lib.plots.force(
                expl_sliced.base_values, 
                expl_sliced.values, 
                expl_sliced.data, 
                feature_names=expl_sliced.feature_names, 
                matplotlib=True, 
                show=False
            )
            path = self.shap_dir / f"shap_force_{sample_id}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved SHAP force plot -> %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate force plot: %s", exc)

    def _plot_waterfall(
        self,
        expl_sliced,
        sample_id: str,
        class_name: str,
    ) -> None:
        """Save a waterfall plot for a single prediction."""
        try:
            plt.figure(figsize=(10, 7))
            _shap_lib.plots.waterfall(expl_sliced, max_display=15, show=False)
            path = self.shap_dir / f"shap_waterfall_{sample_id}.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close("all")
            logger.info("Saved SHAP waterfall -> %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate waterfall plot: %s", exc)
