"""
==============================================================
ExoFind AI — Model Evaluation Module
==============================================================

Responsibility:
  Load the trained model and test data, compute a full suite
  of classification metrics, and automatically save all plots
  to data/plots/.

Metrics generated
-----------------
- Accuracy, Precision, Recall, F1 (macro & per-class)
- Classification report
- Confusion matrix (normalised heatmap)
- One-vs-Rest ROC curve + AUC for each class
- Precision-Recall curve for each class
- XGBoost feature importance (top 20)

Author: Team ExoFind
==============================================================
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    CLASS_NAMES,
    FEATURE_NAMES_FILE,
    FEATURE_SELECTOR_FILE,
    FEATURES_FILE,
    LABEL_COLUMN,
    LABEL_ENCODER_FILE,
    LABELS_FILE,
    MODEL_FILE,
    PLOTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils import get_logger, load_artifact, load_json

logger = get_logger(__name__)

# Matplotlib style
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    }
)

_PALETTE = sns.color_palette("tab10", n_colors=len(CLASS_NAMES))


class ModelEvaluator:
    """
    Evaluate a trained ExoFind AI model on a hold-out test set.

    Parameters
    ----------
    test_size : float
        Fraction of the dataset used as test set.  Must match the
        value used during training.
    plots_dir : Path, optional
        Directory where all plots are saved.  Defaults to
        ``config.PLOTS_DIR``.

    Examples
    --------
    >>> evaluator = ModelEvaluator()
    >>> report = evaluator.run()
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        plots_dir: Optional[Path] = None,
    ) -> None:
        self.test_size = test_size
        self.plots_dir = Path(plots_dir) if plots_dir else PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self) -> dict:
        """
        Load artifacts, reconstruct the test split, and evaluate.

        Returns
        -------
        dict
            All computed metrics.
        """
        logger.info("=== ExoFind AI — Model Evaluation ===")

        model, encoder, selected_features = self._load_artifacts()
        X_test, y_test_enc = self._prepare_test_split(selected_features, encoder)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = self._compute_metrics(y_test_enc, y_pred, y_prob, encoder)
        self._plot_confusion_matrix(y_test_enc, y_pred, encoder)
        self._plot_roc_curves(y_test_enc, y_prob, encoder)
        self._plot_pr_curves(y_test_enc, y_prob, encoder)
        self._plot_feature_importance(model, selected_features)

        logger.info("All evaluation plots saved to %s", self.plots_dir)
        return metrics

    # ----------------------------------------------------------
    # Artifact loading
    # ----------------------------------------------------------

    @staticmethod
    def _load_artifacts():
        """Load model, encoder, and selected feature names."""
        model = load_artifact(MODEL_FILE)
        encoder = load_artifact(LABEL_ENCODER_FILE)
        feature_data = load_json(FEATURE_NAMES_FILE)
        selected_features = feature_data["feature_names"]
        logger.info(
            "Loaded model (%d features, %d classes).",
            len(selected_features),
            len(encoder.classes_),
        )
        return model, encoder, selected_features

    # ----------------------------------------------------------
    # Test split reconstruction
    # ----------------------------------------------------------

    def _prepare_test_split(
        self, selected_features: list[str], encoder
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Reconstruct the same test split used during training.

        Uses the identical random state and test_size so the
        evaluation is on genuinely unseen data.
        """
        from sklearn.model_selection import train_test_split

        features = pd.read_csv(FEATURES_FILE)
        labels_df = pd.read_csv(LABELS_FILE)

        if "kepid" in features.columns:
            features = features.set_index("kepid")
        labels_df = labels_df.set_index("kepid")

        shared_idx = features.index.intersection(labels_df.index)
        features = features.loc[shared_idx, selected_features]
        labels = labels_df.loc[shared_idx, LABEL_COLUMN]
        y = encoder.transform(labels)

        _, X_test, _, y_test = train_test_split(
            features, y,
            test_size=self.test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        logger.info("Test set: %d samples.", len(X_test))
        return X_test, y_test

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        encoder,
    ) -> dict:
        """Compute and log all scalar metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # One-vs-Rest ROC-AUC
        n_classes = y_prob.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        try:
            roc_auc = roc_auc_score(
                y_bin, y_prob, average="macro", multi_class="ovr"
            )
        except ValueError:
            roc_auc = float("nan")

        report = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )

        metrics = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
            "roc_auc_ovr_macro": round(roc_auc, 4),
        }

        logger.info("─" * 50)
        logger.info("EVALUATION METRICS")
        logger.info("─" * 50)
        for k, v in metrics.items():
            logger.info("  %-30s : %s", k, v)
        logger.info("\nClassification Report:\n%s", report)
        logger.info("─" * 50)

        return metrics

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, encoder
    ) -> None:
        """Save a normalised confusion matrix heatmap."""
        cm = confusion_matrix(
            y_true, y_pred, labels=np.arange(len(encoder.classes_)), normalize="true"
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=encoder.classes_
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
        ax.set_title("Normalised Confusion Matrix", fontsize=13, pad=14)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = self.plots_dir / "confusion_matrix.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved -> %s", path)

    def _plot_roc_curves(
        self, y_true: np.ndarray, y_prob: np.ndarray, encoder
    ) -> None:
        """Save One-vs-Rest ROC curves for each class."""
        n_classes = y_prob.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (cls, color) in enumerate(zip(encoder.classes_[:n_classes], _PALETTE)):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})", color=color)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves", fontsize=13)
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        path = self.plots_dir / "roc_curves.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved -> %s", path)

    def _plot_pr_curves(
        self, y_true: np.ndarray, y_prob: np.ndarray, encoder
    ) -> None:
        """Save Precision-Recall curves for each class."""
        n_classes = y_prob.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (cls, color) in enumerate(zip(encoder.classes_[:n_classes], _PALETTE)):
            if y_bin[:, i].sum() == 0:
                continue
            prec_vals, rec_vals, _ = precision_recall_curve(
                y_bin[:, i], y_prob[:, i]
            )
            ap = average_precision_score(y_bin[:, i], y_prob[:, i])
            ax.plot(
                rec_vals, prec_vals,
                label=f"{cls} (AP={ap:.3f})",
                color=color,
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves", fontsize=13)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        path = self.plots_dir / "pr_curves.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved -> %s", path)

    def _plot_feature_importance(
        self, model, selected_features: list[str], top_n: int = 20
    ) -> None:
        """Save a horizontal bar chart of XGBoost feature importances."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [selected_features[i] for i in indices]
        top_scores = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(
            range(len(top_features)), top_scores[::-1], color=colors[::-1]
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(
            [f[:45] + "…" if len(f) > 45 else f for f in top_features[::-1]],
            fontsize=8,
        )
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title(f"Top {top_n} XGBoost Feature Importances", fontsize=13)
        plt.tight_layout()
        path = self.plots_dir / "feature_importance.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved -> %s", path)
