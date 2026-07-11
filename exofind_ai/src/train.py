"""
==============================================================
ExoFind AI — Training Pipeline
==============================================================

Responsibility:
  Load features.csv + labels.csv, run a full sklearn pipeline
  (variance filtering -> correlation removal -> SelectKBest),
  train an XGBoost classifier with StratifiedKFold CV, and
  persist all required artifacts:

    models/xgboost_model.pkl
    models/label_encoder.pkl
    models/feature_selector.pkl
    models/feature_names.json
    models/training_metadata.json

Author: Team ExoFind
==============================================================
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    CORRELATION_THRESHOLD,
    CV_FOLDS,
    FEATURE_NAMES_FILE,
    FEATURE_SELECTOR_FILE,
    FEATURES_FILE,
    LABEL_COLUMN,
    LABEL_ENCODER_FILE,
    LABELS_FILE,
    MODEL_FILE,
    N_JOBS,
    RANDOM_STATE,
    SELECT_K_FEATURES,
    TEST_SIZE,
    TRAINING_METADATA_FILE,
    VARIANCE_THRESHOLD,
    XGB_PARAMS,
)
from src.utils import get_logger, save_artifact, save_json

logger = get_logger(__name__)


class ExoplanetTrainer:
    """
    Full training pipeline for the ExoFind AI classifier.

    Stages
    ------
    1. Load features.csv + labels.csv
    2. Encode string labels -> integers (LabelEncoder)
    3. Variance threshold — drop near-zero variance features
    4. Correlation removal — drop one of each highly correlated pair
    5. SelectKBest (ANOVA F) — keep top K informative features
    6. StratifiedKFold cross-validation — trustworthy performance estimate
    7. Final model fit on full training data
    8. Persist all artifacts + training_metadata.json

    Parameters
    ----------
    xgb_params : dict, optional
        XGBoost hyperparameters.  Defaults to ``config.XGB_PARAMS``.
    test_size : float, optional
        Fraction of data held out as final test set.
    cv_folds : int, optional
        Number of stratified cross-validation folds.
    variance_threshold : float, optional
        Minimum variance; features below this are dropped.
    correlation_threshold : float, optional
        Maximum Pearson correlation allowed between two features.
    select_k : int, optional
        Number of features kept by SelectKBest.

    Examples
    --------
    >>> trainer = ExoplanetTrainer()
    >>> results = trainer.run()
    """

    def __init__(
        self,
        xgb_params: Optional[dict] = None,
        test_size: float = TEST_SIZE,
        cv_folds: int = CV_FOLDS,
        variance_threshold: float = VARIANCE_THRESHOLD,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        select_k: int = SELECT_K_FEATURES,
    ) -> None:
        self.xgb_params = dict(xgb_params or XGB_PARAMS)
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.select_k = select_k

        # State populated during training
        self._encoder: Optional[LabelEncoder] = None
        self._model: Optional[XGBClassifier] = None
        self._selected_feature_names: list[str] = []

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Execute the complete training pipeline.

        Returns
        -------
        dict
            Training results including CV metrics and paths to all
            saved artifacts.
        """
        logger.info("=== ExoFind AI — Training Pipeline ===")
        start_time = time.perf_counter()

        # --- 1. Load data ---
        features_raw, labels_raw = self._load_data()

        # --- 2. Encode labels ---
        y_encoded, class_counts = self._encode_labels(labels_raw)

        # --- 2b. Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            features_raw, y_encoded,
            test_size=self.test_size,
            random_state=RANDOM_STATE,
            stratify=y_encoded,
        )

        # --- 3. Feature selection pipeline ---
        X_selected = self._select_features(X_train, y_train)

        # --- 4. Cross-validation ---
        cv_results = self._cross_validate(X_selected, y_train)

        # --- 5. Final model fit ---
        self._fit_final_model(X_selected, y_train, class_counts)

        # --- 6. Persist artifacts ---
        self._save_all_artifacts()

        elapsed = time.perf_counter() - start_time
        results = self._build_results(cv_results, elapsed)

        self._save_metadata(results, class_counts)
        self._log_summary(results)

        return results

    # ----------------------------------------------------------
    # Step 1: Load data
    # ----------------------------------------------------------

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load features.csv and labels.csv and align them."""
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(
                f"Features file not found: {FEATURES_FILE}\n"
                "Run 'python main.py build-dataset' first."
            )
        if not LABELS_FILE.exists():
            raise FileNotFoundError(
                f"Labels file not found: {LABELS_FILE}\n"
                "Run 'python main.py build-dataset' first."
            )

        features = pd.read_csv(FEATURES_FILE)
        labels_df = pd.read_csv(LABELS_FILE)

        # Drop kepid index column if present
        if "kepid" in features.columns:
            features = features.set_index("kepid")
        labels_df = labels_df.set_index("kepid")

        # Align on shared index (kepid)
        shared_idx = features.index.intersection(labels_df.index)
        if len(shared_idx) == 0:
            raise ValueError(
                "Features and labels have no common kepid values. "
                "Rebuild the dataset."
            )

        features = features.loc[shared_idx]
        labels = labels_df.loc[shared_idx, LABEL_COLUMN]

        logger.info(
            "Loaded %d samples × %d raw features.", *features.shape
        )
        return features, labels

    # ----------------------------------------------------------
    # Step 2: Encode labels
    # ----------------------------------------------------------

    def _encode_labels(
        self, labels: pd.Series
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Fit a LabelEncoder on the class names defined in config.

        Parameters
        ----------
        labels : pd.Series
            String class labels.

        Returns
        -------
        y_encoded : np.ndarray
            Integer-encoded labels.
        class_counts : dict[str, int]
            Map from class name -> count in training data.
        """
        self._encoder = LabelEncoder()
        self._encoder.fit(CLASS_NAMES)   # fit on all classes so order is stable

        unknown = set(labels.unique()) - set(CLASS_NAMES)
        if unknown:
            raise ValueError(
                f"Labels contain unknown classes: {unknown}. "
                f"Expected classes: {CLASS_NAMES}."
            )

        y_encoded = self._encoder.transform(labels)
        class_counts = {
            cls: int((labels == cls).sum()) for cls in CLASS_NAMES
        }
        logger.info("Class distribution: %s", class_counts)
        return y_encoded, class_counts

    # ----------------------------------------------------------
    # Step 3: Feature selection
    # ----------------------------------------------------------

    def _select_features(
        self, features: pd.DataFrame, y: np.ndarray
    ) -> pd.DataFrame:
        """
        Apply three-stage feature reduction.

        Stage A — Variance threshold:
            Drop features whose variance < self.variance_threshold.

        Stage B — Correlation removal:
            For each pair of features with Pearson |r| above
            self.correlation_threshold, drop the one with lower
            mean absolute correlation to the rest (i.e. keep the
            more "central" feature).

        Stage C — SelectKBest (ANOVA F):
            Keep only the top K features ranked by ANOVA F-score
            against the target labels.

        The combined selection is saved as feature_selector.pkl
        (an ordered list of selected column names).
        """
        X = features.copy()

        # -- Stage A: variance filter --
        variances = X.var(axis=0)
        high_var_cols = variances[variances >= self.variance_threshold].index.tolist()
        dropped_var = len(X.columns) - len(high_var_cols)
        X = X[high_var_cols]
        logger.info(
            "Variance threshold: kept %d / %d features (dropped %d).",
            len(high_var_cols), features.shape[1], dropped_var,
        )

        # -- Stage B: correlation removal --
        corr_cols = self._remove_correlated(X)
        dropped_corr = len(X.columns) - len(corr_cols)
        X = X[corr_cols]
        logger.info(
            "Correlation removal: kept %d features (dropped %d).",
            len(corr_cols), dropped_corr,
        )

        # -- Stage C: SelectKBest --
        k = min(self.select_k, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        # SelectKBest does not support NaNs natively. 
        # We fill with 0 temporarily just for feature selection, 
        # so XGBoost can still learn native NaN splits later.
        selector.fit(X.fillna(0), y)
        mask = selector.get_support()
        selected_cols = X.columns[mask].tolist()
        X = X[selected_cols]
        logger.info(
            "SelectKBest: kept %d features.", len(selected_cols)
        )

        self._selected_feature_names = selected_cols
        return X

    def _remove_correlated(self, X: pd.DataFrame) -> list[str]:
        """
        Identify and drop one of each highly correlated feature pair.

        Returns
        -------
        list[str]
            Names of features to keep.
        """
        corr_matrix = X.corr(method="pearson").abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = {
            col
            for col in upper.columns
            if (upper[col] > self.correlation_threshold).any()
        }
        return [c for c in X.columns if c not in to_drop]

    # ----------------------------------------------------------
    # Step 4: Cross-validation
    # ----------------------------------------------------------

    def _cross_validate(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> dict[str, Any]:
        """
        Run StratifiedKFold CV and return aggregate metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Selected feature matrix.
        y : np.ndarray
            Encoded labels.

        Returns
        -------
        dict
            Mean and std of accuracy and macro F1 across folds.
        """
        logger.info(
            "Running %d-fold stratified cross-validation…", self.cv_folds
        )

        sample_weights = self._compute_sample_weights(y)

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE
        )

        # Build a throwaway model for CV (not the final saved model)
        cv_model = self._build_xgb_model(n_classes=len(np.unique(y)))

        cv_scores = cross_validate(
            cv_model,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            params={"sample_weight": sample_weights},
            return_train_score=False,
            n_jobs=N_JOBS,
        )

        results = {
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(cv_scores["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(cv_scores["test_f1_macro"])),
        }

        logger.info(
            "CV Accuracy : %.4f ± %.4f",
            results["cv_accuracy_mean"],
            results["cv_accuracy_std"],
        )
        logger.info(
            "CV F1 Macro : %.4f ± %.4f",
            results["cv_f1_macro_mean"],
            results["cv_f1_macro_std"],
        )
        return results

    # ----------------------------------------------------------
    # Step 5: Final model fit
    # ----------------------------------------------------------

    def _fit_final_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        class_counts: dict[str, int],
    ) -> None:
        """
        Fit XGBoost on the entire selected feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix after selection.
        y : np.ndarray
            Encoded labels.
        class_counts : dict[str, int]
            Used to compute sample weights.
        """
        logger.info("Fitting final XGBoost model on full training data…")
        sample_weights = self._compute_sample_weights(y)
        n_classes = len(self._encoder.classes_)

        self._model = self._build_xgb_model(n_classes=n_classes)
        self._model.fit(X, y, sample_weight=sample_weights)
        logger.info("Final model training complete.")

    def _build_xgb_model(self, n_classes: int) -> XGBClassifier:
        """Instantiate an XGBClassifier from config params."""
        params = dict(self.xgb_params)
        params["num_class"] = n_classes
        return XGBClassifier(**params)

    @staticmethod
    def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
        """
        Compute inverse-frequency sample weights to handle class imbalance.

        Rarer classes receive higher weights so XGBoost treats
        them as equally important during training.
        """
        classes, counts = np.unique(y, return_counts=True)
        class_weight = dict(zip(classes, 1.0 / counts))
        weights = np.array([class_weight[label] for label in y])
        # Normalise so that the total weight equals the number of samples
        weights = weights / weights.mean()
        return weights

    # ----------------------------------------------------------
    # Step 6: Persist artifacts
    # ----------------------------------------------------------

    def _save_all_artifacts(self) -> None:
        """Save model, encoder, selector, and feature names."""
        save_artifact(self._model, MODEL_FILE)
        save_artifact(self._encoder, LABEL_ENCODER_FILE)
        save_artifact(self._selected_feature_names, FEATURE_SELECTOR_FILE)
        save_json(
            {"feature_names": self._selected_feature_names},
            FEATURE_NAMES_FILE,
        )
        logger.info("All model artifacts saved to %s.", MODEL_FILE.parent)

    # ----------------------------------------------------------
    # Helpers: results + metadata
    # ----------------------------------------------------------

    def _build_results(
        self, cv_results: dict[str, Any], elapsed: float
    ) -> dict[str, Any]:
        """Compile a results dictionary for logging and metadata."""
        return {
            **cv_results,
            "n_selected_features": len(self._selected_feature_names),
            "n_classes": len(self._encoder.classes_),
            "class_names": list(self._encoder.classes_),
            "training_time_s": round(elapsed, 2),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }

    def _save_metadata(
        self,
        results: dict[str, Any],
        class_counts: dict[str, int],
    ) -> None:
        """Write training_metadata.json."""
        metadata = {
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "model_file": str(MODEL_FILE),
            "label_encoder_file": str(LABEL_ENCODER_FILE),
            "feature_selector_file": str(FEATURE_SELECTOR_FILE),
            "feature_names_file": str(FEATURE_NAMES_FILE),
            "random_state": RANDOM_STATE,
            "xgb_params": self.xgb_params,
            "class_counts": class_counts,
            **results,
        }
        save_json(metadata, TRAINING_METADATA_FILE)

    @staticmethod
    def _log_summary(results: dict[str, Any]) -> None:
        """Print a concise training summary to the log."""
        logger.info("─" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("─" * 50)
        logger.info(
            "  CV Accuracy  : %.4f ± %.4f",
            results["cv_accuracy_mean"],
            results["cv_accuracy_std"],
        )
        logger.info(
            "  CV F1 Macro  : %.4f ± %.4f",
            results["cv_f1_macro_mean"],
            results["cv_f1_macro_std"],
        )
        logger.info(
            "  Features kept: %d", results["n_selected_features"]
        )
        logger.info(
            "  Training time: %.1f s", results["training_time_s"]
        )
        logger.info("─" * 50)