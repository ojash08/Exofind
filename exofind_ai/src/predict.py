"""
==============================================================
ExoFind AI — Prediction Module
==============================================================

Responsibility:
  Accept a single raw light curve (as a DataFrame or CSV path),
  run the full preprocessing + feature extraction pipeline, load
  all saved model artifacts, and return a structured prediction
  result including:

    - Predicted class label  (or "Uncertain" if confidence is
      below the configured threshold)
    - Probability vector for all classes
    - Confidence score (max probability)
    - Whether the result was flagged as uncertain

This module NEVER retrains the model.

Author: Team ExoFind
==============================================================
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    FEATURE_NAMES_FILE,
    FEATURE_SELECTOR_FILE,
    ID_COLUMN,
    LABEL_ENCODER_FILE,
    MODEL_FILE,
    TIME_COLUMN,
)
from src.feature_extraction import FeatureExtractor
from src.preprocess import LightCurvePreprocessor
from src.utils import get_logger, load_artifact, load_json

logger = get_logger(__name__)


# ==============================================================
# Prediction result dataclass
# ==============================================================

@dataclass
class PredictionResult:
    """
    Structured output of the ExoFind AI predictor.

    Attributes
    ----------
    predicted_class : str
        Human-readable class label, or ``"Uncertain"`` if the
        highest probability is below ``confidence_threshold``.
    confidence : float
        Probability of the predicted class (max of the probability
        vector).  Range [0, 1].
    is_uncertain : bool
        True when ``confidence < confidence_threshold``.
    probabilities : dict[str, float]
        Probability for every class in CLASS_NAMES.
    confidence_threshold : float
        The threshold used to determine uncertainty.
    """

    predicted_class: str
    confidence: float
    is_uncertain: bool
    probabilities: dict[str, float]
    confidence_threshold: float
    star_id: str = "Unknown"
    astronomy_params: dict[str, float] = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain dictionary (JSON-serialisable)."""
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "=" * 36,
            "         ExoFind AI Analysis         ",
            "=" * 36,
            f"Target                        {self.star_id}",
            f"Prediction                    {self.predicted_class}",
            f"Confidence                    {self.confidence * 100:.1f}%",
            "",
            "Top Predictions               ",
        ]
        
        for cls, prob in sorted(self.probabilities.items(), key=lambda x: -x[1]):
            # Dot padding for classes e.g. "Confirmed Planet ......... 96.5%"
            padded_cls = f"{cls} ".ljust(26, ".")
            lines.append(f"{padded_cls} {prob * 100:.1f}%")
            
        if self.astronomy_params:
            lines.extend([
                "",
                "Astronomical Parameters       ",
            ])
            for k, v in self.astronomy_params.items():
                # Add units for specific parameters
                unit = ""
                if k in ["Period", "Transit Duration"]:
                    unit = " days"
                    
                if isinstance(v, float):
                    lines.append(f"{k:<29} {v:.4g}{unit}")
                else:
                    lines.append(f"{k:<29} {v}{unit}")

        if self.top_features:
            lines.extend([
                "",
                "AI Explanation                ",
                "Top Contributing Features     ",
            ])
            for feat in self.top_features:
                lines.append(f"• {feat}")

        if not self.is_uncertain:
            decision = "High-confidence periodic transit consistent with an exoplanet candidate."
        else:
            decision = "Low confidence, manual review required."
            
        lines.extend([
            "",
            "Decision                      ",
            decision,
        ])
        
        return "\n".join(lines)


# ==============================================================
# Predictor class
# ==============================================================

class ExoplanetPredictor:
    """
    Classify a single light curve against the trained ExoFind AI model.

    Parameters
    ----------
    confidence_threshold : float, optional
        Minimum probability required to commit to a class label.
        Predictions below this threshold are returned as
        ``"Uncertain"``.  Defaults to ``config.CONFIDENCE_THRESHOLD``.
    model_file : Path, optional
        Override the default model path from config.
    encoder_file : Path, optional
        Override the default label encoder path from config.
    feature_names_file : Path, optional
        Override the default feature names JSON path from config.

    Examples
    --------
    >>> predictor = ExoplanetPredictor()
    >>> result = predictor.predict_from_csv("data/lightcurves/12345678.csv")
    >>> print(result)

    >>> import pandas as pd
    >>> lc_df = pd.read_csv("some_lightcurve.csv")
    >>> result = predictor.predict_from_dataframe(lc_df)
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        model_file: Optional[Path] = None,
        encoder_file: Optional[Path] = None,
        feature_names_file: Optional[Path] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold

        self._model_file = Path(model_file) if model_file else MODEL_FILE
        self._encoder_file = Path(encoder_file) if encoder_file else LABEL_ENCODER_FILE
        self._feature_names_file = (
            Path(feature_names_file) if feature_names_file else FEATURE_NAMES_FILE
        )

        # Lazy-loaded — loaded on first prediction call
        self._model = None
        self._encoder = None
        self._feature_names: Optional[list[str]] = None

        self._preprocessor = LightCurvePreprocessor()
        self._extractor = FeatureExtractor(n_jobs=1, disable_progressbar=True)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def predict_from_csv(self, csv_path: Union[str, Path], bls_path: Optional[Union[str, Path]] = None) -> PredictionResult:
        """
        Load a light curve CSV and predict its class.

        Parameters
        ----------
        csv_path : str or Path
            Path to a CSV file containing the light curve.
        bls_path : str or Path, optional
            Path to BLS JSON or CSV summary file.

        Returns
        -------
        PredictionResult
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Light curve file not found: {path}")

        df = pd.read_csv(path)
        logger.info("Predicting from file: %s (%d rows)", path.name, len(df))
        
        bls_data = None
        if bls_path:
            bls_path = Path(bls_path)
            if bls_path.exists():
                if bls_path.suffix.lower() == '.json':
                    with open(bls_path, 'r') as f:
                        bls_data = json.load(f)
                elif bls_path.suffix.lower() == '.csv':
                    bls_data = pd.read_csv(bls_path).iloc[0].to_dict()
                logger.info("Loaded BLS data from %s", bls_path.name)
            else:
                logger.warning("BLS file not found: %s", bls_path)

        return self.predict_from_dataframe(df, star_id=path.stem, bls_data=bls_data)

    def predict_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        star_id: str = "target",
        bls_data: Optional[dict[str, Any]] = None,
    ) -> PredictionResult:
        """
        Predict the class of a light curve supplied as a DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Raw light curve.  Will be preprocessed automatically.
        star_id : str, optional
            Identifier added as the ``id`` column for tsfresh.

        Returns
        -------
        PredictionResult
        """
        self._ensure_loaded()

        # -- Preprocess --
        clean_df = self._preprocessor.preprocess(dataframe)
        clean_df = clean_df.copy()
        clean_df[ID_COLUMN] = star_id

        # -- Extract features --
        raw_features = self._extractor.extract(clean_df, bls_data=bls_data)

        # -- Align to training columns --
        features_aligned = FeatureExtractor.align_columns(
            raw_features, self._feature_names
        )

        # -- Predict --
        prob_vector = self._model.predict_proba(features_aligned)[0]
        
        # -- Extract top features --
        importances = self._model.feature_importances_
        # To explain this specific instance, we could use SHAP, but as a fast proxy for the report
        # we will use the model's global feature importances multiplied by the local feature values (approx)
        # or simply return the top global features.
        top_indices = np.argsort(importances)[::-1][:5]
        top_features = [self._feature_names[i] for i in top_indices]
        
        return self._build_result(prob_vector, star_id=star_id, bls_data=bls_data, top_features=top_features)

    # ----------------------------------------------------------
    # Lazy loading
    # ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load all model artifacts if not already loaded."""
        if self._model is not None:
            return

        logger.info("Loading model artifacts…")

        self._model = load_artifact(self._model_file)
        self._encoder = load_artifact(self._encoder_file)

        feature_data = load_json(self._feature_names_file)
        self._feature_names = feature_data["feature_names"]

        logger.info(
            "Model loaded: %d features, classes: %s",
            len(self._feature_names),
            list(self._encoder.classes_),
        )

    # ----------------------------------------------------------
    # Result construction
    # ----------------------------------------------------------

    def _build_result(self, prob_vector: np.ndarray, star_id: str = "Unknown", bls_data: Optional[dict] = None, top_features: Optional[list] = None) -> PredictionResult:
        """
        Convert a raw probability vector into a PredictionResult.

        Parameters
        ----------
        prob_vector : np.ndarray
            Probability for each class in the encoder's class order.

        Returns
        -------
        PredictionResult
        """
        confidence = float(np.max(prob_vector))
        predicted_idx = int(np.argmax(prob_vector))
        is_uncertain = confidence < self.confidence_threshold

        if is_uncertain:
            predicted_label = "Uncertain"
            logger.info(
                "Prediction UNCERTAIN (max prob=%.4f < threshold=%.2f).",
                confidence,
                self.confidence_threshold,
            )
        else:
            predicted_label = self._encoder.inverse_transform([predicted_idx])[0]
            logger.info(
                "Prediction: %s (confidence=%.4f).", predicted_label, confidence
            )

        # Build per-class probability dict using encoder's classes
        prob_dict: dict[str, float] = {
            cls: round(float(prob), 6)
            for cls, prob in zip(self._encoder.classes_, prob_vector)
        }

        return PredictionResult(
            predicted_class=predicted_label,
            confidence=round(confidence, 6),
            is_uncertain=is_uncertain,
            probabilities=prob_dict,
            confidence_threshold=self.confidence_threshold,
            star_id=star_id,
            astronomy_params=bls_data or {},
            top_features=top_features or []
        )
