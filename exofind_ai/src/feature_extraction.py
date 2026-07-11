"""
==============================================================
ExoFind AI — Feature Extraction Module
==============================================================

Responsibility:
  Accept a cleaned light curve DataFrame (from preprocess.py)
  and return a flat feature vector using tsfresh.

Design decisions
----------------
* Feature mode is controlled by config.FEATURE_MODE:
    "efficient"     -> EfficientFCParameters  (~777 features, fast)
    "comprehensive" -> ComprehensiveFCParameters (~787 features, slow)
  The default is "efficient", which is suitable for laptop hardware.

* n_jobs defaults to config.N_JOBS (conservative CPU limit).
  It is NOT set to -1; that would saturate all CPU cores.

* column_value is always set to FLUX_COLUMN so extra columns
  (flux_err, etc.) are not accidentally featurised.

* The exact list of extracted feature names is saved to
  FEATURE_NAMES_FILE (JSON) after the first extraction run.
  Subsequent predict-time extractions must produce the same
  columns in the same order — this is enforced by align_columns().

Author: Team ExoFind
==============================================================
"""

from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute

from src.config import (
    FEATURE_MODE,
    FEATURE_NAMES_FILE,
    FLUX_COLUMN,
    ID_COLUMN,
    N_JOBS,
    TIME_COLUMN,
)
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)

# Map config string to tsfresh parameter objects
_FC_PARAMETER_MAP = {
    "efficient": EfficientFCParameters,
    "comprehensive": ComprehensiveFCParameters,
}


def _build_fc_parameters(mode: str):
    """
    Instantiate the tsfresh feature-calculation parameter object.

    Parameters
    ----------
    mode : str
        One of ``"efficient"`` or ``"comprehensive"``.

    Returns
    -------
    EfficientFCParameters or ComprehensiveFCParameters

    Raises
    ------
    ValueError
        If ``mode`` is not a recognised value.
    """
    cls = _FC_PARAMETER_MAP.get(mode.lower())
    if cls is None:
        raise ValueError(
            f"Unknown FEATURE_MODE '{mode}'. "
            f"Valid options: {list(_FC_PARAMETER_MAP)}. "
            "Update config.FEATURE_MODE."
        )
    return cls()


class FeatureExtractor:
    """
    Extract time-series statistical features from a light curve.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel worker processes for tsfresh.
        Defaults to ``config.N_JOBS`` (conservative CPU limit).
        Set to 1 to disable multiprocessing entirely (safest on
        older Windows machines with limited RAM).
    disable_progressbar : bool
        Suppress tsfresh's per-feature progress bar.
        Defaults to False.
    feature_mode : str, optional
        ``"efficient"`` or ``"comprehensive"``.
        Defaults to ``config.FEATURE_MODE``.

    Examples
    --------
    Single light curve (prediction):

    >>> extractor = FeatureExtractor()
    >>> df_for_tsfresh = clean_df.copy()
    >>> df_for_tsfresh["id"] = "target_001"
    >>> features = extractor.extract(df_for_tsfresh)
    """

    def __init__(
        self,
        n_jobs: int = N_JOBS,
        disable_progressbar: bool = False,
        feature_mode: str = FEATURE_MODE,
    ) -> None:
        self.n_jobs = n_jobs
        self.disable_progressbar = disable_progressbar
        self.feature_mode = feature_mode
        self._fc_parameters = _build_fc_parameters(feature_mode)

        logger.debug(
            "FeatureExtractor initialised: mode=%s, n_jobs=%d.",
            feature_mode,
            n_jobs,
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def extract(
        self, dataframe: pd.DataFrame, bls_data: Optional[dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Run tsfresh feature extraction on one or more light curves,
        and optionally append astronomy features from the BLS pipeline.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Must contain columns:
              - ``id``    : integer or string identifier per light curve
              - ``time``  : timestamps (float)
              - ``flux``  : normalised flux values (float)
        bls_data : dict, optional
            Dictionary containing BLS astronomy features.

        Returns
        -------
        pd.DataFrame
            One row per unique ``id``.  All NaN values produced by
            tsfresh are imputed in-place.

        Raises
        ------
        ValueError
            If required columns are missing from ``dataframe``.
        """
        self._validate_input(dataframe)

        n_series = dataframe[ID_COLUMN].nunique()
        logger.info(
            "Extracting %s features from %d light curve(s) "
            "(n_jobs=%d).",
            self.feature_mode,
            n_series,
            self.n_jobs,
        )

        features: pd.DataFrame = extract_features(
            dataframe,
            column_id=ID_COLUMN,
            column_sort=TIME_COLUMN,
            column_value=FLUX_COLUMN,
            default_fc_parameters=self._fc_parameters,
            impute_function=impute,
            n_jobs=self.n_jobs,
            disable_progressbar=self.disable_progressbar,
        )

        logger.info(
            "Feature extraction complete: %d row(s) x %d features.",
            features.shape[0],
            features.shape[1],
        )

        if bls_data is not None:
            logger.info("Appending %d BLS astronomy features.", len(bls_data))
            # Convert BLS dict to a single-row DataFrame
            star_id = dataframe[ID_COLUMN].iloc[0]
            
            # Map Detection Flag to 1/0 if it exists
            mapped_bls = {}
            for k, v in bls_data.items():
                if isinstance(v, str) and v.upper() in ["YES", "NO"]:
                    mapped_bls[k] = 1.0 if v.upper() == "YES" else 0.0
                else:
                    mapped_bls[k] = float(v)
            
            bls_df = pd.DataFrame([mapped_bls], index=[star_id])
            # For multiple time series (batch mode), this simplistic merge assumes
            # bls_data applies to all or the single row. In our pipeline, it's 1 row at a time.
            # So we can safely join on the index.
            features = features.join(bls_df)

        return features

    # ----------------------------------------------------------
    # Feature-name persistence (train -> predict alignment)
    # ----------------------------------------------------------

    def save_feature_names(
        self,
        features: pd.DataFrame,
        path: Optional[Path] = None,
    ) -> None:
        """
        Save the ordered list of feature column names to JSON.

        Call this once after training-set extraction so that the
        predictor can reorder / validate its columns at inference time.

        Parameters
        ----------
        features : pd.DataFrame
            The extracted feature matrix whose columns will be saved.
        path : Path, optional
            Destination JSON file.  Defaults to
            ``config.FEATURE_NAMES_FILE``.
        """
        dest = Path(path) if path else FEATURE_NAMES_FILE
        save_json(
            {
                "feature_names": list(features.columns),
                "feature_mode": self.feature_mode,
                "n_features": len(features.columns),
            },
            dest,
        )
        logger.info(
            "Saved %d feature names -> %s", len(features.columns), dest
        )

    @staticmethod
    def load_feature_names(path: Optional[Path] = None) -> list[str]:
        """
        Load the saved feature names from JSON.

        Parameters
        ----------
        path : Path, optional
            Source JSON file.  Defaults to
            ``config.FEATURE_NAMES_FILE``.

        Returns
        -------
        list[str]
            Ordered list of feature column names.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist (model not yet trained).
        """
        src = Path(path) if path else FEATURE_NAMES_FILE
        data = load_json(src)
        return data["feature_names"]

    @staticmethod
    def align_columns(
        features: pd.DataFrame,
        expected_columns: list[str],
    ) -> pd.DataFrame:
        """
        Reorder and fill a feature DataFrame to match expected columns.

        Any column in ``expected_columns`` that is absent from
        ``features`` is added with value 0.0.  Extra columns in
        ``features`` that are not in ``expected_columns`` are dropped.

        This ensures that prediction-time feature vectors always
        have the same dimensionality and column order as the
        training-time feature matrix.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix to align.
        expected_columns : list[str]
            Ordered list of column names (from ``load_feature_names``).

        Returns
        -------
        pd.DataFrame
            Aligned feature matrix.
        """
        missing = set(expected_columns) - set(features.columns)
        extra = set(features.columns) - set(expected_columns)

        if missing:
            logger.warning(
                "%d feature column(s) missing at inference time -- "
                "filling with np.nan.  Check that the same tsfresh "
                "parameters were used for training and prediction.",
                len(missing),
            )
            import numpy as np
            for col in missing:
                features[col] = np.nan

        if extra:
            logger.debug(
                "Dropping %d extra column(s) not present during training.",
                len(extra),
            )

        return features[expected_columns]

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """Raise ValueError if required columns are absent."""
        required = {ID_COLUMN, TIME_COLUMN, FLUX_COLUMN}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Feature extraction requires columns {required}. "
                f"Missing: {missing}."
            )