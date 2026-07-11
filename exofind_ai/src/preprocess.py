"""
==============================================================
ExoFind AI — Light Curve Preprocessor
==============================================================

Responsibility:
  Accept a raw light curve in any of several formats produced
  by the astronomy pipeline (or Lightkurve) and return a clean,
  normalised DataFrame with columns ``time`` and ``flux``.

Pipeline steps
--------------
1. Schema normalisation  — remap arbitrary column names to the
   standard ``time`` / ``flux`` / ``flux_err`` schema.
2. Column validation     — raise a clear error if required
   columns are absent after remapping.
3. Duplicate removal     — drop repeated timestamps.
4. Missing-value removal — drop rows where time or flux is NaN.
5. Sigma-clipping        — remove astrophysical outliers
   (cosmic rays, detector artefacts) beyond N sigma.
6. Temporal sorting      — ensure chronological order.
7. Median normalisation  — divide flux by the median so that
   the out-of-transit baseline ≈ 1.0.
8. Index reset           — produce a clean 0-based index.

Author: Team ExoFind
==============================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from src.config import (
    FLUX_COLUMN,
    FLUX_ERR_COLUMN,
    MIN_LC_LENGTH,
    SIGMA_CLIP_THRESHOLD,
    TIME_COLUMN,
)
from src.utils import get_logger

logger = get_logger(__name__)

# ==============================================================
# COLUMN ALIAS MAP
# ==============================================================
# Maps common column name variants produced by Lightkurve,
# TESS, Kepler, and the in-house astronomy pipeline to the
# canonical names used by this module.
#
# Keys   = known aliases (lower-case).
# Values = canonical names from config.py.

_COLUMN_ALIASES: dict[str, str] = {
    # time variants
    "time": TIME_COLUMN,
    "bjd": TIME_COLUMN,
    "bkjd": TIME_COLUMN,
    "btjd": TIME_COLUMN,
    "cadence": TIME_COLUMN,
    "t": TIME_COLUMN,
    # flux variants
    "flux": FLUX_COLUMN,
    "sap_flux": FLUX_COLUMN,
    "pdcsap_flux": FLUX_COLUMN,
    "det_flux": FLUX_COLUMN,
    "norm_flux": FLUX_COLUMN,
    "detrended_flux": FLUX_COLUMN,
    "corrected_flux": FLUX_COLUMN,
    "f": FLUX_COLUMN,
    # flux-error variants
    "flux_err": FLUX_ERR_COLUMN,
    "sap_flux_err": FLUX_ERR_COLUMN,
    "pdcsap_flux_err": FLUX_ERR_COLUMN,
    "flux_error": FLUX_ERR_COLUMN,
    "f_err": FLUX_ERR_COLUMN,
    "err": FLUX_ERR_COLUMN,
}


class LightCurvePreprocessor:
    """
    Cleans and normalises a light curve DataFrame.

    Parameters
    ----------
    sigma_clip_threshold : float
        Number of MAD-based sigmas beyond which flux values are
        treated as outliers and removed.
        Defaults to ``config.SIGMA_CLIP_THRESHOLD``.
    min_length : int
        Minimum number of rows required after all cleaning steps.
        Raises ``ValueError`` if the light curve is shorter.
        Defaults to ``config.MIN_LC_LENGTH``.

    Examples
    --------
    >>> preprocessor = LightCurvePreprocessor()
    >>> clean_df = preprocessor.preprocess(raw_df)
    """

    def __init__(
        self,
        sigma_clip_threshold: float = SIGMA_CLIP_THRESHOLD,
        min_length: int = MIN_LC_LENGTH,
    ) -> None:
        self.sigma_clip_threshold = sigma_clip_threshold
        self.min_length = min_length

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def preprocess(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline on a light curve.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Raw light curve with at minimum a time column and a
            flux column.  Column names are normalised automatically.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with columns ``time``, ``flux``,
            and optionally ``flux_err``.  Index is reset to 0-based.

        Raises
        ------
        ValueError
            If required columns are missing after alias remapping,
            or if the cleaned light curve is shorter than
            ``self.min_length``.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pd.DataFrame, got {type(dataframe).__name__}."
            )

        df = dataframe.copy()
        n_input = len(df)
        logger.debug("Preprocessing light curve with %d rows.", n_input)

        df = self._normalise_schema(df)
        self._validate_columns(df)
        df = self._drop_duplicates(df)
        df = self._drop_nan(df)
        df = self._sigma_clip(df)
        df = self._sort_by_time(df)
        df = self._median_normalise(df)
        df = df.reset_index(drop=True)

        n_out = len(df)
        logger.info(
            "Preprocessing complete: %d -> %d rows (removed %d).",
            n_input,
            n_out,
            n_input - n_out,
        )

        if n_out < self.min_length:
            raise ValueError(
                f"Light curve has only {n_out} rows after preprocessing "
                f"(minimum required: {self.min_length}).  "
                "The input may be too short or mostly noise."
            )

        return df

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    @staticmethod
    def _normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remap any known column-name aliases to the canonical schema.

        Unknown columns are preserved unchanged so that the caller
        is not forced into a rigid CSV format.
        """
        rename_map: dict[str, str] = {}
        for col in df.columns:
            canonical = _COLUMN_ALIASES.get(col.lower().strip())
            if canonical and col != canonical:
                rename_map[col] = canonical

        if rename_map:
            logger.debug("Column rename map: %s", rename_map)
            df = df.rename(columns=rename_map)

        return df

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Raise ValueError if the required time / flux columns are missing."""
        required = {TIME_COLUMN, FLUX_COLUMN}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Required columns missing from light curve: {missing}.\n"
                f"Available columns: {list(df.columns)}.\n"
                "Add a column alias to the _COLUMN_ALIASES map in "
                "preprocess.py if your pipeline uses a different name."
            )

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with duplicate time stamps."""
        n_before = len(df)
        df = df.drop_duplicates(subset=[TIME_COLUMN])
        dropped = n_before - len(df)
        if dropped:
            logger.debug("Dropped %d duplicate time-stamp rows.", dropped)
        return df

    @staticmethod
    def _drop_nan(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where time or flux is NaN."""
        n_before = len(df)
        df = df.dropna(subset=[TIME_COLUMN, FLUX_COLUMN])
        dropped = n_before - len(df)
        if dropped:
            logger.debug("Dropped %d NaN rows.", dropped)
        return df

    def _sigma_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove flux outliers using a robust MAD-based sigma clip.

        Uses Median Absolute Deviation (MAD) instead of standard
        deviation because MAD is resistant to the very outliers
        being removed.
        """
        flux = df[FLUX_COLUMN].values
        median = np.median(flux)
        mad = median_abs_deviation(flux, scale="normal")   # σ-equivalent

        if mad == 0.0:
            logger.debug(
                "MAD is zero — skipping sigma-clipping (constant flux?)."
            )
            return df

        lower = median - self.sigma_clip_threshold * mad
        upper = median + self.sigma_clip_threshold * mad
        mask = (flux >= lower) & (flux <= upper)

        n_clipped = (~mask).sum()
        if n_clipped:
            logger.debug(
                "Sigma-clipped %d outlier flux points "
                "(threshold: %.1f σ, MAD σ: %.6f).",
                n_clipped,
                self.sigma_clip_threshold,
                mad,
            )

        return df[mask].copy()

    @staticmethod
    def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
        """Sort data frame by the time column in ascending order."""
        return df.sort_values(TIME_COLUMN).copy()

    @staticmethod
    def _median_normalise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Divide flux (and flux_err, if present) by the median flux.

        After normalisation the out-of-transit baseline ≈ 1.0,
        making features comparable across targets with different
        absolute brightness.
        """
        median_flux = np.median(df[FLUX_COLUMN].values)

        if median_flux == 0.0:
            raise ValueError(
                "Median flux is exactly 0.0 — normalisation is undefined. "
                "Check that the input light curve has been background-subtracted."
            )

        df = df.copy()
        df[FLUX_COLUMN] = df[FLUX_COLUMN] / median_flux

        if FLUX_ERR_COLUMN in df.columns:
            df[FLUX_ERR_COLUMN] = df[FLUX_ERR_COLUMN] / median_flux

        logger.debug("Median-normalised flux (median = %.6f).", median_flux)
        return df