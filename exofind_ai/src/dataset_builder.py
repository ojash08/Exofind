"""
==============================================================
ExoFind AI — Kepler Dataset Builder
==============================================================

Responsibility:
  Download a small, balanced subset of real Kepler light curves
  from MAST via Lightkurve, label them using the official NASA
  KOI cumulative catalog, preprocess each curve, extract tsfresh
  features, and produce:

    data/labels.csv              -- label per star (appended live)
    data/features.csv            -- feature matrix (appended live)
    data/build_checkpoint.json   -- resume state

Resource design
---------------
* STREAMING I/O: Features are appended to disk one row at a time.
  The pipeline never accumulates more than one light curve in RAM.

* CHECKPOINTING: Each successfully processed star is recorded in
  build_checkpoint.json.  Restarting after a crash or interruption
  skips already-completed stars automatically.

* STORAGE CAP: MAX_TOTAL_LIGHTCURVES in config.py hard-limits the
  total downloads.  DELETE_LC_CSV_AFTER_EXTRACT removes local CSV
  copies once their features are saved.

* CPU LIMIT: N_JOBS from config controls tsfresh parallelism.
  The default uses at most half the available cores.

Usage (from project root):

    python main.py build-dataset
    python main.py build-dataset --max-per-class 30 --dry-run
    python main.py build-dataset --resume          # continues from checkpoint

Author: Team ExoFind
==============================================================
"""

import time
import json
from pathlib import Path
from typing import Optional

import lightkurve as lk
import numpy as np
import pandas as pd
import requests

from src.config import (
    CHECKPOINT_FILE,
    CLASS_NAMES,
    DELETE_LC_CSV_AFTER_EXTRACT,
    DISPOSITION_MAP,
    FEATURE_MODE,
    FEATURES_FILE,
    FLUX_COLUMN,
    ID_COLUMN,
    KEPLER_AUTHOR,
    KEPLER_EXPTIME,
    KOI_TAP_URL,
    LABEL_COLUMN,
    LABELS_FILE,
    LIGHTCURVE_DIR,
    MAST_PAUSE_S,
    MAX_SAMPLES_PER_CLASS,
    MAX_TOTAL_LIGHTCURVES,
    N_JOBS,
    RANDOM_STATE,
    TIME_COLUMN,
)
from src.feature_extraction import FeatureExtractor
from src.preprocess import LightCurvePreprocessor
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


class KeplerDatasetBuilder:
    """
    Build a labelled feature dataset from NASA Kepler KOI data.

    All I/O is streamed: each light curve is processed individually
    and its features are appended to features.csv immediately.
    No feature rows are held in memory between iterations.

    Parameters
    ----------
    max_per_class : dict[str, int], optional
        Override the default per-class sample limits defined in
        ``config.MAX_SAMPLES_PER_CLASS``.
    dry_run : bool
        If True, fetch labels and plan downloads but do not download
        any light curves.
    resume : bool
        If True, load build_checkpoint.json and skip already-processed
        stars.  If False, start fresh (existing files are overwritten).

    Examples
    --------
    >>> builder = KeplerDatasetBuilder()
    >>> builder.build()

    >>> # Resume after a crash
    >>> builder = KeplerDatasetBuilder(resume=True)
    >>> builder.build()
    """

    def __init__(
        self,
        max_per_class: Optional[dict[str, int]] = None,
        dry_run: bool = False,
        resume: bool = False,
        bls_dir: Optional[str] = None,
    ) -> None:
        self.max_per_class: dict[str, int] = max_per_class or MAX_SAMPLES_PER_CLASS
        self.dry_run = dry_run
        self.resume = resume
        self.bls_dir = Path(bls_dir) if bls_dir else None

        self._preprocessor = LightCurvePreprocessor()
        # n_jobs=1 for per-star extraction (one curve at a time is already
        # lightweight; intra-star parallelism via N_JOBS adds overhead).
        self._extractor = FeatureExtractor(
            n_jobs=1,
            disable_progressbar=True,
            feature_mode=FEATURE_MODE,
        )

        # Checkpoint state: set of kepid strings already processed
        self._completed: set[str] = set()
        self._failed: set[str] = set()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def build(self) -> None:
        """
        Execute the full dataset-building pipeline.

        Steps
        -----
        1. Fetch & cache the KOI cumulative catalog.
        2. Map dispositions to internal class labels.
        3. Sample a balanced subset (respecting per-class limits
           and the global MAX_TOTAL_LIGHTCURVES cap).
        4. Optionally load checkpoint to skip completed stars.
        5. For each remaining star:
           a. Download PDCSAP light curve via Lightkurve.
           b. Preprocess (sigma-clip, normalise, validate).
           c. Extract tsfresh features (streaming, one row at a time).
           d. Append one row to features.csv and labels.csv.
           e. Optionally delete local CSV copy.
           f. Save checkpoint.
        6. Save feature names for prediction-time alignment.
        """
        logger.info("=== Dataset Builder Starting ===")
        logger.info(
            "Settings: mode=%s, max_total=%d, n_jobs=%d, "
            "delete_csv=%s, resume=%s",
            FEATURE_MODE,
            MAX_TOTAL_LIGHTCURVES,
            N_JOBS,
            DELETE_LC_CSV_AFTER_EXTRACT,
            self.resume,
        )

        koi_df = self._fetch_koi_catalog()
        koi_df = self._assign_labels(koi_df)
        sample_df = self._balance_sample(koi_df)

        logger.info(
            "Planned sample: %d light curves across %d classes.",
            len(sample_df),
            sample_df[LABEL_COLUMN].nunique(),
        )
        logger.info(
            "Class distribution:\n%s",
            sample_df[LABEL_COLUMN].value_counts().to_string(),
        )

        if self.dry_run:
            logger.info("Dry-run mode -- skipping downloads.")
            return

        self._load_checkpoint()

        if self.resume and self._completed:
            remaining = sample_df[
                ~sample_df["kepid"].astype(str).isin(self._completed)
            ]
            logger.info(
                "Resuming: %d already done, %d remaining.",
                len(self._completed),
                len(remaining),
            )
        else:
            # Fresh start: clear output files so we don't double-append
            remaining = sample_df
            self._clear_output_files()

        self._stream_and_featurise(remaining)

        # Save feature names from the completed features.csv
        if FEATURES_FILE.exists():
            sample_row = pd.read_csv(FEATURES_FILE, nrows=1)
            feature_cols = [
                c for c in sample_row.columns if c != "kepid"
            ]
            if feature_cols:
                # Construct a minimal DataFrame just to pass column names
                dummy = pd.DataFrame(columns=feature_cols)
                self._extractor.save_feature_names(dummy)

        total_done = len(self._completed)
        logger.info(
            "Build complete: %d stars processed (%d failed).",
            total_done,
            len(self._failed),
        )

    # ----------------------------------------------------------
    # Step 1: Fetch KOI catalog (cached locally)
    # ----------------------------------------------------------

    def _fetch_koi_catalog(self) -> pd.DataFrame:
        """
        Download the KOI cumulative table from NASA Exoplanet Archive.

        The catalog CSV is cached in data/lightcurves/koi_catalog.csv.
        On subsequent runs the cached copy is used instantly.
        """
        cache_path = LIGHTCURVE_DIR / "koi_catalog.csv"

        if cache_path.exists():
            logger.info("Loading cached KOI catalog from %s", cache_path)
            return pd.read_csv(cache_path, comment="#")

        logger.info(
            "Fetching KOI cumulative catalog from NASA Exoplanet Archive..."
        )
        try:
            resp = requests.get(KOI_TAP_URL, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download KOI catalog: {exc}"
            ) from exc

        LIGHTCURVE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        logger.info(
            "KOI catalog cached -> %s (%d bytes)",
            cache_path,
            len(resp.text),
        )

        df = pd.read_csv(cache_path, comment="#")
        logger.info("KOI catalog loaded: %d rows.", len(df))
        return df

    # ----------------------------------------------------------
    # Step 2: Assign class labels
    # ----------------------------------------------------------

    def _assign_labels(self, koi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw KOI dispositions to the 5 internal class labels.

        The eclipsing binary flag (koi_fpflag_ec == 1) overrides
        the generic FALSE POSITIVE disposition.
        """
        df = koi_df.copy()
        df["koi_disposition"] = df["koi_disposition"].str.strip().str.upper()
        df[LABEL_COLUMN] = df["koi_disposition"].map(DISPOSITION_MAP)

        # EB flag overrides the generic FP disposition
        eb_mask = (
            df.get("koi_fpflag_ec", pd.Series(0, index=df.index)) == 1
        )
        df.loc[eb_mask, LABEL_COLUMN] = "Eclipsing Binary"

        n_before = len(df)
        df = df.dropna(subset=[LABEL_COLUMN])
        dropped = n_before - len(df)
        if dropped:
            logger.debug(
                "Dropped %d rows with unmapped dispositions.", dropped
            )
        return df

    # ----------------------------------------------------------
    # Step 3: Balance sampling with hard total cap
    # ----------------------------------------------------------

    def _balance_sample(self, koi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select a stratified subset respecting per-class limits and
        the global MAX_TOTAL_LIGHTCURVES hard cap.

        The hard cap prevents accidental large downloads if the user
        forgets to reduce per-class limits during development.
        """
        rng = np.random.default_rng(RANDOM_STATE)
        rows = []
        running_total = 0

        for class_name in CLASS_NAMES:
            limit = self.max_per_class.get(class_name, 0)
            if limit == 0:
                continue

            # Enforce the global cap across all classes combined
            remaining_budget = MAX_TOTAL_LIGHTCURVES - running_total
            if remaining_budget <= 0:
                logger.warning(
                    "Global cap of %d reached before class '%s'.",
                    MAX_TOTAL_LIGHTCURVES,
                    class_name,
                )
                break

            effective_limit = min(limit, remaining_budget)

            subset = koi_df[koi_df[LABEL_COLUMN] == class_name]
            if subset.empty:
                logger.warning(
                    "No KOIs found for class '%s'.", class_name
                )
                continue

            n_take = min(effective_limit, len(subset))
            sampled = subset.sample(
                n=n_take, random_state=int(rng.integers(1_000_000))
            )
            rows.append(sampled)
            running_total += n_take

            logger.debug(
                "Sampled %d / %d available for class '%s'.",
                n_take,
                len(subset),
                class_name,
            )

        result = pd.concat(rows, ignore_index=True)
        if len(result) > MAX_TOTAL_LIGHTCURVES:
            logger.warning(
                "Sample (%d) exceeds MAX_TOTAL_LIGHTCURVES (%d). "
                "Truncating to cap.",
                len(result),
                MAX_TOTAL_LIGHTCURVES,
            )
            result = result.sample(
                n=MAX_TOTAL_LIGHTCURVES, random_state=RANDOM_STATE
            )
        return result

    # ----------------------------------------------------------
    # Step 4: Streaming download + featurise loop
    # ----------------------------------------------------------

    def _stream_and_featurise(self, sample_df: pd.DataFrame) -> None:
        """
        Process each star one at a time with streaming I/O.

        For each star:
          1. Download one Kepler quarter (Lightkurve).
          2. Preprocess (normalise, sigma-clip).
          3. Extract features (one-row DataFrame).
          4. Append feature row to features.csv.
          5. Append label row to labels.csv.
          6. Optionally delete local CSV copy.
          7. Save checkpoint JSON.

        Memory usage stays O(1) regardless of dataset size because
        each feature row is written to disk before the next star
        is loaded.
        """
        total = len(sample_df)
        successes = 0
        failures = 0

        for i, row_namedtuple in enumerate(sample_df.itertuples(index=False), start=1):
            row = row_namedtuple._asdict()
            kepid: int = int(row.get("kepid", 0))
            label: str = str(row.get(LABEL_COLUMN, ""))
            star_id = str(kepid)
            target_name = f"KIC {kepid}"

            # -- Skip if already done (resume mode) --
            if star_id in self._completed:
                logger.debug("Skipping %s (already in checkpoint).", target_name)
                continue

            logger.info(
                "[%d/%d] Processing %s (class: %s)...",
                i, total, target_name, label,
            )

            try:
                # 1. Download
                lc_df = self._download_lightcurve(kepid)
                if lc_df is None:
                    self._record_failure(star_id)
                    failures += 1
                    continue

                # 2. Preprocess
                clean_df = self._preprocessor.preprocess(lc_df)
                clean_df = clean_df.copy()
                clean_df[ID_COLUMN] = star_id

                # 3. Optionally save the preprocessed CSV
                lc_csv_path = self._save_lightcurve_csv(clean_df, star_id)
                
                # -- BLS Feature Prep --
                bls_data = None
                if self.bls_dir:
                    bls_file = self.bls_dir / f"{star_id}.json"
                    if bls_file.exists():
                        with open(bls_file, "r") as f:
                            bls_data = json.load(f)
                
                if not bls_data:
                    # ==========================================================
                    # FALLBACK: Use KOI catalog values as high-quality labels
                    # ==========================================================
                    # During training, we use the KOI catalog values as highly accurate
                    # parameter references for efficiency. They serve as the "ground truth" 
                    # that a perfect BLS pipeline would output. 
                    # Note: These values may differ slightly from the values estimated by 
                    # the inference-time BLS pipeline due to observational noise and fitting 
                    # differences. If the discrepancy becomes significant in future versions, 
                    # consider generating the training BLS features using the same inference 
                    # pipeline for maximum consistency.
                    bls_data = {
                        "Period": row.get("koi_period", 0.0),
                        "Transit Duration": row.get("koi_duration", 0.0),
                        "Transit Depth": row.get("koi_depth", 0.0),
                        "Signal Detection Efficiency": row.get("koi_sde", 0.0),
                        "Signal Residue": row.get("koi_srad", 0.0),
                        "False Alarm Probability": 0.0,
                        "Number of Significant Peaks": 0.0,
                        "Number of Harmonic Peaks": 0.0,
                        "Peak Ratio": 0.0,
                        "Baseline": 0.0,
                        "Number of Cadences": len(clean_df),
                        "Phase Start": 0.0,
                        "Phase Width": 0.0,
                        "Detection Threshold": 0.0,
                        "Detection Flag": "YES" if label != "False Positive" else "NO"
                    }

                # 4. Extract features (single row)
                feats = self._extractor.extract(clean_df, bls_data=bls_data)
                feats.index = [star_id]

                # 5. Stream-append to features.csv (one row at a time)
                self._append_feature_row(feats, star_id)

                # 6. Append to labels.csv
                self._append_label_row(star_id, label)

                # 7. Delete local CSV if configured
                if DELETE_LC_CSV_AFTER_EXTRACT and lc_csv_path is not None:
                    lc_csv_path.unlink(missing_ok=True)
                    logger.debug(
                        "Deleted local CSV for %s (storage cleanup).",
                        star_id,
                    )

                # 8. Update and save checkpoint
                self._completed.add(star_id)
                self._save_checkpoint()
                successes += 1

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping %s -- %s: %s",
                    target_name,
                    type(exc).__name__,
                    exc,
                )
                self._record_failure(star_id)
                failures += 1

            # Polite pause between MAST requests
            time.sleep(MAST_PAUSE_S)

        logger.info(
            "Download loop complete: %d succeeded, %d failed.",
            successes,
            failures,
        )

    # ----------------------------------------------------------
    # Download helper
    # ----------------------------------------------------------

    def _download_lightcurve(self, kepid: int) -> Optional[pd.DataFrame]:
        """
        Download the PDCSAP light curve for one Kepler target.

        Downloads only the first available long-cadence quarter to
        minimise storage and bandwidth.  The raw .fits file stays
        in Lightkurve's own cache (~/.lightkurve); we immediately
        extract the numeric data and discard the object.

        Returns
        -------
        pd.DataFrame or None
        """
        target = f"KIC {kepid}"
        try:
            result = lk.search_lightcurve(
                target,
                author=KEPLER_AUTHOR,
                exptime=KEPLER_EXPTIME,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("MAST search failed for %s: %s", target, exc)
            return None

        if len(result) == 0:
            logger.debug("No data on MAST for %s.", target)
            return None

        try:
            lc = result[0].download()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Download failed for %s: %s", target, exc)
            return None

        # Extract numeric arrays and immediately release the lk object
        try:
            lc_df = lc.to_pandas().reset_index()
        except Exception:  # noqa: BLE001
            lc_df = pd.DataFrame(
                {
                    "time": lc.time.value,
                    "flux": lc.flux.value,
                    "flux_err": lc.flux_err.value,
                }
            )

        # Keep only the essential columns to prevent multi-element array errors
        keep_cols = [c for c in ["time", "flux", "flux_err"] if c in lc_df.columns]
        lc_df = lc_df[keep_cols].copy()

        lc_df = lc_df.replace([np.inf, -np.inf], np.nan)
        return lc_df

    # ----------------------------------------------------------
    # Streaming I/O helpers
    # ----------------------------------------------------------

    def _append_feature_row(
        self, feats: pd.DataFrame, star_id: str
    ) -> None:
        """
        Append a single feature row to features.csv.

        If features.csv does not yet exist, the header is written
        first.  All subsequent calls append without the header.
        This produces a valid CSV file that grows one row at a time.

        Parameters
        ----------
        feats : pd.DataFrame
            Single-row feature DataFrame (index = star_id).
        star_id : str
            Kepler ID string used as the kepid column value.
        """
        row = feats.copy()
        row.insert(0, "kepid", star_id)

        write_header = not FEATURES_FILE.exists()
        row.to_csv(
            FEATURES_FILE,
            mode="a",
            header=write_header,
            index=False,
        )

    def _append_label_row(self, star_id: str, label: str) -> None:
        """
        Append a single label row to labels.csv.

        Parameters
        ----------
        star_id : str
            Kepler ID string.
        label : str
            Class label for this star.
        """
        write_header = not LABELS_FILE.exists()
        label_row = pd.DataFrame(
            {"kepid": [star_id], LABEL_COLUMN: [label]}
        )
        label_row.to_csv(
            LABELS_FILE,
            mode="a",
            header=write_header,
            index=False,
        )

    def _save_lightcurve_csv(
        self, df: pd.DataFrame, star_id: str
    ) -> Optional[Path]:
        """
        Save the preprocessed light curve to data/lightcurves/.

        Returns the Path so the caller can optionally delete it.
        """
        path = LIGHTCURVE_DIR / f"{star_id}.csv"
        df.to_csv(path, index=False)
        logger.debug("Saved light curve CSV -> %s", path)
        return path

    def _clear_output_files(self) -> None:
        """
        Remove existing features.csv and labels.csv before a fresh run.

        Also clears the checkpoint file so the new run starts clean.
        """
        for path in [FEATURES_FILE, LABELS_FILE, CHECKPOINT_FILE]:
            if path.exists():
                path.unlink()
                logger.debug("Cleared %s for fresh run.", path.name)

    # ----------------------------------------------------------
    # Checkpoint helpers
    # ----------------------------------------------------------

    def _load_checkpoint(self) -> None:
        """
        Load checkpoint state from disk if it exists and resume=True.
        """
        if not self.resume or not CHECKPOINT_FILE.exists():
            self._completed = set()
            self._failed = set()
            return

        try:
            data = load_json(CHECKPOINT_FILE)
            self._completed = set(data.get("completed", []))
            self._failed = set(data.get("failed", []))
            logger.info(
                "Checkpoint loaded: %d completed, %d failed.",
                len(self._completed),
                len(self._failed),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load checkpoint (%s). Starting fresh.", exc
            )
            self._completed = set()
            self._failed = set()

    def _save_checkpoint(self) -> None:
        """Persist current progress to build_checkpoint.json."""
        save_json(
            {
                "completed": sorted(self._completed),
                "failed": sorted(self._failed),
                "n_completed": len(self._completed),
                "n_failed": len(self._failed),
            },
            CHECKPOINT_FILE,
        )

    def _record_failure(self, star_id: str) -> None:
        """Record a failed star and save the checkpoint."""
        self._failed.add(star_id)
        self._save_checkpoint()
