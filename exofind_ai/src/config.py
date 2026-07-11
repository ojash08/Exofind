"""
==============================================================
ExoFind AI — Configuration
==============================================================

Single source of truth for ALL paths, constants, model
parameters, and class definitions used across the module.

NO other file should define paths or ML hyperparameters.
Import from here exclusively.

Resource philosophy
-------------------
This project is designed to run on a mid-range Windows laptop
with an older Intel i5 processor.  All resource limits below
are conservative by default and fully configurable here.

Author: Team ExoFind
==============================================================
"""

import os
from pathlib import Path

# ==============================================================
# PROJECT LAYOUT
# ==============================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"

# ==============================================================
# DATA DIRECTORIES
# ==============================================================

DATA_DIR: Path = PROJECT_ROOT / "data"
LIGHTCURVE_DIR: Path = DATA_DIR / "lightcurves"   # preprocessed CSV cache
PLOTS_DIR: Path = DATA_DIR / "plots"               # evaluation & SHAP plots
SHAP_DIR: Path = DATA_DIR / "shap"                 # SHAP output images

# ==============================================================
# DATA FILES
# ==============================================================

FEATURES_FILE: Path = DATA_DIR / "features.csv"   # streamed feature matrix
LABELS_FILE: Path = DATA_DIR / "labels.csv"        # kepid -> label mapping
CHECKPOINT_FILE: Path = DATA_DIR / "build_checkpoint.json"  # resume state

TRAIN_CSV: Path = DATA_DIR / "train.csv"
TEST_CSV: Path = DATA_DIR / "test.csv"

# Legacy sample used for quick smoke-tests
SAMPLE_LIGHTCURVE: Path = DATA_DIR / "sample_lightcurve.csv"

# ==============================================================
# MODEL ARTIFACTS
# ==============================================================

MODEL_DIR: Path = PROJECT_ROOT / "models"

MODEL_FILE: Path = MODEL_DIR / "xgboost_model.pkl"
LABEL_ENCODER_FILE: Path = MODEL_DIR / "label_encoder.pkl"
FEATURE_SELECTOR_FILE: Path = MODEL_DIR / "feature_selector.pkl"
FEATURE_NAMES_FILE: Path = MODEL_DIR / "feature_names.json"
TRAINING_METADATA_FILE: Path = MODEL_DIR / "training_metadata.json"

# ==============================================================
# REPRODUCIBILITY
# ==============================================================

RANDOM_STATE: int = 42

# ==============================================================
# RESOURCE MANAGEMENT
# ==============================================================

# Conservative CPU limit: use at most half the available cores,
# and always leave at least 1 core free for the OS / UI.
#
# Example on a 4-core i5: os.cpu_count() = 4 -> N_JOBS = 2
# Example on a 2-core machine: os.cpu_count() = 2 -> N_JOBS = 1
#
# Override this to 1 for maximum responsiveness, or increase it
# only if you are running the pipeline overnight unattended.
N_JOBS: int = max(1, (os.cpu_count() or 2) // 2)

# Hard cap on the total number of light curves downloaded.
# Prevents accidental runaway downloads on the Kepler archive.
# Increase this when the pipeline is ready for a larger dataset.
MAX_TOTAL_LIGHTCURVES: int = 650

# ==============================================================
# STORAGE MANAGEMENT
# ==============================================================

# If True, the raw Lightkurve download cache (.fits in ~/.lightkurve)
# is left to Lightkurve's own cache management.
# If True, the local preprocessed CSV copy in data/lightcurves/ is
# deleted after its features have been written to features.csv.
# Set to False if you want to keep local CSVs for manual inspection.
DELETE_LC_CSV_AFTER_EXTRACT: bool = False

# ==============================================================
# CLASSIFICATION TARGETS
# ==============================================================

# Maps raw KOI dispositions -> internal class labels.
# Extend this dict to add new classes without touching any other file.
DISPOSITION_MAP: dict[str, str] = {
    "CONFIRMED": "Confirmed Planet",
    "CANDIDATE": "Confirmed Planet",   # Candidates treated as planets
    "FALSE POSITIVE": "False Positive",
    # Eclipsing binaries flagged via koi_fpflag_ec; resolved in dataset_builder
    "ECLIPSING BINARY": "Eclipsing Binary",
    "VARIABLE STAR": "Variable Star",
    "NOISE": "Noise",
}

# Ordered list of class names used for LabelEncoder and evaluation axes.
# Index position == integer label sent to XGBoost.
CLASS_NAMES: list[str] = [
    "Confirmed Planet",
    "False Positive",
    "Eclipsing Binary",
    "Variable Star",
    "Noise",
]

NUM_CLASSES: int = len(CLASS_NAMES)
LABEL_COLUMN: str = "label"

# ==============================================================
# PREPROCESSING
# ==============================================================

# Sigma threshold for outlier flux removal (sigma-clipping)
SIGMA_CLIP_THRESHOLD: float = 5.0

# Minimum number of data points required after cleaning
MIN_LC_LENGTH: int = 20

# ==============================================================
# FEATURE EXTRACTION
# ==============================================================

# Column names expected from the astronomy pipeline.
# The preprocessor accepts flexible schemas and remaps to these.
TIME_COLUMN: str = "time"
FLUX_COLUMN: str = "flux"
FLUX_ERR_COLUMN: str = "flux_err"    # optional
ID_COLUMN: str = "id"

# Feature extraction mode.
# "efficient"     -> EfficientFCParameters  (~777 features, fast)
# "comprehensive" -> ComprehensiveFCParameters (~787 features, slow)
#
# Use "efficient" for all development and training on laptop hardware.
# Switch to "comprehensive" only if benchmarking shows clear benefit
# and you have access to a machine with more CPU cores and RAM.
FEATURE_MODE: str = "efficient"

# ==============================================================
# TRAINING PIPELINE
# ==============================================================

TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# VarianceThreshold — removes near-zero variance features
VARIANCE_THRESHOLD: float = 0.01

# Pearson correlation cutoff — one of a highly correlated pair is dropped
CORRELATION_THRESHOLD: float = 0.95

# SelectKBest — number of features to keep after supervised selection
SELECT_K_FEATURES: int = 100

# ==============================================================
# XGBOOST HYPERPARAMETERS
# ==============================================================

# Note: n_jobs here is intentionally set to N_JOBS (not -1) so that
# XGBoost respects the same conservative CPU limit as the rest of
# the pipeline.  Adjust N_JOBS above, not here.
XGB_PARAMS: dict = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

# ==============================================================
# PREDICTION & CONFIDENCE
# ==============================================================

# Minimum probability required for a definite class label.
# Predictions below this threshold return "Uncertain".
CONFIDENCE_THRESHOLD: float = 0.50

# ==============================================================
# DATASET BUILDER — KEPLER / KOI
# ==============================================================

# Max samples per class.  Total = sum of all values = 650 by default.
# Increase individual classes here; MAX_TOTAL_LIGHTCURVES acts as a
# hard safety cap across all classes combined.
MAX_SAMPLES_PER_CLASS: dict[str, int] = {
    "Confirmed Planet": 200,
    "False Positive": 200,
    "Eclipsing Binary": 100,
    "Variable Star": 100,
    "Noise": 50,
}

# Lightkurve author preference for Kepler data
KEPLER_AUTHOR: str = "Kepler"
KEPLER_EXPTIME: int = 1800   # long-cadence: 1800 s

# Seconds to pause between MAST API requests (be a good citizen)
MAST_PAUSE_S: float = 0.5

# NASA Exoplanet Archive TAP endpoint for KOI cumulative table
KOI_TAP_URL: str = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,kepoi_name,koi_disposition,"
    "koi_fpflag_ec,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,"
    "koi_period,koi_duration,koi_depth,koi_prad,koi_teq"
    "+from+cumulative&format=csv"
)