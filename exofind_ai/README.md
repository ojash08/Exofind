# ExoFind AI 🔭

> **Production-quality AI module for classifying exoplanet transit signals from Kepler/TESS light curves.**

ExoFind AI is the machine-learning component of a larger two-part exoplanet detection project.  
It receives a polished light curve from the astronomy pipeline and returns a classification with an explainability report.

---

## Pipeline Position

```
Astronomy Pipeline (teammate)
  Raw TESS .fits  →  Lightkurve  →  Aperture Photometry
  →  Background Estimation  →  Light Curve Extraction
  →  Normalization  →  Detrending  →  BLS Transit Detection
  →  Periodogram  →  Polished Light Curve
                                          ↓
                              ┌─────────────────────┐
                              │    ExoFind AI        │
                              │  (this repository)   │
                              └─────────────────────┘
                                          ↓
                              Planet / Binary / Blend / Noise
                              + Confidence Score
                              + SHAP Explanation
                              → Ready for frontend
```

---

## Classification Targets

| Class | Source |
|---|---|
| **Confirmed Planet** | KOI CONFIRMED / CANDIDATE disposition |
| **False Positive** | KOI FALSE POSITIVE disposition |
| **Eclipsing Binary** | KOI `koi_fpflag_ec = 1` flag |
| **Variable Star** | Additional catalog sources |
| **Noise** | Additional catalog sources |

---

## Technology Stack

| Layer | Library |
|---|---|
| Feature extraction | `tsfresh` (EfficientFCParameters, ~777 features) |
| Classifier | `XGBoost` (multi:softprob) |
| Feature selection | `scikit-learn` (VarianceThreshold → Pearson → SelectKBest) |
| Explainability | `SHAP` (TreeExplainer) |
| Data access | `lightkurve`, `astroquery`, `requests` |
| Astronomy | `astropy` |

---

## Project Structure

```
ExoFindAI/
├── main.py                         # CLI entry point
├── requirements.txt
├── src/
│   ├── config.py                   # All paths, constants, hyperparameters
│   ├── utils.py                    # Logger, artifact I/O helpers
│   ├── preprocess.py               # LightCurvePreprocessor
│   ├── feature_extraction.py       # FeatureExtractor (tsfresh)
│   ├── dataset_builder.py          # KeplerDatasetBuilder
│   ├── train.py                    # ExoplanetTrainer
│   ├── evaluate.py                 # ModelEvaluator
│   ├── predict.py                  # ExoplanetPredictor
│   └── explain.py                  # SHAPExplainer
├── data/
│   ├── lightcurves/                # Preprocessed light curve CSVs
│   ├── labels.csv                  # kepid → label mapping
│   ├── features.csv                # Merged feature matrix (streamed)
│   ├── build_checkpoint.json       # Resume state for dataset builder
│   └── plots/                      # Evaluation + SHAP plots
└── models/
    ├── xgboost_model.pkl
    ├── label_encoder.pkl
    ├── feature_selector.pkl
    ├── feature_names.json
    └── training_metadata.json
```

---

## Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Build the training dataset

Downloads a balanced subset of Kepler light curves from MAST and
extracts tsfresh features. Features are streamed to disk one row at
a time — no large in-memory accumulation.

```bash
# Default run (up to 650 light curves from config.py)
python main.py build-dataset

# Smaller subset for development / testing
python main.py build-dataset --max-per-class 20

# Dry-run: plan downloads without actually fetching anything
python main.py build-dataset --dry-run

# Resume after a crash or interruption
python main.py build-dataset --resume
```

### Step 2 — Train

```bash
python main.py train
```

Saves the following artifacts to `models/`:

| Artifact | Purpose |
|---|---|
| `xgboost_model.pkl` | Trained classifier |
| `label_encoder.pkl` | String ↔ integer class mapping |
| `feature_selector.pkl` | Selected feature names |
| `feature_names.json` | Column order for prediction alignment |
| `training_metadata.json` | Full training record |

### Step 3 — Evaluate

```bash
python main.py evaluate
```

Saves to `data/plots/`:
- `confusion_matrix.png`
- `roc_curves.png`
- `pr_curves.png`
- `feature_importance.png`

### Step 4 — Predict a single light curve

```bash
python main.py predict --input data/lightcurves/12345678.csv

# With custom confidence threshold
python main.py predict --input data/lightcurves/12345678.csv --threshold 0.65
```

**Output example:**
```
Prediction     : Confirmed Planet
Confidence     : 0.8731
Threshold      : 0.50
Probabilities  :
  Confirmed Planet       0.8731  ████████████████████
  False Positive         0.0812  ██
  Eclipsing Binary       0.0291  █
  Variable Star          0.0121
  Noise                  0.0045
```

If confidence is below the threshold, the prediction is returned as **`Uncertain`**.

### Step 5 — Explain (optional SHAP)

```bash
# Full prediction + SHAP explanation
python main.py explain --input data/lightcurves/12345678.csv

# Prediction only, skip SHAP (faster)
python main.py explain --input data/lightcurves/12345678.csv --no-shap
```

Saves SHAP plots to `data/shap/`:
- `shap_summary.png` — global feature importance
- `shap_beeswarm.png` — feature direction detail
- `shap_force_<id>.png` — local force plot
- `shap_waterfall_<id>.png` — local waterfall plot

### Quick smoke-test (no internet required)

```bash
python main.py smoke-test
```

Runs the full preprocess → extract pipeline on a 10-row dummy light
curve and saves a `features.csv` — useful for verifying the install
without downloading any data.

---

## Resource Configuration

All resource limits live in [`src/config.py`](src/config.py).  
Edit these values to match your hardware — no other file needs to change.

```python
# CPU: use at most half the cores, leave the rest for the OS
N_JOBS = max(1, (os.cpu_count() or 2) // 2)

# Storage: hard cap on total downloads
MAX_TOTAL_LIGHTCURVES = 650

# Feature mode: "efficient" (~777 features, fast) or "comprehensive" (slower)
FEATURE_MODE = "efficient"

# Delete local CSV copies after features are saved
DELETE_LC_CSV_AFTER_EXTRACT = False

# Per-class download limits
MAX_SAMPLES_PER_CLASS = {
    "Confirmed Planet": 200,
    "False Positive":   200,
    "Eclipsing Binary": 100,
    "Variable Star":    100,
    "Noise":             50,
}
```

---

## Flexible Input Schema

The preprocessor accepts light curves from any source — it does not
require a rigid column naming convention.  The following column name
variants are automatically remapped:

| Your column name | Canonical name |
|---|---|
| `time`, `bjd`, `bkjd`, `btjd`, `t` | `time` |
| `flux`, `sap_flux`, `pdcsap_flux`, `det_flux`, `norm_flux`, `f` | `flux` |
| `flux_err`, `sap_flux_err`, `flux_error`, `err` | `flux_err` |

Add additional aliases to `_COLUMN_ALIASES` in
[`src/preprocess.py`](src/preprocess.py) if your pipeline uses a
different naming convention.

---

## Architecture Principles

- **Single responsibility** — each module has exactly one job
- **No hardcoded paths** — everything flows through `config.py`
- **No hardcoded CPU limits** — `N_JOBS` is computed from hardware
- **Streaming I/O** — dataset builder never accumulates features in RAM
- **Checkpointing** — dataset builds are resumable after interruption
- **Train/predict alignment** — `feature_names.json` guarantees identical column order at inference time
- **Configurable uncertainty** — predictions below `CONFIDENCE_THRESHOLD` return `"Uncertain"` instead of a forced label
- **Optional SHAP** — explainability is never mandatory; skip it for batch inference

---

## Data Sources

| Source | URL |
|---|---|
| NASA Exoplanet Archive (KOI catalog) | https://exoplanetarchive.ipac.caltech.edu |
| MAST (Kepler light curves) | https://mast.stsci.edu |

---

## Authors

Namit Agarwal

