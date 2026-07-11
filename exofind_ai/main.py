"""
==============================================================
ExoFind AI — Main Entry Point
==============================================================

CLI orchestrator.  Connects all pipeline stages via subcommands.

Usage
-----

  Build dataset from Kepler KOI catalog:
    python main.py build-dataset
    python main.py build-dataset --max-per-class 30 --dry-run

  Train the model:
    python main.py train

  Evaluate on test split:
    python main.py evaluate

  Predict a single light curve:
    python main.py predict --input data/lightcurves/12345678.csv
    python main.py predict --input data/lightcurves/12345678.csv --threshold 0.65

  Explain a prediction:
    python main.py explain --input data/lightcurves/12345678.csv
    python main.py explain --input data/lightcurves/12345678.csv --no-shap

  Run the legacy smoke-test (single dummy curve -> features.csv):
    python main.py smoke-test

Author: Team ExoFind
==============================================================
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    CONFIDENCE_THRESHOLD,
    MAX_SAMPLES_PER_CLASS,
    SAMPLE_LIGHTCURVE,
)
from src.utils import create_directories, get_logger, print_header

logger = get_logger(__name__)


# ==============================================================
# Sub-command handlers
# ==============================================================

def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Download and build the training dataset from Kepler KOI data."""
    from src.dataset_builder import KeplerDatasetBuilder

    max_per_class = None
    if args.max_per_class is not None:
        max_per_class = {cls: args.max_per_class for cls in MAX_SAMPLES_PER_CLASS}

    print_header("Building Kepler Dataset")
    builder = KeplerDatasetBuilder(
        max_per_class=max_per_class,
        dry_run=args.dry_run,
        resume=args.resume,
        bls_dir=args.bls_dir,
    )
    builder.build()
    if not args.dry_run:
        logger.info("Dataset build finished. Check data/features.csv.")


def cmd_train(args: argparse.Namespace) -> None:
    """Train the XGBoost classifier."""
    from src.train import ExoplanetTrainer

    print_header("Training ExoFind AI Model")
    trainer = ExoplanetTrainer()
    results = trainer.run()
    logger.info("Training complete. Results: %s", results)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate the trained model on the test split."""
    from src.evaluate import ModelEvaluator

    print_header("Evaluating ExoFind AI Model")
    evaluator = ModelEvaluator()
    metrics = evaluator.run()
    logger.info("Evaluation complete. Metrics: %s", metrics)


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict the class of a single light curve."""
    from src.predict import ExoplanetPredictor

    print_header("Predicting Light Curve Class")
    predictor = ExoplanetPredictor(
        confidence_threshold=args.threshold,
    )
    result = predictor.predict_from_csv(args.lightcurve, bls_path=args.bls)
    print(result)


def cmd_explain(args: argparse.Namespace) -> None:
    """Predict and (optionally) generate SHAP explanations."""
    import pandas as pd
    from src.predict import ExoplanetPredictor

    print_header("Predicting + Explaining Light Curve")

    predictor = ExoplanetPredictor(confidence_threshold=args.threshold)
    result = predictor.predict_from_csv(args.lightcurve, bls_path=args.bls)
    print(result)

    if args.no_shap:
        logger.info("SHAP generation skipped (--no-shap flag set).")
        return

    try:
        from src.explain import SHAPExplainer
        from src.config import FEATURE_NAMES_FILE
        from src.feature_extraction import FeatureExtractor
        from src.preprocess import LightCurvePreprocessor
        from src.config import ID_COLUMN

        # Rebuild aligned features for SHAP
        raw_df = pd.read_csv(args.lightcurve)
        preprocessor = LightCurvePreprocessor()
        extractor = FeatureExtractor(n_jobs=1, disable_progressbar=True)

        clean_df = preprocessor.preprocess(raw_df)
        clean_df[ID_COLUMN] = Path(args.lightcurve).stem
        raw_features = extractor.extract(clean_df)

        feature_names = FeatureExtractor.load_feature_names()
        features_aligned = FeatureExtractor.align_columns(
            raw_features, feature_names
        )

        explainer = SHAPExplainer()
        explainer.explain_local(features_aligned, sample_id=Path(args.lightcurve).stem)
        logger.info("SHAP explanation saved to data/shap/")

    except ImportError as exc:
        logger.warning("SHAP not available: %s", exc)


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Quick end-to-end test using the sample dummy light curve."""
    import pandas as pd
    from src.preprocess import LightCurvePreprocessor
    from src.feature_extraction import FeatureExtractor
    from src.config import FEATURES_FILE, ID_COLUMN

    print_header("Smoke Test — Dummy Light Curve")

    if not SAMPLE_LIGHTCURVE.exists():
        logger.error("Sample light curve not found: %s", SAMPLE_LIGHTCURVE)
        sys.exit(1)

    df = pd.read_csv(SAMPLE_LIGHTCURVE)
    logger.info("Loaded sample light curve: %d rows.", len(df))

    preprocessor = LightCurvePreprocessor(min_length=5)
    clean_df = preprocessor.preprocess(df)
    clean_df[ID_COLUMN] = 1

    extractor = FeatureExtractor(disable_progressbar=True)
    features = extractor.extract(clean_df)
    features.to_csv(FEATURES_FILE, index=False)

    logger.info(
        "Smoke test complete: %d features extracted.", features.shape[1]
    )
    logger.info("Features saved -> %s", FEATURES_FILE)
    print(features.head())


# ==============================================================
# Argument parser
# ==============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exofind",
        description="ExoFind AI — Exoplanet Transit Signal Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- build-dataset ---
    p_build = sub.add_parser(
        "build-dataset", help="Download and featurise Kepler light curves."
    )
    p_build.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        metavar="N",
        help="Override the max samples per class (default: from config.py).",
    )
    p_build.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the dataset without downloading any light curves.",
    )
    p_build.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint — skip already-processed stars.",
    )
    p_build.add_argument(
        "--bls-dir",
        type=str,
        default=None,
        help="Directory containing BLS JSON files for dataset building.",
    )

    # --- train ---
    sub.add_parser("train", help="Train the XGBoost classifier.")

    # --- evaluate ---
    sub.add_parser("evaluate", help="Evaluate on the hold-out test split.")

    # --- predict ---
    p_predict = sub.add_parser(
        "predict", help="Classify a single light curve."
    )
    p_predict.add_argument(
        "--lightcurve", "-l",
        required=True,
        type=str,
        help="Path to the light curve CSV file.",
    )
    p_predict.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD}).",
    )
    p_predict.add_argument(
        "--bls",
        type=str,
        default=None,
        help="Path to BLS summary JSON or CSV for hybrid features.",
    )

    # --- explain ---
    p_explain = sub.add_parser(
        "explain", help="Predict and generate SHAP explanations."
    )
    p_explain.add_argument(
        "--lightcurve", "-l",
        required=True,
        type=str,
        help="Path to the light curve CSV file.",
    )
    p_explain.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD}).",
    )
    p_explain.add_argument(
        "--bls",
        type=str,
        default=None,
        help="Path to BLS summary JSON or CSV for hybrid features.",
    )
    p_explain.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP generation (only show prediction).",
    )

    # --- smoke-test ---
    sub.add_parser(
        "smoke-test",
        help="End-to-end test using the sample dummy light curve.",
    )

    return parser


# ==============================================================
# Entry point
# ==============================================================

_COMMAND_MAP = {
    "build-dataset": cmd_build_dataset,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "predict": cmd_predict,
    "explain": cmd_explain,
    "smoke-test": cmd_smoke_test,
}


def main() -> None:
    """Main entry point — parse arguments and dispatch to handler."""
    create_directories()

    parser = build_parser()
    args = parser.parse_args()

    handler = _COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()