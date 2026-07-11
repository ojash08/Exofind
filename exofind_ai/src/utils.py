"""
==============================================================
ExoFind AI — Utility Functions
==============================================================

Provides:
  - Project-wide logger factory
  - Directory initialisation
  - Console header printing
  - JSON / joblib artifact helpers

All modules obtain their logger via ``get_logger(__name__)``.

Author: Team ExoFind
==============================================================
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib

# Force stdout to UTF-8 so Unicode characters (arrows, symbols)
# render correctly on Windows consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.config import (
    DATA_DIR,
    LIGHTCURVE_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    SHAP_DIR,
)

# ==============================================================
# LOGGING
# ==============================================================

_LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Root logger is configured once on first import.
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(
            stream=open(  # noqa: WPS515
                sys.stdout.fileno(),
                mode="w",
                encoding="utf-8",
                errors="replace",
                closefd=False,
            )
        )
    ],
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that inherits the root configuration.

    Parameters
    ----------
    name:
        Typically passed as ``__name__`` from the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


# ==============================================================
# DIRECTORY INITIALISATION
# ==============================================================

def create_directories() -> None:
    """
    Create all required project directories if they do not exist.

    This function is idempotent — safe to call multiple times.
    """
    dirs = [DATA_DIR, LIGHTCURVE_DIR, MODEL_DIR, PLOTS_DIR, SHAP_DIR]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

    log = get_logger(__name__)
    log.debug("All project directories verified / created.")


# ==============================================================
# CONSOLE UTILITIES
# ==============================================================

def print_header(title: str) -> None:
    """
    Print a formatted section header to stdout.

    Parameters
    ----------
    title:
        Title string to display inside the banner.
    """
    border = "=" * 60
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


# ==============================================================
# ARTIFACT I/O HELPERS
# ==============================================================

def save_json(data: dict[str, Any], path: Path) -> None:
    """
    Serialise a dictionary to a JSON file.

    Parameters
    ----------
    data:
        Dictionary to serialise. Keys must be strings.
    path:
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    get_logger(__name__).info("Saved JSON artifact -> %s", path)


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_artifact(obj: Any, path: Path) -> None:
    """
    Persist an arbitrary Python object with joblib.

    Parameters
    ----------
    obj:
        Object to persist (model, encoder, selector, …).
    path:
        Destination ``.pkl`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    get_logger(__name__).info("Saved artifact -> %s", path)


def load_artifact(path: Path) -> Any:
    """
    Load a joblib-persisted object from disk.

    Parameters
    ----------
    path:
        Source ``.pkl`` file path.

    Returns
    -------
    Any
        The deserialised object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``path``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)