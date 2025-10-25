"""Common path helpers for the credit appraisal PoC.

This centralises the logic that discovers the project root and the
locations used by the API, UI and training scripts. Previously many
modules hard-coded ``~/demo-library`` which breaks when the repository is
cloned elsewhere. The helpers below honour environment overrides while
falling back to paths relative to this file.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "API_DIR",
    "UI_DIR",
    "RUNS_DIR",
    "LANDING_IMG_DIR",
    "MODELS_DIR",
    "ensure_dir",
]


def _resolve(path: Path | str) -> Path:
    """Expand user/home references and resolve without requiring existence."""
    return Path(path).expanduser().resolve(strict=False)


def _env_path(*names: str, default: Path) -> Path:
    """Return the first environment variable that is set, else ``default``."""
    for name in names:
        value = os.getenv(name)
        if value:
            return _resolve(value)
    return _resolve(default)


# ────────────────────────────────────────────────────────────────
# Core locations
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = _env_path(
    "PROJECT_ROOT",
    "REPO",
    "ROOT",
    default=Path(__file__).resolve().parent.parent,
)
API_DIR = PROJECT_ROOT / "services" / "api"
UI_DIR = PROJECT_ROOT / "services" / "ui"

# Runtime artefacts (FastAPI runs, UI assets, trained models)
RUNS_DIR = _env_path("RUNS_DIR", "RUNS_ROOT", default=API_DIR / ".runs")
LANDING_IMG_DIR = _env_path("LANDING_IMG_DIR", "LANDING_IMAGES_DIR", default=UI_DIR / "landing_images")
MODELS_DIR = _env_path(
    "MODELS_DIR",
    "TRAINING_MODELS_DIR",
    "MODELS_ROOT",
    default=PROJECT_ROOT / "agents" / "credit_appraisal" / "models",
)


def ensure_dir(path: Path | str) -> Path:
    """Ensure ``path`` exists as a directory and return it as :class:`Path`."""
    resolved = _resolve(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
