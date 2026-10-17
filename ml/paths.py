"""Repo-relative locations for caches, models, and footage.

Import this module before torch, ultralytics, or Hugging Face so the
cache directories are in the environment those libraries read.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def ml_home() -> Path:
    raw = os.environ.get("NFL_ML_HOME")
    if raw:
        return Path(raw).expanduser()
    return REPO / ".nfl-ml"


def models_dir() -> Path:
    return REPO / "models"


def cache_root() -> Path:
    raw = os.environ.get("NFL_ML_CACHE")
    if raw:
        return Path(raw).expanduser()
    return ml_home() / "cache"


def footage_dir() -> Path:
    raw = os.environ.get("NFL_FOOTAGE_DIR")
    if raw:
        return Path(raw).expanduser()
    matches = sorted(path for path in (REPO / "data").glob("drive-download-*") if path.is_dir())
    if matches:
        return matches[-1]
    return REPO / "data" / "drive-download"


def python_bin() -> Path:
    raw = os.environ.get("NFL_PYTHON")
    if raw:
        return Path(raw).expanduser()
    return REPO / ".venv" / "bin" / "python"


def reid_weights() -> Path:
    return models_dir() / "reid_resnet18.pt"


def apply_runtime_env() -> None:
    home = ml_home()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HOME", str(home / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(home / "hf" / "hub"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(home / "Ultralytics"))
    os.environ.setdefault("NFL_ML_CACHE", str(cache_root()))


apply_runtime_env()
