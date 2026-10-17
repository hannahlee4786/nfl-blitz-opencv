"""Paths and environment for the self-training pipeline.

Large artifacts live under ART inside the ML home directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ml.paths import REPO, footage_dir, ml_home, models_dir, python_bin

PY = python_bin()
FOOTAGE = footage_dir()
ART = ml_home() / "selftrain"
SCRATCH = ml_home() / "scratch"
MODELS = models_dir()
SPLITS = ART / "splits.json"
YOLO11M = MODELS / "yolo11m.pt"
YOLO11N = MODELS / "yolo11n.pt"
YOLO11S = MODELS / "yolo11s.pt"

TEST = {
    "clip4": {
        "video": REPO / "data/input/Clip_4 Camera Swing and Distance Change Test.mp4",
        "labels": REPO / "data/labels/clip4.json",
    },
    "clip64": {
        "video": REPO / "data/input/Clip 64.mov",
        "labels": REPO / "data/labels/clip64.json",
    },
}

# Thumbnail used for near-duplicate and frame de-duplication.
THUMB_W = 64
THUMB_H = 36


def ensure_repo_on_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def ensure_dirs() -> None:
    for path in (
        ART,
        ART / "logs",
        ART / "leakage",
        ART / "reports",
        SCRATCH,
        SCRATCH / "pseudo_review",
        SCRATCH / "failures",
    ):
        path.mkdir(parents=True, exist_ok=True)


def footage_path(clip: int) -> Path:
    return FOOTAGE / f"Clip {clip}.mov"


def clip_name(clip: int) -> str:
    return f"clip_{clip:02d}"
