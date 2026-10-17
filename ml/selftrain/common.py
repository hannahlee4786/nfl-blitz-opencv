"""Paths and environment for the self-training pipeline.

Large artifacts live under ART. The repo only holds this code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "/Users/starfengsmac/Downloads/nfl-ml/models/hf")
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    "/Users/starfengsmac/Downloads/nfl-ml/models/hf/hub",
)
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    "/Users/starfengsmac/Downloads/nfl-ml/models/Ultralytics",
)

REPO = Path("/Users/starfengsmac/Downloads/nfl-blitz-opencv-pipeline")
PY = Path("/Users/starfengsmac/Downloads/nfl-ml/.venv/bin/python")
OLD_PY = Path("/Users/starfengsmac/Downloads/[MIDWAY] NFL Blitz 2000/.venv/bin/python")
FOOTAGE = REPO / "data/drive-download-20261016T223913Z-1-001"
ART = Path("/Users/starfengsmac/Downloads/nfl-ml/selftrain")
SCRATCH = Path("/tmp/ml_st")
MODELS = Path("/Users/starfengsmac/Downloads/nfl-ml/models")
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
