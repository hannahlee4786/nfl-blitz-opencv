#!/usr/bin/env bash
# Create .venv on Python 3.12, install the pinned packages, and download YOLO11m.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
YOLO="$ROOT/models/yolo11m.pt"
EXPECTED="d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$ROOT/.venv"
  else
    python3.12 -m venv "$ROOT/.venv"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install -r "$ROOT/requirements.txt" --python "$ROOT/.venv/bin/python"
else
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi

mkdir -p "$ROOT/models"
got=""
if [ -f "$YOLO" ]; then
  got="$(shasum -a 256 "$YOLO" | awk '{print $1}')"
fi
if [ "$got" != "$EXPECTED" ]; then
  curl -L --fail --retry 3 -o "$YOLO.partial" "$URL"
  mv "$YOLO.partial" "$YOLO"
  got="$(shasum -a 256 "$YOLO" | awk '{print $1}')"
  if [ "$got" != "$EXPECTED" ]; then
    echo "yolo11m.pt sha256 is $got, expected $EXPECTED" >&2
    exit 1
  fi
fi
echo "yolo11m.pt sha256 $got"
echo "Setup finished."
echo "Put videos in data/input/, then run:"
echo "  bash ml/run_pipeline.sh clip4 data/output/clip4"
