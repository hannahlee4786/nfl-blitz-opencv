#!/bin/bash
# YOLO11m detections, learned re-ID embeddings, and the shipped associator.
# Usage: ml/run_tracker.sh <clip4|clip64|video path> <out_csv>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/Users/starfengsmac/Downloads/nfl-ml/.venv/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE
export HF_HOME="/Users/starfengsmac/Downloads/nfl-ml/models/hf"
export HUGGINGFACE_HUB_CACHE="/Users/starfengsmac/Downloads/nfl-ml/models/hf/hub"
export YOLO_CONFIG_DIR="/Users/starfengsmac/Downloads/nfl-ml/models/Ultralytics"
export NFL_ML_CACHE="${NFL_ML_CACHE:-/Users/starfengsmac/Downloads/nfl-ml/cache}"
W="/Users/starfengsmac/Downloads/nfl-ml/reid/checkpoints/best.pt"
WEIGHTS="/Users/starfengsmac/Downloads/nfl-ml/models/yolo11m.pt"
V4="$ROOT/data/input/Clip_4 Camera Swing and Distance Change Test.mp4"
V64="$ROOT/data/input/Clip 64.mov"

if [ "$#" -ne 2 ]; then
  echo "usage: ml/run_tracker.sh <clip4|clip64|video path> <out_csv>" >&2
  exit 2
fi
TARGET="$1"
OUT="$2"
cd "$ROOT"
mkdir -p "$(dirname "$OUT")" "$NFL_ML_CACHE/det"

if [ "$TARGET" = "clip4" ]; then
  CLIP="clip4"
  VIDEO="$V4"
elif [ "$TARGET" = "clip64" ]; then
  CLIP="clip64"
  VIDEO="$V64"
else
  VIDEO="$TARGET"
  CLIP="$(basename "$VIDEO")"
  CLIP="${CLIP%.*}"
fi

CACHE="$NFL_ML_CACHE/det/${CLIP}_yolo11m"
if [ "$TARGET" = "clip4" ] || [ "$TARGET" = "clip64" ]; then
  "$PY" ml/detect_all.py --clip "$CLIP" --detector yolo \
    --weights "$WEIGHTS" --imgsz 640 --conf 0.05 --tag yolo11m \
    --out-dir "$NFL_ML_CACHE/det"
else
  "$PY" ml/detect_all.py --video "$VIDEO" --detector yolo \
    --weights "$WEIGHTS" --imgsz 640 --conf 0.05 --tag yolo11m \
    --out-dir "$NFL_ML_CACHE/det"
fi
"$PY" ml/reid/embed_cache.py --cache "$CACHE" --video "$VIDEO" --weights "$W"
"$PY" ml/track_assoc.py --cache "$CACHE" --out "$OUT" \
  --low 0.07 --high 0.18 --min-hits 3 --coast 14 --stitch-gap 22 --interp-gap 10 \
  --contain 0.70 --hud-max 0.50 --min-h 20 --min-w 10 --max-aspect 5.2 --min-aspect 0.45 \
  --reid "$W" --reid-assoc 0.35 --reid-stitch 0.8
echo "wrote $OUT"
