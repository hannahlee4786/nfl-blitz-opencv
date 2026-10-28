#!/bin/bash
# YOLO11m detections, learned re-ID embeddings, and the shipped associator.
# Usage: ml/run_tracker.sh <clip4|clip64|video path> <out_csv> [motion_out]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${NFL_PYTHON:-$ROOT/.venv/bin/python}"
ML_HOME="${NFL_ML_HOME:-$ROOT/.nfl-ml}"
export NFL_ML_HOME="$ML_HOME"
export KMP_DUPLICATE_LIB_OK=TRUE
export HF_HOME="${HF_HOME:-$ML_HOME/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$ML_HOME/hf/hub}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ML_HOME/Ultralytics}"
export NFL_ML_CACHE="${NFL_ML_CACHE:-$ML_HOME/cache}"
W="${NFL_REID_WEIGHTS:-$ROOT/models/reid_resnet18.pt}"
WEIGHTS="$ROOT/models/yolo11m.pt"
V4="$ROOT/data/input/Clip_4 Camera Swing and Distance Change Test.mp4"
V64="$ROOT/data/input/Clip 64.mov"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: ml/run_tracker.sh <clip4|clip64|video path> <out_csv> [motion_out]" >&2
  exit 2
fi
TARGET="$1"
OUT="$2"
MOTION_OUT="${3:-}"
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
  --team-penalty 0.5 --team-margin 0.6 --team-partial 0.9 \
  --roster-rescue-len 12 --roster-rescue-conf 0.30 --icon-chain-share 0.5 \
  --cone-run-frames 12 --cone-warm 0.30 \
  --reid "$W" --reid-assoc 0.35 --reid-stitch 0.8
if [ -n "$MOTION_OUT" ]; then
  mkdir -p "$(dirname "$MOTION_OUT")"
  cp "$CACHE.motion.npy" "$MOTION_OUT"
fi
echo "wrote $OUT"
