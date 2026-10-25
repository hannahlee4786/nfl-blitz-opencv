#!/bin/bash
# Detect players, associate tracks, project them onto the field, and
# calibrate team colors from the boxes.
# Usage: ml/run_pipeline.sh <clip4|clip64|video path> <out_dir> [extra track_clip4 args]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${NFL_PYTHON:-$ROOT/.venv/bin/python}"
V4="$ROOT/data/input/Clip_4 Camera Swing and Distance Change Test.mp4"
V64="$ROOT/data/input/Clip 64.mov"

if [ "$#" -lt 2 ]; then
  echo "usage: ml/run_pipeline.sh <clip4|clip64|video path> <out_dir> [extra track_clip4 args]" >&2
  exit 2
fi
TARGET="$1"
OUT="$2"
shift 2

if [ "$TARGET" = "clip4" ]; then
  VIDEO="$V4"
elif [ "$TARGET" = "clip64" ]; then
  VIDEO="$V64"
else
  VIDEO="$TARGET"
fi

mkdir -p "$OUT"
MOTION="$OUT/player_boxes.motion.npy"
bash "$ROOT/ml/run_tracker.sh" "$TARGET" "$OUT/player_boxes.csv" "$MOTION"

anchor=(--anchor -15)
for arg in "$@"; do
  if [ "$arg" = "--anchor" ]; then
    anchor=()
  fi
done

cd "$ROOT"
export PYTHONUNBUFFERED=1
set -o pipefail
"$PY" pipeline/track_clip4.py \
  --video "$VIDEO" \
  --out-dir "$OUT" \
  --player-boxes "$OUT/player_boxes.csv" \
  --camera-motion "$MOTION" \
  "${anchor[@]}" \
  "$@" \
  | awk 'BEGIN { bad = 0 } /^FATAL/ { bad = 1 } { print; fflush() } END { exit bad }'
