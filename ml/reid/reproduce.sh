#!/bin/bash
# Self-supervised player re-id and the Clip-4-chosen tracker.
# Detection caches and crops stay under ${NFL_ML_HOME:-$ROOT/.nfl-ml}.
# Training stays off Clip 4 and Clip 64 (the labeled test videos).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${NFL_PYTHON:-$ROOT/.venv/bin/python}"
ML_HOME="${NFL_ML_HOME:-$ROOT/.nfl-ml}"
export NFL_ML_HOME="$ML_HOME"
export KMP_DUPLICATE_LIB_OK=TRUE
export HF_HOME="${HF_HOME:-$ML_HOME/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$ML_HOME/hf/hub}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ML_HOME/Ultralytics}"
export NFL_ML_CACHE="${NFL_ML_CACHE:-$ML_HOME/cache}"
W="${NFL_REID_WEIGHTS:-$ROOT/models/reid_resnet18.pt}"
STUDENT="$ML_HOME/selftrain/runs/round1/weights/best.pt"
STU_CACHE="$ML_HOME/selftrain/student_cache/round9_imgsz640"
V4="$ROOT/data/input/Clip_4 Camera Swing and Distance Change Test.mp4"
V64="$ROOT/data/input/Clip 64.mov"
SCORE="$ML_HOME/reid/score"
cd "$ROOT"

# 1. YOLO11m test caches (skipped when meta is complete) and the student caches.
mkdir -p "$NFL_ML_CACHE/det" "$SCORE"
for clip in clip4 clip64; do
  "$PY" ml/detect_all.py --clip "$clip" --detector yolo \
    --weights "$ROOT/models/yolo11m.pt" \
    --imgsz 640 --conf 0.05 --tag yolo11m --out-dir "$NFL_ML_CACHE/det"
done
"$PY" ml/selftrain/cache_video.py --mode student --which test \
  --weights "$STUDENT" --team-source hsv --imgsz 640 --conf 0.05 --batch 4 --round 9

# 2. Crops from forward tracklets on the footage split, then the embedding.
"$PY" ml/reid/build_dataset.py
# Fresh run. Delete checkpoints/last.pt first if you do not want to resume.
"$PY" ml/reid/train.py
"$PY" ml/reid/eval_retrieval.py

# 3. Embed every test-cache box. Weights path must match the file passed to --reid.
#    embed_cache.py skips a finished npy. Delete *.reid.npy after a retrain.
for clip in clip4 clip64; do
  video="$V4"
  if [ "$clip" = clip64 ]; then video="$V64"; fi
  "$PY" ml/reid/embed_cache.py --cache "$NFL_ML_CACHE/det/${clip}_yolo11m" --video "$video" --weights "$W"
  "$PY" ml/reid/embed_cache.py --cache "$STU_CACHE/$clip" --video "$video" --weights "$W"
done

# Chosen on Clip 4 only. Same flags on Clip 64.
TRACK=(
  --low 0.07 --high 0.18 --min-hits 3 --coast 14 --stitch-gap 22 --interp-gap 10
  --contain 0.70 --hud-max 0.50 --min-h 20 --min-w 10 --max-aspect 5.2 --min-aspect 0.45
)
REID=(--reid "$W" --reid-assoc 0.35 --reid-stitch 0.8)

track_score() {
  local clip="$1" cache="$2" tag="$3"
  shift 3
  "$PY" ml/track_assoc.py --cache "$cache" --out "$SCORE/${clip}_${tag}.csv" "${TRACK[@]}" "$@"
  "$PY" tools/eval_tracking.py --labels "data/labels/${clip}.json" \
    --boxes "$SCORE/${clip}_${tag}.csv" --json-out "$SCORE/${clip}_${tag}.score.json" \
    > "$SCORE/${clip}_${tag}.txt"
}

for clip in clip4 clip64; do
  track_score "$clip" "$NFL_ML_CACHE/det/${clip}_yolo11m" yolo_off
  track_score "$clip" "$NFL_ML_CACHE/det/${clip}_yolo11m" yolo_reid "${REID[@]}"
  track_score "$clip" "$STU_CACHE/$clip" stu_off
  track_score "$clip" "$STU_CACHE/$clip" stu_reid "${REID[@]}"
done

mkdir -p "$ML_HOME/review/reid"
"$PY" ml/review_tracks.py --clip clip4 --boxes "$SCORE/clip4_yolo_reid.csv" \
  --frames 0,30,60,120,180,240 --out-dir "$ML_HOME/review/reid"
"$PY" ml/review_tracks.py --clip clip64 --boxes "$SCORE/clip64_yolo_reid.csv" \
  --frames 0,60,120,180,240,300 --out-dir "$ML_HOME/review/reid"
echo "scores in $SCORE"
