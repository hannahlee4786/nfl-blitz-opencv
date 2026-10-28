#!/bin/bash
# Run ml/run_pipeline.sh on several clips, one output folder per clip, then
# print label-free tracking statistics side by side.
# Usage: ml/run_clips.sh <out_root> [video ...]
# Without videos, every .mp4 and .mov in data/input/ is used. A file named
# "Clip_N ..." writes to <out_root>/clipN; other files use their lowercased
# stem. The Clip 4 video runs as the named target clip4, so it shares the
# detection cache and flags of `ml/run_pipeline.sh clip4`. A clip whose
# field fit fails is reported and the loop continues.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${NFL_PYTHON:-$ROOT/.venv/bin/python}"
V4="$ROOT/data/input/Clip_4 Camera Swing and Distance Change Test.mp4"
V64="$ROOT/data/input/Clip 64.mov"

if [ "$#" -lt 1 ]; then
  echo "usage: ml/run_clips.sh <out_root> [video ...]" >&2
  exit 2
fi
OUT_ROOT="$1"
shift
VIDEOS=("$@")
if [ "${#VIDEOS[@]}" -eq 0 ]; then
  shopt -s nullglob
  VIDEOS=("$ROOT"/data/input/*.mp4 "$ROOT"/data/input/*.mov)
  shopt -u nullglob
fi
if [ "${#VIDEOS[@]}" -eq 0 ]; then
  echo "no videos given and none in data/input/" >&2
  exit 2
fi

slug() {
  local stem
  stem="$(basename "$1")"
  stem="${stem%.*}"
  if [[ "$stem" =~ ^[Cc]lip[_\ ]?([0-9]+) ]]; then
    echo "clip${BASH_REMATCH[1]}"
  else
    echo "$stem" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_\n' '_'
  fi
}

mkdir -p "$OUT_ROOT"
FOLDERS=()
LABEL_ARGS=()
STATUS=()
for video in "${VIDEOS[@]}"; do
  name="$(slug "$video")"
  target="$video"
  if [ "$(cd "$(dirname "$video")" && pwd)/$(basename "$video")" = "$V4" ]; then
    target="clip4"
  elif [ "$(cd "$(dirname "$video")" && pwd)/$(basename "$video")" = "$V64" ]; then
    target="clip64"
  fi
  out="$OUT_ROOT/$name"
  mkdir -p "$out"
  echo "== $name: $(basename "$video") -> $out"
  started=$(date +%s)
  bash "$ROOT/ml/run_pipeline.sh" "$target" "$out" > "$out/run.log" 2>&1
  code=$?
  echo "   exit $code after $(( $(date +%s) - started ))s (log $out/run.log)"
  grep -E "^FATAL|sideline cones removed|automatic team colors|Team-color separation|^status:" "$out/run.log" | sed 's/^/   /'
  STATUS+=("$name=$code")
  if [ -f "$out/player_boxes.csv" ]; then
    FOLDERS+=("$out")
    if [ -f "$ROOT/data/labels/$name.json" ]; then
      LABEL_ARGS+=(--labels "$out=$ROOT/data/labels/$name.json")
    fi
  fi
done

echo "== exit codes: ${STATUS[*]}"
if [ "${#FOLDERS[@]}" -gt 0 ]; then
  "$PY" "$ROOT/tools/clip_stats.py" "${FOLDERS[@]}" "${LABEL_ARGS[@]}" --json-out "$OUT_ROOT/clip_stats.json"
fi
