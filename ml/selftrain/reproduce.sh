#!/usr/bin/env bash
# End-to-end self-training: leakage split, YOLO11m teacher cache, pseudo labels,
# YOLO11n student (12 epochs), Clip-4 checkpoint/threshold selection, Clip 64 held out.
# Large artifacts: ${NFL_ML_HOME:-$ROOT/.nfl-ml}/selftrain/
# Review overlays: ${NFL_ML_HOME:-$ROOT/.nfl-ml}/scratch/pseudo_review/ and student_review/
#
# macOS has no setsid. Long jobs are started in a new session so they survive
# the launching shell, and caffeinate -i -s keeps the machine awake.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ML_HOME="${NFL_ML_HOME:-$ROOT/.nfl-ml}"
export NFL_ML_HOME="$ML_HOME"
export KMP_DUPLICATE_LIB_OK=TRUE
export HF_HOME="${HF_HOME:-$ML_HOME/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$ML_HOME/hf/hub}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ML_HOME/Ultralytics}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
PY="${NFL_PYTHON:-$ROOT/.venv/bin/python}"
ART="$ML_HOME/selftrain"
SCRATCH="$ML_HOME/scratch"
cd "$ROOT"
mkdir -p "$ART/logs" "$SCRATCH"

# Detach a command into its own session. Prints the leader PID.
# Usage: detach <pidfile> <logfile> -- <cmd...>
detach() {
  local pidfile="$1" logfile="$2"
  shift 2
  [[ "$1" == "--" ]] && shift
  "$PY" - "$pidfile" "$logfile" "$@" << 'PY'
import os, sys
pidfile, logfile, *cmd = sys.argv[1:]
if os.fork() > 0:
    raise SystemExit(0)
os.setsid()
log = open(logfile, "w")
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
devnull = open("/dev/null", "r")
os.dup2(devnull.fileno(), 0)
open(pidfile, "w").write(str(os.getpid()))
os.execvp(cmd[0], cmd)
PY
}

"$PY" -u ml/selftrain/leakage.py --mae-threshold 5.0
"$PY" -u ml/selftrain/cache_video.py --mode teacher --which trainval \
  --weights "$ROOT/models/yolo11m.pt" \
  --imgsz 640 --conf 0.05 --batch 1
"$PY" -u ml/selftrain/pseudo_labels.py --round 1 --team-source hsv
"$PY" -u ml/selftrain/pseudo_labels.py --round 1 --review-only

# Fresh 12-epoch student (skip if last.pt already exists and you only need resume).
detach "$ART/logs/train_round1.pid" "$ART/logs/train_round1.log" -- \
  caffeinate -i -s "$PY" -u ml/selftrain/train_student.py \
  --round 1 --epochs 12 --batch 8 --imgsz 640 --workers 0
echo "train pid $(cat "$ART/logs/train_round1.pid") log $ART/logs/train_round1.log"

# If training stopped after epoch 7 (last.pt epoch field == 6), resume to 12:
# detach "$ART/logs/train_round1_resume.pid" "$ART/logs/train_round1_resume.log" -- \
#   caffeinate -i -s "$PY" -u ml/selftrain/train_student.py \
#   --round 1 --epochs 12 --batch 8 --imgsz 640 --workers 0 --resume \
#   --weights "$ART/runs/round1/weights/last.pt"

# Epoch-7 checkpoint (best.pt == last.pt == epoch6.pt before resume). Clip 4 picks the threshold.
"$PY" -u ml/selftrain/evaluate.py --round 1 --stage all --reselect --tag epoch7 \
  --weights "$ART/runs/round1/weights/best.pt"

# After 12 epochs, score the finished checkpoints. Clip 4 F1 picks the winner; Clip 64 is not used to choose.
"$PY" -u ml/selftrain/evaluate.py --round 1 --stage all --reselect --tag final \
  --weights "$ART/runs/round1/weights/best.pt" "$ART/runs/round1/weights/last.pt"
