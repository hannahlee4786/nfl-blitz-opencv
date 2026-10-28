# NFL Blitz OpenCV pipeline

## Quickstart

Clone the `opencv-pipeline` branch, then run setup from the repository root. `./setup.sh` creates `.venv` on Python 3.12, installs the pinned packages in `requirements.txt`, and downloads `models/yolo11m.pt`.

```bash
git clone -b opencv-pipeline git@github.com:hannahlee4786/nfl-blitz-opencv.git
cd nfl-blitz-opencv
./setup.sh
```

Run every later command from the repository root.

### Your video

Put a gameplay clip in `data/input/` and pass that path to the pipeline. Quote the path if the file name has spaces.

```bash
# drop your file here, then change the two paths
bash ml/run_pipeline.sh "data/input/YOUR_CLIP.mp4" data/output/myclip
```

The tracked video is `data/output/myclip/tracked_players.mp4`. Boxes and field positions are `boxes.csv` and `positions.csv` in the same folder.

Optional flags after the output directory:

```bash
bash ml/run_pipeline.sh "data/input/YOUR_CLIP.mp4" data/output/myclip \
  --team-names Dallas,Tampa
```

`--team-names` is blue team first, then red. Without it the labels read Blue and Red.

The clip needs a field calibration (see Field calibration below). Without one the script prints `FATAL: no field calibration` and exits 1.

## Field calibration

Player positions on the field come from a click calibration, not from automatic line detection. Do this once per clip:

```bash
.venv/bin/python -m fieldtrack.calibrate "data/input/YOUR_CLIP.mp4"
```

Click the center of the painted 20, 30 and 40 yard numbers, left then right on each line, in the order shown. Options: `--frame N` to click on a clearer frame, `--yard-lines 40 50 F40` to click other numbers, `--far-half` when the lines are on the half away from the Cowboys end zone, `--swap-sides` when the camera faces the other end zone. This writes `data/calibration/YOUR_CLIP.npz` and a preview image; check that the grid sits on the yard lines.

`ml/run_pipeline.sh` and `pipeline/track_clip4.py` pick that file up on their own (or pass `--field-calibration path.npz`). `fieldtrack/` is the field tracker and top-down map from the `player-tracking` branch. The tracked field fit is then pinned to the painted yard lines on every frame (`fieldtrack/refine.py`), so drift cannot build up, and the fit is smoothed over time. Player positions in `x_yd` / `y_yd` are the measured values, not smoothed: smoothing did not improve accuracy against hand-labeled feet. `x_smooth` / `y_smooth` are a light 5-frame smoothing, and the map in the video is drawn from those. Frames with too little field in view (close-ups) keep the plain tracker fit and show as `STALE` in the video; their positions are less reliable.

Check the fit with `.venv/bin/python tools/eval_field.py`. It scores the yard lines against hand-labeled painted lines in `data/labels/field_lines.json`, for the plain tracker and the refined fit side by side. The yard-line correction only touches downfield position; sideline-to-sideline position is exactly what the tracker measures. (A version that also corrected sideways position drifted about 2 yd laterally on stationary sideline cones, so it was removed.) `x_yd` is yards from the goal line (50 is midfield) and `y_yd` is yards from the far sideline, growing toward screen right. Calibrations for the Clip 1-4 videos are in `data/calibration/`. The field model's sidelines sit on the inner edge of the white sideline band (the `player-tracking` model put them 2-3 yd outside it); see `fieldtrack/field_model.py`. That re-anchoring assumes a yard is the same length across the field as along it.

### Labeled clips

These two files are the scored examples. Put them in `data/input/` if you have them:

```text
data/input/Clip_4 Camera Swing and Distance Change Test.mp4
data/input/Clip 64.mov
```

```bash
bash ml/run_pipeline.sh clip4 data/output/clip4
.venv/bin/python tools/eval_tracking.py --labels data/labels/clip4.json --boxes data/output/clip4/boxes.csv

bash ml/run_pipeline.sh clip64 data/output/clip64
.venv/bin/python tools/eval_tracking.py --labels data/labels/clip64.json --boxes data/output/clip64/boxes.csv
```

Game files, footage, and extracted textures stay outside the repository. You supply the video locally.

## Setup

`./setup.sh` is the install path. It uses `uv` when that command is on `PATH`, and `python3.12 -m venv` otherwise. Running it again reinstalls the same pins and skips the YOLO download when the file hash already matches.

## Player tracker

`ml/run_pipeline.sh` detects players, keeps an id on each person, projects the boxes onto the field, and fills team names. It accepts `clip4`, `clip64`, or a video path, then an output directory, then any extra `pipeline/track_clip4.py` arguments. The script exits non-zero when the field fit prints a line that starts with `FATAL`.

When you pass `--player-boxes`, the run builds team colors itself. It samples upper-torso crops from large, confident boxes that do not overlap another box, measures a jersey color for each tracker team, and writes `<out-dir>/team_classification.json`. The team name on every box is the tracker team's display name. `team_conf` is the share of that track's confident colour frames that agree with this name, and it is 0 when the track has none.

A label such as `4 Red 97% | trk 82%` shows the player ID, team name, team confidence, and the separate tracking confidence.

`src.classify_teams` still writes a manual calibration when you want to drag jersey patches yourself. Pass that file with `--team-calibration` and the automatic colors are left unused:

```bash
.venv/bin/python pipeline/track_clip4.py \
  --video "data/input/YOUR_CLIP.mp4" \
  --out-dir data/output/myclip \
  --player-boxes data/output/myclip/player_boxes.csv \
  --team-calibration data/output/myclip/team_classification.json
```

The video always shows a top-down field map beside the clip.

The run writes these files into `--out-dir`:

| File | Contents |
|---|---|
| `tracked_players.mp4` | Clip with boxes, player IDs, team names, team confidence, and tracking confidence |
| `positions.csv` | `frame,id,team,team_name,x_yd,y_yd,track_conf,team_conf,color_coverage,x_smooth,y_smooth` |
| `field.npz` | Smoothed image-to-field matrix for each frame, reused on the next run |
| `frames/` | Still frames from the video |

`x_yd` runs downfield. `y_yd` runs from the far sideline toward the camera. `track_conf` is high when the detection agrees with the position predicted from nearby frames. `team_conf` is the agreement among jersey-color pixels, and `color_coverage` is the share of usable colorful pixels matching either team. The first run fits the field, and a later run on the same video reuses `field.npz`.
