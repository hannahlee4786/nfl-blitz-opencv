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
  --team-names Dallas,Tampa \
  --anchor -15
```

`--team-names` is blue team first, then red. Without it the labels read Blue and Red. `--anchor` is the yard offset on the opening frame so `x_yd` matches the painted number. The default is `-15`.

The opening frame has to show yard lines the field fitter can lock onto. A clip that opens on the end-zone camera prints `FATAL: no fresh fit at reference frame` and the script exits 1.

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

## Field and camera tracker

Three commands. The first draws the detected field lines. The second asks you to click six landmarks so the field has a coordinate system. The third follows those landmarks through the clip and redraws the grid.

```bash
.venv/bin/python -m src.main
.venv/bin/python -m src.initialize_homography
.venv/bin/python -m src.track_video
```

Click the points in this order. Each pair is one yard line crossing the two hash rows.

```text
Point 1: 20-yard line, left hash
Point 2: 20-yard line, right hash
Point 3: 30-yard line, left hash
Point 4: 30-yard line, right hash
Point 5: 40-yard line, left hash
Point 6: 40-yard line, right hash
```

Left click selects a point. `U` undoes the last point. `R` clears them. Enter computes the homography. Escape cancels.

The initializer writes `data/output/initial_homography.npz` and `data/output/initial_homography_preview.png`. The purple grid should sit on the yard lines before you continue. `.venv/bin/python -m src.track_video` then writes `data/output/tracked_field.mp4`. It keeps a landmark only when optical flow can track it forward and back to the same pixel, and it refits the homography from the landmarks that remain.

## Player tracker

`ml/run_pipeline.sh` detects players, keeps an id on each person, projects the boxes onto the field, and fills team names. It accepts `clip4`, `clip64`, or a video path, then an output directory, then any extra `pipeline/track_clip4.py` arguments. The default anchor is `-15`. The script exits non-zero when the field fit prints a line that starts with `FATAL`.

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

`--field-map` is optional. Pass a top-down field image and the video gains a map panel beside the clip. Without it, the output is the annotated clip only.

The run writes these files into `--out-dir`:

| File | Contents |
|---|---|
| `tracked_players.mp4` | Clip with boxes, player IDs, team names, team confidence, and tracking confidence |
| `positions.csv` | `frame,id,team,team_name,x_yd,y_yd,track_conf,team_conf,color_coverage` |
| `homography.npz` | Field homography for each frame, reused on the next run |
| `frames/` | Still frames from the video |

`x_yd` runs downfield. `y_yd` runs from the far sideline toward the camera. `track_conf` is high when the detection agrees with the position predicted from nearby frames. `team_conf` is the agreement among jersey-color pixels, and `color_coverage` is the share of usable colorful pixels matching either team. The first run fits the field, and a later run on the same video reuses `homography.npz`.
