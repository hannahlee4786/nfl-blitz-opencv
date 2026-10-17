# NFL Blitz OpenCV pipeline

This branch is the OpenCV tracking pipeline. It reads a gameplay video and estimates where the camera is on the field, then where each player is standing. Game files, footage, and extracted textures are not part of this branch. You supply the video locally.

Run every command from the repository root.

## Setup
Create a virtual environment and install the two libraries the pipeline uses.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` lists `opencv-python` and `numpy`.

Put the video here:

```text
data/input/gameplay3.mp4
```

## Field and camera tracker
Three commands. The first draws the detected field lines. The second asks you to click six landmarks so the field has a coordinate system. The third follows those landmarks through the clip and redraws the grid.

```bash
python -m src.main
python -m src.initialize_homography
python -m src.track_video
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

The initializer writes `data/output/initial_homography.npz` and `data/output/initial_homography_preview.png`. The purple grid should sit on the yard lines before you continue. `python -m src.track_video` then writes `data/output/tracked_field.mp4`. It keeps a landmark only when optical flow can track it forward and back to the same pixel, and it refits the homography from the landmarks that remain.

## Player tracker
This command tracks the field from the opening frame, then keeps a number on each player for the rest of the clip. The opening roster is seven red and seven blue. A number stays with that player, and the box remains while their feet are still in the picture.

```bash
python pipeline/track_clip4.py --video data/input/gameplay3.mp4 --out-dir data/output
```

When you pass `--player-boxes`, the run builds team colors itself. It samples
upper-torso crops from large, confident boxes that do not overlap another box,
measures a jersey color for each tracker team, and writes
`<out-dir>/team_classification.json`. The team name on every box is the tracker team's display name. `team_conf` is the share of that track's confident colour frames that agree with this name, and it is 0 when the track has none. Display names default to the tracker
labels with a capital first letter, such as Red and Blue. `--team-names`
overrides those names in alphabetical order of the tracker team column, so
`--team-names Blue,Red` names the `blue` rows Blue and the `red` rows Red.

```bash
bash ml/run_pipeline.sh clip4 data/output/clip4
```

`ml/run_pipeline.sh` accepts `clip4`, `clip64`, or a video path, then an
output directory, then any extra `track_clip4.py` arguments. It writes
`player_boxes.csv`, projects those boxes, and fills team names. The default
anchor is `-15`. The script exits non-zero when the field fit prints a line
that starts with `FATAL`.

A label such as `4 Red 97% | trk 82%` shows the player ID, team name, team
confidence, and the separate tracking confidence.

`src.classify_teams` still writes a manual calibration when you want to drag
jersey patches yourself. Pass that file with `--team-calibration` and the
automatic colors are left unused:

```bash
python pipeline/track_clip4.py \
  --video data/input/gameplay3.mp4 \
  --out-dir data/output \
  --player-boxes data/output/player_boxes.csv \
  --team-calibration data/output/team_classification.json
```

`--anchor` is the yard offset on the reference frame. The default is `-15`, which places the line of scrimmage at the 50 for the clip this pipeline was built on. If your opening frame is a different yard line, set `--anchor` so the reported `x_yd` matches the painted number.

```bash
python pipeline/track_clip4.py --video data/input/gameplay3.mp4 --out-dir data/output --anchor -15
```

`--field-map` is optional. Pass a top-down field image and the video gains a map panel beside the clip. Without it, the output is the annotated clip only.

The run writes three things into `--out-dir`:

| File | Contents |
|---|---|
| `tracked_players.mp4` | Clip with boxes, player IDs, team names, team confidence, and tracking confidence |
| `positions.csv` | `frame,id,team,team_name,x_yd,y_yd,track_conf,team_conf,color_coverage` |
| `homography.npz` | Field homography for each frame, reused on the next run |
| `frames/` | Still frames from the video |

`x_yd` runs downfield. `y_yd` runs from the far sideline toward the camera.
`track_conf` is high when the detection agrees with the position predicted
from nearby frames. `team_conf` is the agreement among jersey-color pixels,
and `color_coverage` is the share of usable colorful pixels matching either
team. The first run fits the field, and a later run on the same video reuses
`homography.npz`.
