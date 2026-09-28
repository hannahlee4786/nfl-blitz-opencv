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

`--anchor` is the yard offset on the reference frame. The default is `-15`, which places the line of scrimmage at the 50 for the clip this pipeline was built on. If your opening frame is a different yard line, set `--anchor` so the reported `x_yd` matches the painted number.

```bash
python pipeline/track_clip4.py --video data/input/gameplay3.mp4 --out-dir data/output --anchor -15
```

`--field-map` is optional. Pass a top-down field image and the video gains a map panel beside the clip. Without it, the output is the annotated clip only.

The run writes three things into `--out-dir`:

| File | Contents |
|---|---|
| `tracked_players.mp4` | Clip with boxes, player numbers, and confidence |
| `positions.csv` | `frame,id,team,x_yd,y_yd,conf` |
| `homography.npz` | Field homography for each frame, reused on the next run |
| `frames/` | Still frames from the video |

`x_yd` runs downfield. `y_yd` runs from the far sideline toward the camera. `conf` is high when the detection agrees with the position predicted from the previous frame. The first run fits the field, and a later run on the same video reuses `homography.npz`.
