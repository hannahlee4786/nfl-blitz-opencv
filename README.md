# NFL Blitz Field and Camera Tracker

OpenCV prototype for detecting football-field lines, establishing a fixed field coordinate system, and tracking virtual camera movement in prerecorded NFL Blitz 2000 footage.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries:

```bash
python3 -m pip install -r requirements.txt
```

If `requirements.txt` has not been created, it should contain:

```text
opencv-python
numpy
```

## Add the Input Video

The default gameplay clip is:

```text
data/input/gameplay3.mp4
```

The initialization and tracking commands also accept a different video with
`--input`.

Run all commands from the repository’s root directory:

```text
nfl-blitz-tracker/
```

## Run Field-Line Detection

```bash
python -m src.main
```

The processed video will be saved in:

```text
data/output/detected_lines.mp4
```

## Initialize the Field Coordinate System

```bash
python -m src.initialize_homography
```

The first window is a frame browser. Move to a frame where three yard lines
and both hash rows are visible:

```text
A / D      Previous or next frame
[ / ]      Previous or next second
Enter      Use the displayed frame
Escape     Cancel
```

The yard lines do not have to be visible in frame zero. After selecting a
frame, click the requested intersections in the displayed order.

The default calibration uses the 20, 30, and 40-yard lines on the left half
of the fixed field model. To initialize the supplied cropped clip around
frame 60 using the visible 30, 40, and 50-yard lines, run:

```bash
python -m src.initialize_homography \
  --input ../cropped_clip_3.mp4 \
  --output-dir data/output/cropped_clip_3 \
  --frame-index 60 \
  --yard-lines 30 40 50 \
  --field-half left
```

Omit `--frame-index 60` to choose the frame interactively. Use
`--field-half right` when the selected markings are after midfield in the
fixed field model.

For the default calibration, select the requested points in this order:

```text
Point 1: 20-yard line × upper hash row
Point 2: 20-yard line × lower hash row

Point 3: 30-yard line × upper hash row
Point 4: 30-yard line × lower hash row

Point 5: 40-yard line × upper hash row
Point 6: 40-yard line × lower hash row
```

The points should form this arrangement:

```text
Point 1 ---------------- Point 2

Point 3 ---------------- Point 4

Point 5 ---------------- Point 6
```

Click the intersections between the three yard lines and the two columns of hash marks near the middle of the field. Do not place all six points on the same line or click the painted yard numbers.

Controls:

```text
Left click — Select a point
U          — Undo the previous point
R          — Reset all points
Enter      — Calculate the homography
Escape     — Cancel
```

The initializer will save:

```text
data/output/initial_homography.npz
data/output/initial_homography_preview.png
```

Check that the purple grid aligns with the yard lines before continuing.

## Run Field and Camera Tracking

```bash
python -m src.track_video
```

The tracked video will be saved at:

```text
data/output/tracked_field.mp4
```

For a custom clip and initialization file, run:

```bash
python -m src.track_video \
  --input ../cropped_clip_3.mp4 \
  --initialization data/output/cropped_clip_3/initial_homography.npz \
  --output data/output/cropped_clip_3/tracked_field.mp4
```

## Run Order

```bash
source .venv/bin/activate
python -m src.main
python -m src.initialize_homography
python -m src.track_video
```
