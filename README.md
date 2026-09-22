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

Place the recorded gameplay clip at:

```text
data/input/gameplay.mp4
```

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

Select the requested points in this order:

```text
Point 1: 20-yard line × left hash row
Point 2: 20-yard line × right hash row

Point 3: 30-yard line × left hash row
Point 4: 30-yard line × right hash row

Point 5: 40-yard line × left hash row
Point 6: 40-yard line × right hash row
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

## Run Order

```bash
source .venv/bin/activate
python -m src.main
python -m src.initialize_homography
python -m src.track_video
```
