# Merge summary: Joyce's pipeline + camera tracking

Branch `merge/game_state_camera_calibration`, started from Joyce's latest
commit on `joyce/game-state-reconstruction`. Instead of a git merge, the
camera code from `camera-tracking` was copied into Joyce's `reconstruction/`
package.

**In short:** Joyce's pipeline is the base. Her stages, mosaic, player
detection, player tracking, and data formats are kept. Her camera stage now
runs the camera-tracking method: track points on the grass, snap the grid
onto the painted lines, solve and smooth a real camera, and score every frame
by line error.

## Kept from Joyce

These files are unchanged:

| File | What it does |
|---|---|
| `pipeline.py` | Runs the stages in order and reuses finished ones |
| `video.py` | Reads the video and saves the frames |
| `annotations.py` | Loads the hand-clicked anchor points |
| `mosaic.py` | Builds the top-down picture of the field |
| `detection.py` | YOLO player detection and training-set export |
| `tracking.py` | Follows players and puts them on the field |
| `visualization.py` | Grid overlay and player minimap |
| `optimize.py` | Her camera optimizer |
| `motion.py` | Her SIFT frame motion (kept, but the camera stage no longer uses it) |

Also kept as they were:

- the stage order: prepare, camera, mosaic, detect, track, evaluate
- the cropped Clip 3 config and anchor (`configs/clip3.json`)
- her results write-up (`docs/clip3-mvp-results.md`)
- all 47 of her tests, which still pass

## Added from camera-tracking

These are new files in `reconstruction/src/nfl_reconstruction/`, ported from
`src/` on `camera-tracking`:

| New file | Copied from | What it does |
|---|---|---|
| `point_tracking.py` | `tracking.py` | Tracks points on the grass; matches ORB features when the camera moves fast |
| `grid_tracking.py` | `track_video.py` | Follows the grid from the anchor to both ends of the clip, snapping every frame |
| `line_snapping.py` | `line_snapping.py` | Pulls the grid onto the painted yard lines |
| `line_score.py` | `line_score.py` | Line error score: GOOD under 1.5 px, FAIR under 4 px, else LOST |
| `camera_model.py` | `camera_model.py` | Turns a grid into a real camera (lens, position, angles) and back |
| `camera_solve.py` | `solve_camera.py` | Solves the camera on every frame, checks for zoom, smooths the path |
| `preprocessing.py` | `preprocessing.py` | Finds black bars and the grass in each frame |

The logic is the same as on `camera-tracking`. Only the imports and type
hints changed, and the code reads RGB frames because that's what Joyce's
video reader gives.

## Changed in Joyce's files

Small additions only. Her existing behavior stays the same where nothing
new is used.

| File | Change |
|---|---|
| `cli.py` | The camera stage now runs the steps above. The mosaic uses the black bars. The run prints the lens and line-error report. |
| `schemas.py` | Camera records gain `line_error_px` and `source` (`snapped`, `predicted`, or `filled`). `CameraState` gains `up_sign` for mirrored clips like Clip 1. |
| `camera.py` | Supports `up_sign` |
| `evidence.py` | HUD boxes are placed on the game picture, inside any black bars |
| `evaluation.py` | Adds median and 95th-percentile line error and the share of LOST frames |
| `field.py` | Adds where the yard numbers are painted (15.8 yards from each sideline) |
| `pyproject.toml` | Adds OpenCV |
| `README.md` | New "Full clips" and "Camera stage" sections |

## What replaced Joyce's camera

| | Before | After |
|---|---|---|
| Following the camera | SIFT matches between frames, chained from the anchor | Points on the grass, snapped onto the painted lines every frame |
| Camera values | Same placeholder on every frame | Real lens, position, pan, tilt, roll per frame, smoothed |
| Confidence | Mix of SIFT match quality and distance to bright line pixels | From line error: GOOD 1.0, FAIR 0.5, LOST 0.0 |
| Outputs | `camera.jsonl`, `camera-summary.json` | Same files, plus `camera-pose.json` for other tools |

Her SIFT motion couldn't stay as the predictor. When players fill the
picture it follows them instead of the field. On Clip 3 her original camera
drifted off the painted lines within about a second. With snapping added, the
grid stayed on the lines but slid 14 yards sideways along them. Snapping only
fixes errors across the yard lines, so it needs the grass-only point tracker.

## New configs and clips

- `configs/clip1-full.json` to `configs/clip4-full.json`: one run config per clip
- `annotations/clip1-full-anchor.json` to `clip4-full-anchor.json`: anchors made
  from the camera-tracking calibration clicks (Clip 1's rejected click left out)
- `data/input/`: Clips 1, 2 and 4, the same files as on `main`. Clip 3 is
  `gameplay3.mp4`.

Run a clip with:

```bash
cd reconstruction
nfl-reconstruct run --config configs/clip4-full.json --run-dir runs/clip4
```

## Results

Compared with the `camera-tracking` outputs on the same clips:

| Clip | LOST frames (merged / camera-tracking) | Lens (merged / camera-tracking) | Camera position difference |
|---|---|---|---|
| 1 | 33 / 35 | 44.7° / 44.5°, fixed | 0.12 yd median, 0.96 yd max |
| 2 | 2 / 2 | 45.6° / 45.7°, fixed | 0.05 yd median, 0.12 yd max |
| 3 | 2 / 2 | 46.4° / 46.4°, fixed | 0.08 yd median, 0.43 yd max |
| 4 | 2 / 2 | 46.5→42.4° / 46.5→42.6°, zooms | 0.13 yd median, 0.26 yd max |

The median line error is 0.0 px on every clip, and each clip takes under a
minute. For comparison, Joyce's original camera on Clip 3 had a 12.9 px median
line error, with 153 of 186 frames LOST.

## Tests

97 tests pass: Joyce's 47 plus 50 new ones. The new tests check snapping,
the camera solve (fixed lens, zoom, lost frames, mirrored clips), black bars,
point tracking, and one end-to-end run on a small generated clip.

## Not done yet

- Not committed.
- Joyce's cropped Clip 3 config wasn't re-run: that video isn't on this machine.
- Not ported: the calibration click tool, the camera video, the drift report,
  and the duplicate-frame flag.
- Player detection still needs 50 to 100 labeled frames and a trained model.

## Preview output

`mosaic.png` is one static median composite of every reliable observation warped
onto the field. Colored areas were observed by the camera; black areas were not.
It summarizes coverage and is not a video frame.

`reconstruction-preview.mp4` is the frame-by-frame evaluation output. The left
side shows the source video with the estimated field grid. The right side shows
the same frame warped onto fixed field coordinates. If calibration is working,
the right-side field stays still while the visible coverage changes as the source
camera pans or zooms. Green labels mark snapped reliable estimates; yellow labels
mark predicted, filled, uncertain, or missing estimates.

On the full Clip 3 run, 184 of 186 frames are reliable; the fixed-field pane is
intentionally black outside each frame's current camera footprint.

## Run the full Clip 3 flow

```bash
cd reconstruction
source .venv/bin/activate
nfl-reconstruct run \
  --config configs/clip3-full.json \
  --run-dir runs/clip3-full \
  --stages prepare,camera,mosaic
```

The main review files are
`runs/clip3-full/mosaic/reconstruction-preview.mp4`,
`runs/clip3-full/mosaic/mosaic.png`,
`runs/clip3-full/camera/camera-summary.json`, and
`runs/clip3-full/mosaic/evaluation.json`.
