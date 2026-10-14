# NFL Blitz game-state reconstruction

This package is a separate workflow from the repository's existing OpenCV
experiments. It treats the NFL field as a fixed 120 by 53.333 yard world and
estimates how the source camera moves over that world. Stage artifacts are
versioned, written atomically, and reusable.

## Setup and camera/mosaic smoke run

```bash
cd reconstruction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
nfl-reconstruct run \
  --config configs/clip3.json \
  --run-dir runs/clip3-mvp \
  --stages prepare,camera,mosaic
```

The checked-in Clip 3 path is relative to the normal repository checkout. From
an isolated worktree, supply the same clip explicitly:

```bash
nfl-reconstruct run --config configs/clip3.json --run-dir runs/clip3-mvp \
  --stages prepare,camera,mosaic \
  --input /absolute/path/to/cropped_clip_3.mp4
```

Use `--frame-range 26:36` for a ten-frame bounded smoke test. The run writes a
manifest, decoded frames, camera JSONL, camera summary, static-field mosaic,
projected-field overlay, and evaluation JSON. Repeating an identical command
reuses completed stages.

## Full clips

`configs/clip1-full.json` to `configs/clip4-full.json` run the four full-size
clips in `data/input` (Clip 3 is `gameplay3.mp4`), with anchors made from the
camera-tracking calibration clicks. Clip 1 is calibrated on the far half of the
field.

```bash
nfl-reconstruct run --config configs/clip4-full.json --run-dir runs/clip4
```

## Camera stage

The camera stage uses the field tracking, line snapping, camera solve,
line-error score, and black-bar detection ported from the camera-tracking
branch. The code lives in this package; it does not call the scripts in the
repository's `src/`. A full clip takes about a minute.

1. Black bars on widescreen captures are found automatically.
   `hud_rectangles` are fractions of the game picture inside them.
2. Starting from the anchor, in both directions, optical-flow points on the
   grass, each tied to a spot on the field, predict where the field grid
   moved. When the camera moves too fast for them, ORB features matched
   between the two frames give the prediction instead. Kornia frame motion in
   `motion.py` is not used here: it follows the players when they fill the
   picture.
3. Each prediction is snapped onto the painted yard lines, so errors across
   the lines cannot build up. Along the lines, the position still comes from
   the tracked points.
4. A pinhole camera is solved for every frame and smoothed over time. The clip
   gets one focal length unless the fit shows the lens zooming, and frames
   with no usable grid are filled in from their neighbors.
5. Every frame is scored by its line error: the median distance, in pixels,
   from the projected yard lines to the painted ones. GOOD is under 1.5 px,
   FAIR under 4 px, and LOST beyond that or when too little of the field is in
   view. LOST frames are marked uncertain and left out of the mosaic and the
   player projection.

The stage writes `camera.jsonl`, one record per frame with the camera's
homography, line error, and how its grid was found (`snapped`, `predicted`,
or `filled`); `camera-pose.json`, the camera's position in yards, pan, tilt,
roll, and focal length per frame for other tools; and `camera-summary.json`.
`up_sign` is -1 when the calibration labels the sidelines so that the camera
sits on the -z side of the field, as it does for Clip 1.

## Player flow

Player detection is replaceable and deliberately downstream of camera recovery:

```bash
nfl-reconstruct export-labels --run-dir runs/clip3-mvp \
  --output annotations/clip3-players
python -m pip install -e '.[detector]'
nfl-reconstruct detect --run-dir runs/clip3-mvp \
  --weights weights/clip3-best.pt
nfl-reconstruct track --run-dir runs/clip3-mvp
nfl-reconstruct evaluate --run-dir runs/clip3-mvp
```

Detector recall cannot be accepted until a representative 50–100-frame player
annotation set and trained weights exist. Camera and mosaic stages remain fully
runnable from the checked-in field anchor without those assets.
