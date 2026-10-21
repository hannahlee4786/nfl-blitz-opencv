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
projected-field overlay, side-by-side reconstruction preview, and evaluation
JSON. Repeating an identical command reuses completed stages.

## Full clips

`configs/clip1-full.json` to `configs/clip4-full.json` run the four full-size
clips in `data/input` (Clip 3 is `gameplay3.mp4`), with anchors made from the
camera-tracking calibration clicks.

## Field coordinates

Coordinates follow the offense, not anything painted on the field. NFL Blitz
starts every play with the camera behind the team with the ball, so x = 0 is
the end zone behind the camera on the anchor frame and x increases in the
direction the offense attacks (to x = 120). With z up, y = 0 is the sideline
on the offense's right. Every clip's camera starts with pan near 0.

```bash
nfl-reconstruct run --config configs/clip4-full.json --run-dir runs/clip4
```

## A new clip in one command

```bash
nfl-reconstruct clip "../data/input/Clip_5.mp4"
```

If the clip has no anchor yet, a frame browser opens first. Step to a frame
where the yard numbers are not covered and press Enter:

```text
A / D      Previous or next frame
[ / ]      Previous or next second
Enter      Use the displayed frame
Escape     Cancel
```

The clicking window then opens on that frame. Click the centers of
the painted 20, 30, and 40 yard numbers in the order shown: on the yard line,
between the two digits. Left click selects, `U` undoes, `R` resets, Enter
accepts, and Escape cancels. A click that disagrees with the rest is dropped.

You do not say which half of the field the numbers are on. Farther numbers sit
higher on the screen, so if the 20 is the nearest of the three it is on the
offense's own half (x = 30); if it is the farthest, it is on the half being
attacked (x = 90). Left and right come from where you click, not the order.
The command then writes `annotations/<name>-anchor.json` and
`configs/<name>.json` and runs prepare, camera, and mosaic into
`runs/<name>`. Running it again reuses the anchor and config.

| Option | Use |
| --- | --- |
| `--frame N` | Skip the browser and click on frame `N` |
| `--yard-lines 10,20,30` | Click different yard numbers |
| `--reclick` | Replace the clip's anchor; stages that depend on it rerun |
| `--name` | Name for the files; defaults to the video's name |
| `--frame-range 0:30` | Run only part of the clip |

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
4. After a turnover the camera swings around behind the new offense in a few
   frames, faster than either predictor can follow. When most tracked points
   are lost or the grid falls off the lines, the tracker circles the last good
   camera around the middle of the picture, snaps each turn onto the lines,
   and keeps the one whose paint (numbers, hash marks, sidelines) best lines
   up with the last good frame. A grid turned 180 degrees fits the yard lines
   too, so once a swing starts turning one way, only turns that keep going
   that way count. On Clip 14 (an interception) this keeps 224 of 277 frames
   reliable instead of 106.
5. A pinhole camera is solved for every frame and smoothed over time. The clip
   gets one focal length unless the fit shows the lens zooming, and frames
   with no usable grid are filled in from their neighbors. At the start or end
   of a clip, the camera coasts on for up to 10 frames and is then held.
6. Every frame is scored by its line error: the median distance, in pixels,
   from the projected yard lines to the painted ones. GOOD is under 1.5 px,
   FAIR under 4 px, and LOST beyond that or when too little of the field is in
   view. LOST frames are marked uncertain and left out of the mosaic and the
   player projection.

The stage writes `camera.jsonl`, one record per frame with the camera's
homography, line error, and how its grid was found (`snapped`, `predicted`,
or `filled`); `camera-pose.json`, the camera's position in yards, pan, tilt,
roll, and focal length per frame for other tools; and `camera-summary.json`.
`up_sign` is -1 if an anchor's sidelines are swapped, which mirrors the field
and puts the camera below it; every current clip is +1.

## Side-by-side preview

The mosaic stage automatically writes
`runs/<run-name>/mosaic/reconstruction-preview.mp4`. The left pane shows each
source frame with the estimated field grid projected over it. The right pane is
a complete fixed-field overview showing the solved camera, its visible footprint,
and its motion path. Its clip-wide orientation looks from behind the camera, so
the camera faces upward into the field in the same direction as the gameplay.

Green labels mark snapped reliable estimates. Yellow labels keep predicted,
removing those frames from the timeline, and the camera is gray in those frames.
A selected frame range produces only those selected frames while preserving the
source video's frame rate.

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
