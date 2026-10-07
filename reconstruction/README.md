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

The current camera bootstrap composes a manual field-to-image homography with
phase-correlation translation recovered directly from consecutive RGB frames;
it does not call the existing OpenCV pipeline. Low-confidence frames are marked
uncertain and excluded from the mosaic.

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
