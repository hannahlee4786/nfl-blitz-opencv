# Clip 3 reconstruction verification

Verified on 2026-10-06 from the isolated `joyce/game-state-reconstruction`
worktree. The workflow does not import or call the repository's existing OpenCV
pipeline.

## Commands

The clean environment and automated suite were run with:

```bash
python3.11 -m venv /tmp/nfl-blitz-reconstruction-verify
/tmp/nfl-blitz-reconstruction-verify/bin/python -m pip install -e '.[test]'
/tmp/nfl-blitz-reconstruction-verify/bin/python -m pytest -q
```

The bounded real-video run was regenerated from scratch with:

```bash
/tmp/nfl-blitz-reconstruction-verify/bin/nfl-reconstruct run \
  --config configs/clip3.json \
  --run-dir /tmp/nfl-blitz-clip3-verified-final \
  --stages prepare,camera,mosaic \
  --frame-range 26:86 \
  --input /Users/joyceng/Desktop/Desktop/csci401/cropped_clip_3.mp4
```

The explicit input override is needed only because verification ran from a
temporary worktree. The checked-in relative input path targets the normal
repository layout.

## Automated result

- 47 tests passed in 4.10 seconds in the final clean-environment run.
- The run decoded exactly 60 requested frames and schema validation passed for
  the run manifest, frame manifest, and all 60 camera records.
- Camera statuses: 60 reliable, 0 uncertain, 0 missing.
- All output images and the compressed mosaic arrays opened successfully with
  their expected dimensions.
- Re-running the same configuration reuses atomic stage outputs; package source
  code participates in each stage input hash, so code changes invalidate stale
  artifacts.

Environment: Python 3.11.5, PyAV 13.1.0, Kornia 0.8.3, NumPy 2.4.6,
Pillow 12.3.0, Pydantic 2.13.5, SciPy 1.17.1, PyTorch 2.14.1, Typer
0.27.3, and pytest 8.4.2.

## Measured camera and mosaic results

| Metric | Result |
| --- | ---: |
| Seed-anchor reprojection error | 4.5655 px |
| Median field-evidence distance | 28.6598 px |
| P95 field-evidence distance | 36.3528 px |
| Camera trajectory second-difference | 22.6055 px/frame² |
| Static-field mosaic coverage | 19.9511% |
| Maximum observations at one mosaic pixel | 56 |

The seed anchor meets the planned sub-5-pixel criterion. Non-anchor “reprojection
error” records measure distance to automatically extracted line evidence, not
manual ground-truth correspondences, so those values must not be interpreted as
annotated reprojection accuracy.

The motion model uses Kornia SIFT correspondences with RANSAC to estimate a full
projective transform between adjacent frames, composed onto the manual
field-to-image anchor. Phase correlation is retained as a bounded translation
fallback. This increased accepted camera frames from 5/60 in the discarded
translation-only trial to 60/60 and increased mosaic coverage from 7.9431% to
19.9511%.

## Manual inspection

The four-frame overlay contact sheet keeps the projected regulation-field lines
attached to the visible field as the broadcast camera pans and zooms. The mosaic
shows coherent, recognizable yard markings and sideline geometry in one static
field coordinate system. This is a material improvement over translation-only
registration.

The visual-stability criterion is only partially accepted: the center of the
reconstructed region is coherent, but the 22.6 px/frame² motion smoothness value
and small edge discontinuities show residual frame-to-frame jitter. Persistent
players also leave ghosts because no trained player mask is available. A second
manual anchor later in the clip plus temporal homography smoothing are the next
steps before treating the camera path as production-quality ground truth.

Artifacts inspected:

- `/tmp/nfl-blitz-clip3-verified-final/mosaic/overlay-contact-sheet.png`
- `/tmp/nfl-blitz-clip3-verified-final/mosaic/mosaic.png`
- `/tmp/nfl-blitz-clip3-verified-final/mosaic/minimap.png`
- `/tmp/nfl-blitz-clip3-verified-final/evaluation.json`

The minimap is intentionally empty because this run excludes the detector and
tracking stages.

## Data-gated criteria

Real-video player detection recall, identity fragmentation, projected player
trajectories, and player-masked mosaic quality remain unmeasured. They require a
representative 50–100-frame offense/defense annotation set and trained detector
weights. The detector adapter, field projection, short-occlusion association,
dataset export, and synthetic tests are implemented, but no claim about Clip 3
player accuracy is made without those assets.
