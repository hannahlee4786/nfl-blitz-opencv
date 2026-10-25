The shipped player tracker is YOLO11m detections on every frame, a learned re-ID embedding, and the camera-compensated associator. `pipeline/track_clip4.py` accepts those boxes through `--player-boxes` and still tracks the field with the existing homography. On the labeled frames the shipped boxes score Clip 4 at coverage 0.744, precision 0.807, IDF1 0.568, 32 identity switches, 1 duplicate, 62 misses, 43 false boxes, and the quarterback correct on 17 of 17 frames. Clip 64 scores coverage 0.739, precision 0.789, IDF1 0.394, 36 switches, 12 duplicates, 78 misses, 59 false boxes, and the marked player correct on 14 of 32 frames. The classical pipeline on the same labels is Clip 4 IDF1 0.537, coverage 0.760, precision 0.748, 62 false boxes, and Clip 64 IDF1 0.336, coverage 0.605, precision 0.545, 151 false boxes.

| Approach | Clip 4 IDF1 | Clip 64 IDF1 | Clip 4 coverage | Clip 64 coverage | Clip 4 precision | Clip 64 precision | Clip 4 false boxes | Clip 64 false boxes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical pipeline | 0.537 | 0.336 | 0.760 | 0.605 | 0.748 | 0.545 | 62 | 151 |
| Zero-shot YOLO11m plus associator | 0.512 | 0.408 |  |  |  |  |  |  |
| Grounding DINO plus associator, about 9 s/frame | 0.550 | 0.352 |  |  |  |  |  |  |
| Self-trained YOLO11n detector, epoch 7, plus associator | 0.527 | 0.365 |  |  |  |  |  |  |
| YOLO11m plus re-ID in association (shipped) | 0.568 | 0.394 | 0.744 | 0.739 | 0.807 | 0.789 | 43 | 59 |
| Crossing resolver | 0.529 | 0.412 |  |  |  |  |  |  |

Blank cells are metrics that were not part of the recorded summary for that run. Differences under about 0.03 IDF1 sit inside the noise of a test set this small.

## How to run

`./setup.sh` creates one Python 3.12 environment at `.venv`. Detection, re-ID, association, and `pipeline/track_clip4.py` all use `${NFL_PYTHON:-$ROOT/.venv/bin/python}`. The field script runs with `PYTHONUNBUFFERED=1`.

Caches and library download folders live under `${NFL_ML_HOME:-$ROOT/.nfl-ml}`. The scripts set `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `YOLO_CONFIG_DIR` inside that directory. `NFL_ML_CACHE` defaults to `$NFL_ML_HOME/cache`.

`ml/run_pipeline.sh` detects, embeds, associates, projects the boxes, and writes automatic team colors. A finished detection cache and a finished embedding cache are reused. The field script reuses `homography.npz` when that file is already in the output directory.

```
bash ml/run_pipeline.sh clip4 data/output/clip4
bash ml/run_pipeline.sh clip64 data/output/clip64
.venv/bin/python tools/eval_tracking.py --labels data/labels/clip4.json --boxes data/output/clip4/boxes.csv
.venv/bin/python tools/eval_tracking.py --labels data/labels/clip64.json --boxes data/output/clip64/boxes.csv
```

An arbitrary video is `bash ml/run_pipeline.sh /path/to/video.mp4 /path/to/out_dir`. The cache name is the file stem. `ml/detect_all.py` accepts `--video` for that case. Named clips `clip4` and `clip64` keep the previous video paths. Extra arguments after the output directory are passed to `pipeline/track_clip4.py`. The script already passes `--anchor -15` unless you pass `--anchor` yourself.

To rebuild the detection cache, delete `$NFL_ML_CACHE/det/<clip>_yolo11m.meta.json` and the sibling `.csv`, `.npy`, and `.reid.*` files, then run the shell script again.

## Where the artifacts live

The virtual environment is `.venv`. YOLO11m weights are `models/yolo11m.pt` (downloaded by `setup.sh`). The re-ID weights shipped with the repo are `models/reid_resnet18.pt`. Detection caches are `$NFL_ML_HOME/cache/det/clip4_yolo11m*` and `clip64_yolo11m*`. Training crops and the embedding run stay off Clip 4 and Clip 64. The sequence is `ml/reid/reproduce.sh`, and its outputs stay under `$NFL_ML_HOME`. Shipped `boxes.csv` and `positions.csv` are in `data/output/clip4/` and `data/output/clip64/`. Input videos are gitignored and belong in `data/input/`. Footage for self-training defaults to `data/drive-download-*/`, or to `NFL_FOOTAGE_DIR` when that variable is set.

## Field positions

`--player-boxes` projects each box foot, then keeps the gates that reject a bad projection. A missing projection is dropped. Field y above 50 is dropped, which is the upper bound inside `_accepted_body`. The lower bound of -10 is not applied: a learned box on the quarterback lands just past that line, at the same point the classical track publishes. Foot width is the centre 40% of the bottom edge, and it still has to be at most 6.5 yards. The 0.25 yard minimum is not applied, because distant players are narrower than that. `_on_this_field` then keeps x in -15 to 125 yards, or a point within 18 yards of that frame's median x. `_clamp_step` moves an accepted point at most 6 yards from that id's previous point. A rejected point leaves `x_yd` and `y_yd` empty and stays in `boxes.csv`.

After those gates, a whole track is removed when it is a sideline cone. The track lasts at least 12 frames. Its median box is at most 6 percent of the frame tall and at most 4 percent of the frame wide, and the median share of warm orange or red pixels in the inner crop is at least 0.30. The detector's camera-motion transforms map each foot back to frame zero; at least 60 percent must remain within 30 pixels of the median, with a median distance no greater than 18 pixels and inlier spread no greater than 24 pixels horizontally and 14 vertically. This camera-fixed test still works when field homographies are stale. When camera motion is unavailable, the older ground-plane boundary and formation tests remain as a fallback.

On Clip 4, id 9 (the quarterback) has a yard point on all 54 frames from 109 to 162. Frame 109 is x_yd 74.45, beside the classical main player at 74.7.

| | Learned Clip 4 | Classical Clip 4 | Learned Clip 64 | Classical Clip 64 |
| --- | --- | --- | --- | --- |
| ids | 22 | 13 | 30 | 13 |
| max step | 6.005 yd, id 2, frames 176 to 177 | 6.006 yd, id 7, frames 72 to 73 | 6.008 yd, id 26, frames 345 to 346 | 6.011 yd, id 5, frames 352 to 353 |
| steps over 8 yd | 0 | 0 | 0 | 0 |
| empty positions | 3 of 2270 (id 5) | 0 of 2447 | 110 of 2873 (ids 13, 26, 29, 30) | 0 of 3809 |
| x_yd | 13.98 to 81.97 | 14.58 to 81.53 | -41.43 to 63.25 | -86.33 to 58.28 |
| y_yd | -38.59 to 34.52 | -34.7 to 34.61 | -21.86 to 49.92 | -43.37 to 44.27 |

## Known limitations

Frame 0 of Clip 4 has 11 boxes. Thirteen players are visible at the snap, and the interior linemen merge into shared boxes. Track ids continue past that opening roster, so ids above 11 appear later in the clip. Identity swaps show up when players are in contact. On the Clip 4 stills at frames 0, 120, 180, and 240, every box sits on a player, including players cut off at the screen edges, and the scoreboard, banners, INCOMPLETE graphic, and KAG logo carry no box. The labeled test is 242 boxes on Clip 4 and 299 boxes on Clip 64, so an IDF1 gap under about 0.03 is within noise.

## Next steps

Exact player labels from the game's memory can train the detector and the embedding offline, and the same readout can build a large automatic test set. Heavier detectors and longer embedding training want a cloud GPU.
