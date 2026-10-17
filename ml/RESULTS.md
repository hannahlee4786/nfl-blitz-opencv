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

The learned steps use Python 3.12 at `/Users/starfengsmac/Downloads/nfl-ml/.venv/bin/python`. The field pipeline uses Python 3.14 at `/Users/starfengsmac/Downloads/[MIDWAY] NFL Blitz 2000/.venv/bin/python`. Set `PYTHONUNBUFFERED=1` for the field script. The learned process needs:

```
export KMP_DUPLICATE_LIB_OK=TRUE
export HF_HOME=/Users/starfengsmac/Downloads/nfl-ml/models/hf
export HUGGINGFACE_HUB_CACHE=/Users/starfengsmac/Downloads/nfl-ml/models/hf/hub
export YOLO_CONFIG_DIR=/Users/starfengsmac/Downloads/nfl-ml/models/Ultralytics
export NFL_ML_CACHE=/Users/starfengsmac/Downloads/nfl-ml/cache
```

`ml/run_pipeline.sh` detects, embeds, associates, projects the boxes, and writes automatic team colors. A finished detection cache and a finished embedding cache are reused. The field script reuses `homography.npz` when that file is already in the output directory.

```
bash ml/run_pipeline.sh clip4 data/output/clip4
bash ml/run_pipeline.sh clip64 data/output/clip64
```

An arbitrary video is `bash ml/run_pipeline.sh /path/to/video.mp4 /path/to/out_dir`. The cache name is the file stem. `ml/detect_all.py` accepts `--video` for that case. Named clips `clip4` and `clip64` keep the previous video paths. Extra arguments after the output directory are passed to `pipeline/track_clip4.py`. The script already passes `--anchor -15` unless you pass `--anchor` yourself.

To rebuild the detection cache, delete `$NFL_ML_CACHE/det/<clip>_yolo11m.meta.json` and the sibling `.csv`, `.npy`, and `.reid.*` files, then run the shell script again.

## Where the artifacts live

The torch environment, YOLO weights, and Hugging Face cache are under `/Users/starfengsmac/Downloads/nfl-ml`. Detection caches are `$NFL_ML_CACHE/det/clip4_yolo11m*` and `clip64_yolo11m*`, with the default cache root `/Users/starfengsmac/Downloads/nfl-ml/cache`. The re-ID checkpoint is `/Users/starfengsmac/Downloads/nfl-ml/reid/checkpoints/best.pt`. YOLO11m weights are `/Users/starfengsmac/Downloads/nfl-ml/models/yolo11m.pt`. Training crops and the embedding run stay off Clip 4 and Clip 64. The sequence is `ml/reid/reproduce.sh`. Shipped videos, `boxes.csv`, and `positions.csv` are in `data/output/clip4/` and `data/output/clip64/`.

## Field positions

`--player-boxes` projects each box foot, then keeps the gates that reject a bad projection. A missing projection is dropped. Field y above 50 is dropped, which is the upper bound inside `_accepted_body`. The lower bound of -10 is not applied: a learned box on the quarterback lands just past that line, at the same point the classical track publishes. Foot width is the centre 40% of the bottom edge, and it still has to be at most 6.5 yards. The 0.25 yard minimum is not applied, because distant players are narrower than that. `_on_this_field` then keeps x in -15 to 125 yards, or a point within 18 yards of that frame's median x. `_clamp_step` moves an accepted point at most 6 yards from that id's previous point. A rejected point leaves `x_yd` and `y_yd` empty and stays in `boxes.csv`.

After those gates, a whole track is removed when it is a sideline cone. The track lasts at least 12 frames, and a field foot exists on at least half of them. Its median box is at most 6 percent of the frame tall and at most 4 percent of the frame wide. The median share of warm orange or red pixels in the inner crop is at least 0.30. At least half of the feet lie within 8 yards of the median foot, and those inliers have a standard deviation of at most 2.2 yards in x and in y. The median foot lies on or outside a sideline (field y at or below 1, or at or above 52.3), or it sits more than 6 yards past every player-sized track toward a sideline.

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
