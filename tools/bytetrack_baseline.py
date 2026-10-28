#!/usr/bin/env python3
"""Run ultralytics ByteTrack on our cached YOLO detections, for a tracker comparison.

Both trackers see the same input: the detection cache from ml/detect_all.py,
filtered by ml/track_assoc.build_frames exactly as our tracker filters it
(confidence floor, size and aspect limits, HUD mask, '1' plaque boxes dropped
before tracking, duplicate boxes on one body). ByteTrack then associates those
boxes on its own: Kalman motion and IoU, with no camera-motion compensation,
re-ID, stitching, or team logic. Each output track takes the majority colour
label of its detections, and the result is written in the boxes.csv schema
(frame,id,team,x,y,w,h,conf) so tools/eval_tracking.py can score it.

usage:
    tools/bytetrack_baseline.py --cache .nfl-ml/cache/det/clip4_yolo11m --out out/clip4/player_boxes.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import ml.track_assoc as ta  # noqa: E402

# ultralytics cfg/trackers/bytetrack.yaml in the pinned 8.3.49.
DEFAULT = {"track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.25,
           "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True}
# The same, with thresholds on our detector's score scale (ml/run_tracker.sh uses high 0.18, low 0.07).
OUR_SCALE = dict(DEFAULT, track_high_thresh=0.18, track_low_thresh=0.07, new_track_thresh=0.18)
CONFIGS = {"default": DEFAULT, "our_scale": OUR_SCALE}


class Detections:
    """The three arrays BYTETracker.update reads from an ultralytics Boxes object."""

    def __init__(self, dets: list):
        self.conf = np.array([d.conf for d in dets], dtype=np.float32)
        self.xywh = np.array([[d.x + d.w / 2, d.y + d.h / 2, d.w, d.h] for d in dets], dtype=np.float32).reshape(-1, 4)
        self.cls = np.zeros(len(dets), dtype=np.float32)


def run(cache: Path, out: Path, config: dict, params: dict) -> dict:
    from ultralytics.trackers.byte_tracker import BYTETracker

    meta, rows, hist, motion, hud, markers = ta.load_cache(cache)
    frames = ta.build_frames(meta, rows, hist, hud, params, icons=ta.load_icons(cache, len(rows)))
    fps = 30
    tracker = BYTETracker(SimpleNamespace(**config), frame_rate=fps)
    boxes = []
    for frame_index, dets in enumerate(frames):
        result = tracker.update(Detections(dets))
        for x1, y1, x2, y2, track_id, score, _cls, idx in result:
            det = dets[int(idx)]
            boxes.append({"frame": frame_index, "id": int(track_id), "x": float(x1), "y": float(y1),
                          "w": float(x2 - x1), "h": float(y2 - y1), "conf": float(score), "team_vote": det.team})
    votes = collections.defaultdict(collections.Counter)
    for b in boxes:
        votes[b["id"]][b["team_vote"]] += 1
    team_of = {tid: ("red" if c["red"] >= c["blue"] else "blue") for tid, c in votes.items()}
    width, height = int(meta["width"]), int(meta["height"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "id", "team", "x", "y", "w", "h", "conf"])
        writer.writeheader()
        for b in sorted(boxes, key=lambda b: (b["frame"], b["id"])):
            x = min(max(b["x"], 0.0), width - 1)
            y = min(max(b["y"], 0.0), height - 1)
            w = min(b["w"], width - x)
            h = min(b["h"], height - y)
            if w < 2 or h < 2:
                continue
            writer.writerow({"frame": b["frame"], "id": b["id"], "team": team_of[b["id"]], "x": round(x, 2),
                             "y": round(y, 2), "w": round(w, 2), "h": round(h, 2), "conf": round(b["conf"], 6)})
    summary = {"tracker": "bytetrack", "config": config, "cache": str(cache), "input_boxes": sum(len(f) for f in frames),
               "output_boxes": len(boxes), "ids": len(team_of)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True, help="Detection cache stem, e.g. .nfl-ml/cache/det/clip4_yolo11m")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="our_scale")
    parser.add_argument("--params", type=Path, default=None,
                        help="A player_boxes.summary.json from our tracker; its input-filter settings are reused.")
    args = parser.parse_args(argv)
    params = dict(ta.DEFAULTS)
    if args.params:
        params.update(json.loads(args.params.read_text())["params"])
    summary = run(args.cache, args.out, CONFIGS[args.config], params)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
