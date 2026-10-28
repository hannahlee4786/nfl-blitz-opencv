#!/usr/bin/env python3
"""Label-free tracking statistics for one or more pipeline output folders.

Each folder is an ml/run_pipeline.sh output directory. The script reads
boxes.csv (or player_boxes.csv when boxes.csv is missing), the camera
motion the detector cached (player_boxes.motion.npy), and the associator
summary (player_boxes.summary.json). Nothing here needs ground truth.

A play has 14 players, 7 per team. The counts below compare the tracker
against that roster:

boxes/frame       boxes drawn on a frame
over_team         frames where one team has more than 7 boxes
ids               distinct track ids over the clip
short_ids         ids that live fewer than --short frames
births_after_0    ids that first appear after frame 0
gap_frames        frames inside an id's lifetime where it has no box
jumps             frame-to-frame moves larger than --jump body heights
                  after camera motion is removed; each one is a likely
                  identity switch or a box moving to another player
team_flips        frames where the published team differs from the id's
                  majority team (0 for the shipped tracker, which
                  publishes one team per id)

With --labels, the scorer in tools/eval_tracking.py also runs for that folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.eval_tracking import evaluate, load_boxes, load_labels  # noqa: E402

ROSTER_PER_TEAM = 7


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({
                "frame": int(float(row["frame"])),
                "id": int(float(row["id"])),
                "team": row["team"].strip().lower(),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "w": float(row["w"]),
                "h": float(row["h"]),
            })
    return rows


def step_matrix(motion: np.ndarray | None, frame: int) -> np.ndarray:
    if motion is None or frame <= 0 or frame >= len(motion):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return np.asarray(motion[frame], dtype=np.float64)


def compensated_jumps(rows: list[dict], motion: np.ndarray | None, jump: float) -> list[dict]:
    """Moves between consecutive frames of one id, after camera motion."""
    by_id: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    found = []
    for tid, items in by_id.items():
        items.sort(key=lambda r: r["frame"])
        for prev, cur in zip(items, items[1:]):
            if cur["frame"] != prev["frame"] + 1:
                continue
            m = step_matrix(motion, cur["frame"])
            px = prev["x"] + 0.5 * prev["w"]
            py = prev["y"] + prev["h"]
            ex = m[0, 0] * px + m[0, 1] * py + m[0, 2]
            ey = m[1, 0] * px + m[1, 1] * py + m[1, 2]
            cx = cur["x"] + 0.5 * cur["w"]
            cy = cur["y"] + cur["h"]
            height = max(20.0, 0.5 * (prev["h"] + cur["h"]))
            moved = float(np.hypot(cx - ex, cy - ey)) / height
            if moved > jump:
                found.append({"id": tid, "frame": cur["frame"], "body_heights": round(moved, 2)})
    return found


def stats_for(folder: Path, nframes: int | None, short: int, jump: float) -> dict:
    boxes = folder / "boxes.csv"
    source = "boxes.csv"
    if not boxes.is_file():
        boxes = folder / "player_boxes.csv"
        source = "player_boxes.csv (field step did not finish)"
    rows = load_rows(boxes)
    motion_path = folder / "player_boxes.motion.npy"
    motion = np.load(motion_path) if motion_path.is_file() else None
    if nframes is None:
        nframes = len(motion) if motion is not None else (max(r["frame"] for r in rows) + 1 if rows else 0)

    per_frame = Counter(r["frame"] for r in rows)
    per_frame_team: Counter = Counter((r["frame"], r["team"]) for r in rows)
    counts = np.array([per_frame.get(f, 0) for f in range(nframes)], dtype=np.float64)
    over_team = sorted({f for (f, _t), c in per_frame_team.items() if c > ROSTER_PER_TEAM})

    by_id: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    lengths = {tid: len(items) for tid, items in by_id.items()}
    spans = {tid: (min(r["frame"] for r in items), max(r["frame"] for r in items)) for tid, items in by_id.items()}
    gaps = sum((spans[t][1] - spans[t][0] + 1) - lengths[t] for t in by_id)
    teams_of = {tid: Counter(r["team"] for r in items) for tid, items in by_id.items()}
    flips = sum(sum(c.values()) - max(c.values()) for c in teams_of.values())
    ids_by_team = Counter(c.most_common(1)[0][0] for c in teams_of.values())
    jumps = compensated_jumps(rows, motion, jump)
    heights = np.array([r["h"] for r in rows]) if rows else np.zeros(1)

    result = {
        "folder": str(folder),
        "source": source,
        "frames": nframes,
        "boxes": len(rows),
        "frames_without_boxes": int((counts == 0).sum()),
        "boxes_per_frame_mean": round(float(counts.mean()), 2) if nframes else 0.0,
        "boxes_per_frame_median": float(np.median(counts)) if nframes else 0.0,
        "boxes_per_frame_min": int(counts.min()) if nframes else 0,
        "boxes_per_frame_max": int(counts.max()) if nframes else 0,
        "frames_over_14": int((counts > 2 * ROSTER_PER_TEAM).sum()),
        "frames_over_7_one_team": len(over_team),
        "box_height_median_px": round(float(np.median(heights)), 1),
        "ids": len(by_id),
        "ids_by_team": dict(sorted(ids_by_team.items())),
        "frame0_ids": int(per_frame.get(0, 0)),
        "births_after_0": sum(1 for s in spans.values() if s[0] > 0),
        "short_ids": sum(1 for n in lengths.values() if n < short),
        "track_len_median": float(np.median(list(lengths.values()))) if lengths else 0.0,
        "gap_frames": int(gaps),
        "jumps": len(jumps),
        "jump_detail": jumps[:20],
        "team_flips": int(flips),
    }
    summary_path = folder / "player_boxes.summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for key in (
            "raw_boxes", "filtered_boxes", "forward_tracklets", "stitched_chains",
            "unswaps", "repaired_handoffs", "merged_occlusion_boxes",
            "sustained_team_drift_tracks",
        ):
            if key in summary:
                result[f"assoc_{key}"] = summary[key]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label-free tracking statistics per pipeline output folder.")
    parser.add_argument("folders", nargs="+", type=Path, help="ml/run_pipeline.sh output directories.")
    parser.add_argument("--labels", action="append", default=[],
                        help="FOLDER=LABELS.json to also run the labeled scorer for that folder.")
    parser.add_argument("--short", type=int, default=15, help="An id with fewer frames than this is short.")
    parser.add_argument("--jump", type=float, default=0.8,
                        help="Camera-compensated move, in body heights per frame, counted as a jump.")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    labels_for = {}
    for item in args.labels:
        folder, _sep, path = item.partition("=")
        labels_for[Path(folder).resolve()] = Path(path)

    results = []
    for folder in args.folders:
        result = stats_for(folder, None, args.short, args.jump)
        labels = labels_for.get(folder.resolve())
        if labels is not None:
            boxes = folder / "boxes.csv"
            if not boxes.is_file():
                boxes = folder / "player_boxes.csv"
            scored = evaluate(load_labels(str(labels)), load_boxes(str(boxes)))
            result["labeled"] = {
                key: scored[key]
                for key in (
                    "frames_scored", "visible", "matches", "misses", "false_boxes",
                    "duplicates", "id_switches", "idf1", "coverage", "precision",
                    "main_accuracy",
                )
            }
        results.append(result)

    keys = [
        "frames", "boxes_per_frame_mean", "boxes_per_frame_min", "boxes_per_frame_max",
        "frames_without_boxes", "frames_over_14", "frames_over_7_one_team",
        "box_height_median_px", "ids", "frame0_ids", "births_after_0", "short_ids",
        "track_len_median", "gap_frames", "jumps", "team_flips",
    ]
    names = [r["folder"].rstrip("/").split("/")[-1] for r in results]
    width = max(22, *(len(n) + 2 for n in names))
    print("metric".ljust(26) + "".join(n.rjust(width) for n in names))
    for key in keys:
        print(key.ljust(26) + "".join(str(r.get(key, "")).rjust(width) for r in results))
    for r, name in zip(results, names):
        if r["source"] != "boxes.csv":
            print(f"{name}: read {r['source']}")
        if "labeled" in r:
            print(f"{name} labeled: " + json.dumps(r["labeled"]))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
