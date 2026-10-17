#!/usr/bin/env python3
"""Turn teacher detections into full-body, team-labeled pseudo boxes.

Temporal tracklets (ml.track_assoc) drop isolated false detections and fill
short gaps. Each kept box is extended to the head and feet by walking a
narrow column until the grass (or the HUD) resumes, then the extension is
median-smoothed along the tracklet. Team is a majority jersey vote on the
real (non-interpolated) boxes of that tracklet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2
import numpy as np

from ml.track_assoc import DEFAULTS, Det, hud_fraction, run as assoc_run, suppress_duplicates
from tools.eval_tracking import box_iou
from ml.selftrain.common import (
    ART,
    SCRATCH,
    SPLITS,
    THUMB_H,
    THUMB_W,
    clip_name,
    ensure_dirs,
    footage_path,
)

MIN_TRACK = 12
MIN_REAL = 5
HEAD_CAP = 0.60
FOOT_CAP = 1.35
HEIGHT_CAP = 2.60
TRAIN_TARGET = 2000
VAL_TARGET = 320
DEDUP_GAP = 4
DEDUP_MAE = 4.0
MIN_BOXES_PER_FRAME = 4
ASSOC_OVERRIDES = {
    "interp_gap": 12,
    "hud_max": 0.45,
    "min_h": 22.0,
    "min_w": 12.0,
    # 0.85 keeps two overlapping linemen; 0.70 was merging them when teams agreed.
    "contain": 0.85,
    "min_aspect": 0.40,
    "max_aspect": 4.8,
}
CSV_FIELDS = [
    "frame",
    "id",
    "team",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "interpolated",
    "ox",
    "oy",
    "ow",
    "oh",
]


def assoc_params() -> dict:
    params = dict(DEFAULTS)
    params.update(ASSOC_OVERRIDES)
    return params


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def frow(row: dict) -> dict:
    return {
        "frame": int(row["frame"]),
        "id": int(float(row["id"])),
        "team": str(row["team"]).strip().lower(),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "w": float(row["w"]),
        "h": float(row["h"]),
        "conf": float(row["conf"]),
    }


def count_raw_filters(rows: list[dict], hud: np.ndarray, width: int, height: int, params: dict) -> dict:
    counts = {
        "raw": len(rows),
        "conf": 0,
        "size": 0,
        "aspect": 0,
        "huge": 0,
        "hud": 0,
        "containment": 0,
    }
    kept: list[Det] = []
    for index, row in enumerate(rows):
        item = frow(row)
        if item["conf"] < params["low"]:
            counts["conf"] += 1
            continue
        if item["h"] < params["min_h"] or item["w"] < params["min_w"]:
            counts["size"] += 1
            continue
        aspect = item["h"] / max(item["w"], 1e-6)
        if aspect < params["min_aspect"] or aspect > params["max_aspect"]:
            counts["aspect"] += 1
            continue
        if item["h"] > 0.9 * height or item["w"] > 0.7 * width:
            counts["huge"] += 1
            continue
        det = Det(index, item["frame"], item["x"], item["y"], item["w"], item["h"], item["conf"], item["team"], None)
        if hud_fraction(hud, det, width, height) > params["hud_max"]:
            counts["hud"] += 1
            continue
        kept.append(det)
    by_frame: dict[int, list[Det]] = defaultdict(list)
    for det in kept:
        by_frame[det.frame].append(det)
    after = 0
    for dets in by_frame.values():
        survived = suppress_duplicates(dets, params["contain"])
        counts["containment"] += len(dets) - len(survived)
        after += len(survived)
    counts["after_frame_filters"] = after
    return counts


def jersey_feature(bgr: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Fraction of torso pixels that are red, blue, or white/silver."""
    height, width = bgr.shape[:2]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(3, np.float32)
    roi = bgr[y1:y2, x1:x2]
    top = int(roi.shape[0] * 0.12)
    bot = max(top + 1, int(roi.shape[0] * 0.62))
    torso = roi[top:bot]
    if torso.size == 0:
        return np.zeros(3, np.float32)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colorful = (sat >= 50) & (val >= 40)
    red = colorful & ((hue <= 15) | (hue >= 165))
    blue = colorful & (hue >= 90) & (hue <= 135)
    white = (sat <= 45) & (val >= 160)
    return np.array([float(red.mean()), float(blue.mean()), float(white.mean())], np.float32)


def cluster_centers(features: np.ndarray):
    """Two jersey clusters. Returns (red_center, blue_center) or None."""
    if len(features) < 8:
        return None
    score = features[:, 0] - features[:, 1]
    red_center = features[int(score.argmax())].astype(np.float64).copy()
    blue_center = features[int(score.argmin())].astype(np.float64).copy()
    labels = np.zeros(len(features), np.int32)
    for _ in range(12):
        dist_red = ((features - red_center) ** 2).sum(axis=1)
        dist_blue = ((features - blue_center) ** 2).sum(axis=1)
        labels = (dist_blue + 1e-8 < dist_red).astype(np.int32)
        if int(labels.min()) == int(labels.max()):
            return None
        red_center = features[labels == 0].mean(axis=0)
        blue_center = features[labels == 1].mean(axis=0)
    score0 = float((features[labels == 0, 0] - features[labels == 0, 1]).mean())
    score1 = float((features[labels == 1, 0] - features[labels == 1, 1]).mean())
    if abs(score0 - score1) < 0.04:
        return None
    if score0 >= score1:
        return red_center, blue_center
    return blue_center, red_center


def reteam_by_cluster(video: Path, rows: list[dict]) -> dict:
    """Give each detection the clip's red or other-team cluster, then keep HSV if the clip does not split."""
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_frame[int(float(row["frame"]))].append(index)
    features = np.zeros((len(rows), 3), np.float32)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    index = 0
    last = max(by_frame) if by_frame else -1
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for row_index in by_frame.get(index, []):
            item = frow(rows[row_index])
            features[row_index] = jersey_feature(frame, item["x"], item["y"], item["w"], item["h"])
        index += 1
    cap.release()
    fit = [i for i, row in enumerate(rows) if float(row["conf"]) >= 0.12]
    centers = cluster_centers(features[fit]) if len(fit) >= 8 else None
    if centers is None:
        return {"mode": "hsv", "reason": "clusters did not separate"}
    red_center, blue_center = centers
    changed = 0
    n_red = 0
    n_blue = 0
    for row_index, row in enumerate(rows):
        feat = features[row_index]
        dist_red = float(((feat - red_center) ** 2).sum())
        dist_blue = float(((feat - blue_center) ** 2).sum())
        team = "red" if dist_red <= dist_blue else "blue"
        if row["team"] != team:
            changed += 1
        row["team"] = team
        n_red += int(team == "red")
        n_blue += int(team == "blue")
    return {
        "mode": "cluster",
        "changed_from_hsv": changed,
        "n_red": n_red,
        "n_blue": n_blue,
        "red_center": [round(float(v), 4) for v in red_center],
        "blue_center": [round(float(v), 4) for v in blue_center],
    }


def smooth_teams(rows: list[dict]) -> int:
    """Flip isolated HSV mistakes when both neighboring frames agree."""
    parsed = [frow(row) for row in rows]
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(parsed):
        by_frame[item["frame"]].append(index)
    flips = 0
    for _pass in range(2):
        for frame, indices in by_frame.items():
            for index in indices:
                item = parsed[index]
                votes = [item["team"]]
                box = (item["x"], item["y"], item["w"], item["h"])
                for other_frame in (frame - 1, frame + 1):
                    best_iou = 0.0
                    best_team = None
                    for other_index in by_frame.get(other_frame, []):
                        other = parsed[other_index]
                        iou = box_iou(box, (other["x"], other["y"], other["w"], other["h"]))
                        if iou > best_iou:
                            best_iou = iou
                            best_team = other["team"]
                    if best_team is not None and best_iou >= 0.40:
                        votes.append(best_team)
                red = votes.count("red")
                blue = votes.count("blue")
                if red == blue:
                    continue
                majority = "red" if red > blue else "blue"
                if majority != item["team"]:
                    item["team"] = majority
                    flips += 1
    for row, item in zip(rows, parsed):
        row["team"] = item["team"]
    return flips


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def publish_smoothed(src_stem: Path, dst_stem: Path, rows: list[dict]) -> None:
    dst_stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        dst_stem.with_suffix(".csv"),
        rows,
        ["frame", "id", "team", "x", "y", "w", "h", "conf"],
    )
    for suffix in (".hist.npy", ".motion.npy", ".hud.npy", ".markers.csv", ".meta.json"):
        link_or_copy(src_stem.with_suffix(suffix), dst_stem.with_suffix(suffix))


def grass_rows(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = (hue >= 28) & (hue <= 95) & (sat >= 22) & (val >= 18)
    return (~grass).mean(axis=1)


def row_is_hud(hud: np.ndarray, y: int, height: int, x1: int, x2: int, width: int) -> bool:
    if hud is None or hud.size == 0:
        return False
    mask_h, mask_w = hud.shape[:2]
    yy = int(np.clip(y / max(height, 1) * mask_h, 0, mask_h - 1))
    xa = int(np.clip(x1 / max(width, 1) * mask_w, 0, mask_w - 1))
    xb = int(np.clip(x2 / max(width, 1) * mask_w, xa + 1, mask_w))
    return float((hud[yy, xa:xb] > 0).mean()) > 0.55


def column_pads(bgr: np.ndarray, hud: np.ndarray, x: float, y: float, w: float, h: float) -> tuple[float, float]:
    """How far the body continues above and below the teacher box, in pixels."""
    height, width = bgr.shape[:2]
    shrink = 0.14
    xa = int(np.clip(round(x + shrink * w), 0, width - 2))
    xb = int(np.clip(round(x + (1.0 - shrink) * w), xa + 1, width))
    y_top = int(max(0, round(y - HEAD_CAP * h)))
    y_bot = int(min(height, round(y + h + FOOT_CAP * h)))
    if y_bot - y_top < 4 or xb - xa < 2:
        return 0.0, 0.0
    frac = grass_rows(bgr[y_top:y_bot, xa:xb])
    top_local = int(np.clip(round(y) - y_top, 0, len(frac) - 1))
    bot_local = int(np.clip(round(y + h) - 1 - y_top, 0, len(frac) - 1))

    def walk(start: int, step: int) -> int:
        last = start
        gap = 0
        cursor = start
        while 0 <= cursor < len(frac):
            if row_is_hud(hud, y_top + cursor, height, xa, xb, width):
                break
            if frac[cursor] >= 0.30:
                last = cursor
                gap = 0
            else:
                gap += 1
                if gap > 3:
                    break
            nxt = cursor + step
            if nxt < 0 or nxt >= len(frac):
                break
            cursor = nxt
        return last

    new_top = walk(top_local, -1)
    new_bot = walk(bot_local, 1)
    head = float(np.clip(y - (y_top + new_top), 0.0, HEAD_CAP * h))
    foot = float(np.clip((y_top + new_bot + 1) - (y + h), 0.0, FOOT_CAP * h))
    return head, foot


def median_smooth(values: list[float], frames: list[int], win: int = 5) -> list[float]:
    out = list(values)
    radius = win // 2
    start = 0
    for index in range(1, len(frames) + 1):
        boundary = index == len(frames) or frames[index] - frames[index - 1] > 2
        if not boundary:
            continue
        for cursor in range(start, index):
            lo = max(start, cursor - radius)
            hi = min(index, cursor + radius + 1)
            out[cursor] = float(np.median(values[lo:hi]))
        start = index
    return out


def nongrass_fraction(bgr: np.ndarray, x: float, y: float, w: float, h: float) -> float:
    height, width = bgr.shape[:2]
    x1 = max(0, int(round(x + 0.1 * w)))
    y1 = max(0, int(round(y + 0.05 * h)))
    x2 = min(width, int(round(x + 0.9 * w)))
    y2 = min(height, int(round(y + 0.95 * h)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = (hue >= 28) & (hue <= 95) & (sat >= 22) & (val >= 18)
    return float((~grass).mean())


def clamp_box(x: float, y: float, w: float, h: float, width: int, height: int):
    x2 = min(float(width), max(0.0, x + w))
    y2 = min(float(height), max(0.0, y + h))
    x1 = min(float(width), max(0.0, x))
    y1 = min(float(height), max(0.0, y))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2 - x1, y2 - y1


def raw_overlap_index(raw_rows: list[dict], low: float) -> dict[int, list[tuple]]:
    by_frame: dict[int, list[tuple]] = defaultdict(list)
    for row in raw_rows:
        item = frow(row)
        if item["conf"] < low:
            continue
        by_frame[item["frame"]].append((item["x"], item["y"], item["w"], item["h"]))
    return by_frame


def is_interpolated(box: tuple, raw_on_frame: list[tuple]) -> bool:
    best = 0.0
    for other in raw_on_frame:
        best = max(best, box_iou(box, other))
        if best >= 0.5:
            return False
    return True


def process_clip(
    number: int,
    teacher_dir: Path,
    round_n: int,
    team_source: str,
    params: dict,
) -> dict:
    stem = teacher_dir / clip_name(number)
    if not stem.with_suffix(".meta.json").is_file():
        raise RuntimeError(f"missing teacher cache {stem}")
    meta = json.loads(stem.with_suffix(".meta.json").read_text())
    width, height = int(meta["width"]), int(meta["height"])
    raw_rows = read_csv(stem.with_suffix(".csv"))
    hud = np.load(stem.with_suffix(".hud.npy"))
    filters = count_raw_filters(raw_rows, hud, width, height, params)
    smoothed = [dict(row) for row in raw_rows]
    if team_source == "hsv":
        cluster_info = reteam_by_cluster(footage_path(number), smoothed)
    else:
        cluster_info = {"mode": "class"}
    flips = smooth_teams(smoothed)
    smooth_stem = ART / "teacher_smoothed" / f"round{round_n}" / clip_name(number)
    publish_smoothed(stem, smooth_stem, smoothed)
    track_csv = ART / "tracks" / f"round{round_n}" / f"{clip_name(number)}.csv"
    summary_path = track_csv.with_suffix(".summary.json")
    rerun = True
    if summary_path.is_file() and track_csv.is_file():
        saved = json.loads(summary_path.read_text())
        rerun = json.loads(json.dumps(saved.get("params"))) != json.loads(json.dumps(params))
    if rerun:
        assoc_run(smooth_stem, track_csv, params)
    else:
        print(f"assoc cache hit {track_csv}", flush=True)
    tracks = [frow(row) for row in read_csv(track_csv)]
    raw_index = raw_overlap_index(raw_rows, params["low"])
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for item in tracks:
        box = (item["x"], item["y"], item["w"], item["h"])
        item["interpolated"] = is_interpolated(box, raw_index.get(item["frame"], []))
        by_frame[item["frame"]].append(item)

    video = footage_path(number)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    records: list[dict] = []
    frame_index = 0
    needed = set(by_frame)
    last = max(needed) if needed else -1
    while frame_index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for item in by_frame.get(frame_index, []):
            head, foot = column_pads(frame, hud, item["x"], item["y"], item["w"], item["h"])
            # Team was already set on the detection (jersey cluster, or the student class).
            # A fresh HSV call collapses silver/white jerseys back to red.
            vote = None if item["interpolated"] else item["team"]
            records.append(
                {
                    "frame": frame_index,
                    "id": item["id"],
                    "conf": item["conf"],
                    "fallback_team": item["team"],
                    "interpolated": bool(item["interpolated"]),
                    "ox": item["x"],
                    "oy": item["y"],
                    "ow": item["w"],
                    "oh": item["h"],
                    "head": head,
                    "foot": foot,
                    "vote": vote,
                    "nongrass": nongrass_fraction(frame, item["x"], item["y"], item["w"], item["h"]),
                }
            )
        frame_index += 1
    cap.release()

    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["id"]].append(record)
    dropped_short = 0
    dropped_few_real = 0
    dropped_ambiguous = 0
    kept_records: list[dict] = []
    for recs in grouped.values():
        recs.sort(key=lambda record: record["frame"])
        real = sum(1 for record in recs if not record["interpolated"])
        if len(recs) < MIN_TRACK:
            dropped_short += len(recs)
            continue
        if real < MIN_REAL:
            dropped_few_real += len(recs)
            continue
        frames = [record["frame"] for record in recs]
        heads = median_smooth([record["head"] for record in recs], frames)
        feet = median_smooth([record["foot"] for record in recs], frames)
        # Majority of per-box teams. For HSV clips those teams are the clip's
        # two jersey clusters (red = the redder cluster), not a fresh HSV call.
        votes = [record["vote"] for record in recs if record["vote"] in ("red", "blue")]
        red = votes.count("red")
        blue = votes.count("blue")
        if red > blue:
            team = "red"
        elif blue > red:
            team = "blue"
        else:
            team = recs[0]["fallback_team"]
        for record, head, foot in zip(recs, heads, feet):
            record["head"] = head
            record["foot"] = foot
            record["team"] = team
            kept_records.append(record)

    dropped_tiny = 0
    dropped_hud = 0
    dropped_grass = 0
    preliminary = []
    ratios = []
    for record in kept_records:
        y = record["oy"] - record["head"]
        h = record["oh"] + record["head"] + record["foot"]
        if h > HEIGHT_CAP * record["oh"]:
            h = HEIGHT_CAP * record["oh"]
        clamped = clamp_box(record["ox"], y, record["ow"], h, width, height)
        if clamped is None or clamped[3] < 26 or clamped[2] < 12:
            dropped_tiny += 1
            continue
        if float(record.get("nongrass") or 0.0) < 0.22:
            dropped_grass += 1
            continue
        x, y, w, h = clamped
        det = Det(0, record["frame"], x, y, w, h, record["conf"], record["team"], None)
        if hud_fraction(hud, det, width, height) > params["hud_max"]:
            dropped_hud += 1
            continue
        ratios.append(h / max(record["oh"], 1e-6))
        preliminary.append(
            {
                "frame": record["frame"],
                "id": record["id"],
                "team": record["team"],
                "x": round(x, 2),
                "y": round(y, 2),
                "w": round(w, 2),
                "h": round(h, 2),
                "conf": round(float(record["conf"]), 6),
                "interpolated": int(record["interpolated"]),
                "ox": round(record["ox"], 2),
                "oy": round(record["oy"], 2),
                "ow": round(record["ow"], 2),
                "oh": round(record["oh"], 2),
            }
        )

    by_out: dict[int, list[dict]] = defaultdict(list)
    for row in preliminary:
        by_out[row["frame"]].append(row)
    dropped_wide = 0
    final = []
    for frame, boxes in by_out.items():
        median_w = float(np.median([box["w"] for box in boxes])) if boxes else 0.0
        for box in boxes:
            aspect = box["h"] / max(box["w"], 1e-6)
            if median_w > 0 and box["w"] > 1.85 * median_w and aspect < 1.15:
                dropped_wide += 1
                continue
            final.append(box)
    final.sort(key=lambda row: (row["frame"], row["id"]))
    out_csv = ART / "pseudo" / f"round{round_n}" / f"{clip_name(number)}.csv"
    write_csv(out_csv, final, CSV_FIELDS)
    interp = sum(int(row["interpolated"]) for row in final)
    summary = {
        "clip": number,
        "filters": filters,
        "team_flips_before_assoc": flips,
        "assoc_boxes": len(tracks),
        "assoc_interpolated": sum(1 for item in tracks if item["interpolated"]),
        "dropped_short_track": dropped_short,
        "dropped_few_real": dropped_few_real,
        "dropped_ambiguous_team": dropped_ambiguous,
        "dropped_tiny": dropped_tiny,
        "dropped_grass": dropped_grass,
        "team_cluster": cluster_info,
        "dropped_hud_after_expand": dropped_hud,
        "dropped_wide_merge": dropped_wide,
        "final_boxes": len(final),
        "final_interpolated": interp,
        "final_frames": len({row["frame"] for row in final}),
        "red": sum(1 for row in final if row["team"] == "red"),
        "blue": sum(1 for row in final if row["team"] == "blue"),
        "median_height_ratio": round(float(np.median(ratios)), 4) if ratios else None,
        "csv": str(out_csv),
    }
    print(
        f"clip {number}: raw {filters['raw']} final {len(final)} "
        f"interp {interp} height x{summary['median_height_ratio']} "
        f"red {summary['red']} blue {summary['blue']}",
        flush=True,
    )
    return summary


def thumb(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def boxes_by_frame(csv_path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    if not csv_path.is_file():
        return grouped
    for row in read_csv(csv_path):
        item = frow(row)
        item["interpolated"] = int(row.get("interpolated") or 0)
        item["ox"] = float(row["ox"])
        item["oy"] = float(row["oy"])
        item["ow"] = float(row["ow"])
        item["oh"] = float(row["oh"])
        grouped[item["frame"]].append(item)
    return grouped


def select_frames(video: Path, candidates: list[int]) -> list[int]:
    if not candidates:
        return []
    wanted = set(candidates)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    kept: list[int] = []
    last_thumb = None
    last_index = -10**9
    index = 0
    last = max(wanted)
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            small = thumb(frame)
            mae = 1e9
            if last_thumb is not None:
                mae = float(np.abs(small.astype(np.int16) - last_thumb.astype(np.int16)).mean())
            if last_thumb is None or index - last_index >= DEDUP_GAP or mae >= DEDUP_MAE:
                kept.append(index)
                last_thumb = small
                last_index = index
        index += 1
    cap.release()
    return kept


def subsample(pairs: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    if len(pairs) <= target:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, target)
    chosen = []
    seen = set()
    for index in indices:
        key = int(round(index))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(pairs[key])
    return chosen


def clip_xywh(x, y, w, h, width, height):
    return clamp_box(x, y, w, h, width, height)


def yolo_lines(boxes: list[dict], width: int, height: int, swap: bool) -> list[str]:
    lines = []
    for box in boxes:
        clamped = clip_xywh(box["x"], box["y"], box["w"], box["h"], width, height)
        if clamped is None:
            continue
        x, y, w, h = clamped
        if w < 4 or h < 8:
            continue
        cls = 0 if box["team"] == "red" else 1
        if swap:
            cls = 1 - cls
        lines.append(
            f"{cls} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"
        )
    return lines


def motion_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    kernel_size = int(rng.choice([7, 9, 11, 13]))
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    return cv2.filter2D(image, -1, kernel)


def swap_colors(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    red = (sat >= 50) & ((hue <= 12) | (hue >= 168))
    blue = (sat >= 50) & (hue >= 95) & (hue <= 130)
    hsv = hsv.copy()
    hsv[:, :, 0][red] = 110
    hsv[:, :, 0][blue] = 5
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment(image: np.ndarray, boxes: list[dict], kind: str, rng: np.random.Generator):
    height, width = image.shape[:2]
    if kind == "blur":
        if float(rng.random()) < 0.5:
            out = motion_blur(image, rng)
        else:
            kernel = int(rng.choice([5, 7, 9]))
            out = cv2.GaussianBlur(image, (kernel, kernel), 0)
        return out, boxes, False
    if kind == "swap":
        flipped = []
        for box in boxes:
            copied = dict(box)
            copied["team"] = "blue" if box["team"] == "red" else "red"
            flipped.append(copied)
        return swap_colors(image), flipped, False
    if kind == "letterbox":
        canvas_h = int(round(width * 3 / 4))
        if canvas_h <= height + 2:
            return None
        top = int(rng.integers(0, canvas_h - height + 1))
        canvas = np.zeros((canvas_h, width, 3), np.uint8)
        canvas[top : top + height] = image
        moved = []
        for box in boxes:
            copied = dict(box)
            copied["y"] = box["y"] + top
            moved.append(copied)
        return canvas, moved, False
    if kind == "zoom":
        scale = float(rng.uniform(0.62, 0.88))
        crop_h = max(32, int(round(height * scale)))
        crop_w = max(32, int(round(width * scale)))
        if crop_h >= height or crop_w >= width:
            return None
        y0 = int(rng.integers(0, height - crop_h + 1))
        x0 = int(rng.integers(0, width - crop_w + 1))
        resized = cv2.resize(image[y0 : y0 + crop_h, x0 : x0 + crop_w], (width, height), interpolation=cv2.INTER_LINEAR)
        sx = width / crop_w
        sy = height / crop_h
        moved = []
        for box in boxes:
            cx = box["x"] + box["w"] / 2
            cy = box["y"] + box["h"] / 2
            if not (x0 <= cx <= x0 + crop_w and y0 <= cy <= y0 + crop_h):
                continue
            copied = dict(box)
            copied["x"] = (box["x"] - x0) * sx
            copied["y"] = (box["y"] - y0) * sy
            copied["w"] = box["w"] * sx
            copied["h"] = box["h"] * sy
            moved.append(copied)
        if not moved:
            return None
        return resized, moved, False
    raise ValueError(kind)


def grab_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video))
    found = {}
    index = 0
    last = max(wanted)
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            found[index] = frame
        index += 1
    cap.release()
    return found


def draw_overlay(frame: np.ndarray, boxes: list[dict]) -> np.ndarray:
    vis = frame.copy()
    for box in boxes:
        x, y, w, h = (int(round(box[key])) for key in ("ox", "oy", "ow", "oh"))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)
        color = (0, 0, 255) if box["team"] == "red" else (255, 60, 0)
        px, py, pw, ph = (int(round(box[key])) for key in ("x", "y", "w", "h"))
        cv2.rectangle(vis, (px, py), (px + pw, py + ph), color, 2)
        tag = ("R" if box["team"] == "red" else "B") + ("i" if box["interpolated"] else "")
        cv2.putText(vis, tag, (px, max(14, py - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def write_dataset(round_n: int, train_clips: list[int], val_clips: list[int]) -> dict:
    root = ART / "dataset" / f"round{round_n}"
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    def candidates(numbers: list[int]) -> list[tuple[int, int]]:
        pairs = []
        for number in numbers:
            grouped = boxes_by_frame(ART / "pseudo" / f"round{round_n}" / f"{clip_name(number)}.csv")
            frames = sorted(frame for frame, boxes in grouped.items() if len(boxes) >= MIN_BOXES_PER_FRAME)
            kept = select_frames(footage_path(number), frames)
            pairs.extend((number, frame) for frame in kept)
        return pairs

    print("de-duplicating train/val frames", flush=True)
    train_pairs = subsample(candidates(train_clips), TRAIN_TARGET)
    val_pairs = subsample(candidates(val_clips), VAL_TARGET)
    manifest = []
    aug_saved = 0
    kinds = ["blur", "letterbox", "zoom", "swap"]

    def write_split(pairs: list[tuple[int, int]], split: str, do_aug: bool) -> int:
        nonlocal aug_saved
        written = 0
        by_clip: dict[int, list[int]] = defaultdict(list)
        for number, frame in pairs:
            by_clip[number].append(frame)
        ordinal = 0
        for number, frames in by_clip.items():
            grouped = boxes_by_frame(ART / "pseudo" / f"round{round_n}" / f"{clip_name(number)}.csv")
            images = grab_frames(footage_path(number), set(frames))
            for frame in frames:
                image = images.get(frame)
                boxes = grouped.get(frame, [])
                if image is None or not boxes:
                    continue
                height, width = image.shape[:2]
                stem = f"c{number:02d}_f{frame:05d}"
                lines = yolo_lines(boxes, width, height, swap=False)
                if not lines:
                    continue
                image_path = root / "images" / split / f"{stem}.jpg"
                label_path = root / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                label_path.write_text("\n".join(lines) + "\n")
                manifest.append({"split": split, "clip": number, "frame": frame, "aug": None, "image": str(image_path)})
                written += 1
                if do_aug and ordinal % 3 == 0:
                    kind = kinds[(ordinal // 3) % len(kinds)]
                    rng = np.random.default_rng(10_000 + number * 100_000 + frame)
                    result = augment(image, boxes, kind, rng)
                    if result is not None:
                        aug_image, aug_boxes, _swap_flag = result
                        ah, aw = aug_image.shape[:2]
                        aug_lines = yolo_lines(aug_boxes, aw, ah, swap=False)
                        if aug_lines:
                            aug_stem = f"{stem}_aug_{kind}"
                            cv2.imwrite(
                                str(root / "images" / split / f"{aug_stem}.jpg"),
                                aug_image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                            )
                            (root / "labels" / split / f"{aug_stem}.txt").write_text("\n".join(aug_lines) + "\n")
                            manifest.append(
                                {
                                    "split": split,
                                    "clip": number,
                                    "frame": frame,
                                    "aug": kind,
                                    "image": str(root / "images" / split / f"{aug_stem}.jpg"),
                                }
                            )
                            aug_saved += 1
                            written += 1
                ordinal += 1
        return written

    n_train = write_split(train_pairs, "train", True)
    n_val = write_split(val_pairs, "val", False)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: red_player",
                "  1: blue_player",
                "",
            ]
        )
    )
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return {
        "root": str(root),
        "yaml": str(yaml_path),
        "train_images": n_train,
        "val_images": n_val,
        "aug_images": aug_saved,
        "train_base_pairs": len(train_pairs),
        "val_base_pairs": len(val_pairs),
    }


def write_review(round_n: int, clips: list[int], review_dir: Path) -> list[str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    pool = []
    for number in clips:
        grouped = boxes_by_frame(ART / "pseudo" / f"round{round_n}" / f"{clip_name(number)}.csv")
        for frame, boxes in grouped.items():
            if len(boxes) >= MIN_BOXES_PER_FRAME:
                pool.append((number, frame, boxes))
    rng = np.random.default_rng(0)
    if not pool:
        return []
    if len(pool) > 40:
        pick = rng.choice(len(pool), size=40, replace=False)
        pool = [pool[int(index)] for index in sorted(pick)]
    by_clip: dict[int, list[tuple[int, list[dict]]]] = defaultdict(list)
    for number, frame, boxes in pool:
        by_clip[number].append((frame, boxes))
    saved = []
    sheets = []
    for number, items in by_clip.items():
        images = grab_frames(footage_path(number), {frame for frame, _boxes in items})
        for frame, boxes in items:
            image = images.get(frame)
            if image is None:
                continue
            vis = draw_overlay(image, boxes)
            path = review_dir / f"c{number:02d}_f{frame:05d}.jpg"
            cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved.append(str(path))
            sheets.append(vis)
    if sheets:
        sheet_n = min(8, len(sheets))
        indexes = np.linspace(0, len(sheets) - 1, sheet_n).astype(int)
        cells = []
        for index in indexes:
            cell = cv2.resize(sheets[int(index)], (480, 270), interpolation=cv2.INTER_AREA)
            cells.append(cell)
        while len(cells) < 8:
            cells.append(np.zeros_like(cells[0]))
        top = np.concatenate(cells[:4], axis=1)
        bot = np.concatenate(cells[4:], axis=1)
        contact = np.concatenate([top, bot], axis=0)
        contact_path = review_dir / "contact_sheet.jpg"
        cv2.imwrite(str(contact_path), contact, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        saved.append(str(contact_path))
    return saved


def sum_counts(rows: list[dict]) -> dict:
    keys = [
        "assoc_boxes",
        "assoc_interpolated",
        "dropped_short_track",
        "dropped_few_real",
        "dropped_ambiguous_team",
        "dropped_tiny",
        "dropped_grass",
        "dropped_hud_after_expand",
        "dropped_wide_merge",
        "final_boxes",
        "final_interpolated",
        "final_frames",
        "red",
        "blue",
        "team_flips_before_assoc",
    ]
    total = {key: int(sum(row[key] for row in rows)) for key in keys}
    filter_keys = ["raw", "conf", "size", "aspect", "huge", "hud", "containment", "after_frame_filters"]
    total["raw_filters"] = {
        key: int(sum(row["filters"][key] for row in rows)) for key in filter_keys
    }
    ratios = [row["median_height_ratio"] for row in rows if row["median_height_ratio"] is not None]
    total["median_of_clip_height_ratios"] = round(float(np.median(ratios)), 4) if ratios else None
    return total


def teacher_time(teacher_dir: Path) -> dict:
    total = 0.0
    frames = 0
    steady = []
    for path in sorted(teacher_dir.glob("*.meta.json")):
        meta = json.loads(path.read_text())
        total += float(meta.get("total_sec") or 0.0)
        frames += int(meta.get("nframes") or 0)
        if meta.get("sec_per_frame_steady") is not None:
            steady.append(float(meta["sec_per_frame_steady"]))
    return {
        "clips": len(list(teacher_dir.glob("*.meta.json"))),
        "frames": frames,
        "total_sec": round(total, 1),
        "total_min": round(total / 60.0, 2),
        "mean_sec_per_frame": round(float(np.mean(steady)), 4) if steady else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build pseudo labels and a YOLO dataset.")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--teacher-dir", type=Path, default=None)
    parser.add_argument("--team-source", choices=["hsv", "class"], default="hsv")
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--review-dir", type=Path, default=SCRATCH / "pseudo_review")
    parser.add_argument("--review-only", action="store_true", help="Redraw review overlays from existing pseudo CSVs.")
    args = parser.parse_args(argv)
    ensure_dirs()
    started = time.perf_counter()
    splits = json.loads(SPLITS.read_text())
    train_clips = list(splits["train_clips"])
    val_clips = list(splits["val_clips"])
    numbers = train_clips + val_clips
    if args.review_only:
        review_paths = write_review(args.round, numbers, args.review_dir)
        print(json.dumps({"review": len(review_paths), "dir": str(args.review_dir)}, indent=2), flush=True)
        return 0
    if args.only:
        wanted = set(args.only)
        numbers = [number for number in numbers if number in wanted]
    teacher_dir = args.teacher_dir or (ART / "teacher" / "yolo11m")
    params = assoc_params()
    per_clip = []
    for number in numbers:
        per_clip.append(process_clip(number, teacher_dir, args.round, args.team_source, params))
    review_paths = write_review(args.round, numbers, args.review_dir)
    dataset = None
    if not args.no_dataset:
        dataset_train = train_clips if not args.only else [n for n in train_clips if n in set(args.only)]
        dataset_val = val_clips if not args.only else [n for n in val_clips if n in set(args.only)]
        if not args.only:
            dataset = write_dataset(args.round, dataset_train, dataset_val)
    report = {
        "round": args.round,
        "teacher_dir": str(teacher_dir),
        "team_source": args.team_source,
        "assoc_params": params,
        "min_track": MIN_TRACK,
        "min_real": MIN_REAL,
        "expansion": {
            "method": "non-grass column walk from the teacher box, capped, then median-smoothed along the tracklet",
            "head_cap_frac": HEAD_CAP,
            "foot_cap_frac": FOOT_CAP,
            "height_cap_frac": HEIGHT_CAP,
        },
        "teacher_time": teacher_time(teacher_dir),
        "gdino": "not used; ~9s/frame would be hours even on a subsample, so the teacher is YOLO11m every frame plus temporal filters",
        "counts": sum_counts(per_clip) if per_clip else {},
        "per_clip": per_clip,
        "dataset": dataset,
        "review_images": review_paths,
        "seconds": round(time.perf_counter() - started, 1),
    }
    dest = ART / "reports" / f"pseudo_round{args.round}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": report["counts"], "dataset": dataset, "review": review_paths[:8]}, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
