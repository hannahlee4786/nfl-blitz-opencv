#!/usr/bin/env python3
"""Crops and labels from forward tracklets on unlabeled footage.

Positives are the same forward tracklet at distant frames. The manifest also
stores team, frame, and center so training can mine same-team hard negatives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.paths import ml_home  # noqa: E402
from ml.selftrain.common import ART, SPLITS, footage_path  # noqa: E402
from ml.track_assoc import DEFAULTS, associate_clean, build_frames, load_cache  # noqa: E402

OUT = ml_home() / "reid" / "dataset"
CROP_H = 128
CROP_W = 64
MIN_OBS = 10
MIN_SPAN = 16
MAX_PER_TRACK = 8
MIN_GAP = 10


def params_for(clip: int) -> dict:
    summary = ART / "tracks" / "round1" / f"clip_{clip:02d}.summary.json"
    params = dict(DEFAULTS)
    if summary.is_file():
        stored = json.loads(summary.read_text()).get("params") or {}
        for key in params:
            if key in stored:
                params[key] = stored[key]
    return params


def split_pieces(track) -> list[list]:
    """Break a track where the associator coasted. A gap often hides a swap."""
    obs = sorted(track.obs, key=lambda det: det.frame)
    pieces: list[list] = []
    current = [obs[0]]
    for det in obs[1:]:
        if det.frame - current[-1].frame > 4:
            pieces.append(current)
            current = [det]
        else:
            current.append(det)
    pieces.append(current)
    return pieces


def reliable(obs) -> bool:
    if len(obs) < MIN_OBS:
        return False
    span = obs[-1].frame - obs[0].frame
    if span < MIN_SPAN:
        return False
    jumps = 0
    for left, right in zip(obs, obs[1:]):
        dt = max(1, right.frame - left.frame)
        step = float(np.hypot(right.cx - left.cx, right.cy - left.cy))
        if step > 0.85 * max(left.h, 28.0) * dt:
            jumps += 1
    if jumps >= 2:
        return False
    red = sum(1 for det in obs if det.team == "red")
    if min(red, len(obs) - red) / len(obs) > 0.15:
        return False
    areas = np.array([det.w * det.h for det in obs], dtype=np.float64)
    med = float(np.median(areas))
    if med <= 0 or float(np.mean(areas > 2.2 * med)) > 0.15:
        return False
    aspects = np.array([det.h / max(det.w, 1.0) for det in obs], dtype=np.float64)
    if float(np.median(aspects)) < 0.7 or float(np.median(aspects)) > 4.8:
        return False
    return True


def pick_frames(obs) -> list:
    chosen = [obs[0]]
    for det in obs[1:]:
        if det.frame - chosen[-1].frame >= MIN_GAP:
            chosen.append(det)
    if len(chosen) > MAX_PER_TRACK:
        idx = np.linspace(0, len(chosen) - 1, MAX_PER_TRACK).round().astype(int)
        chosen = [chosen[i] for i in idx]
    return chosen


def crop_box(frame, det) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1 = int(np.clip(np.floor(det.x), 0, width - 1))
    y1 = int(np.clip(np.floor(det.y), 0, height - 1))
    x2 = int(np.clip(np.ceil(det.x + det.w), x1 + 1, width))
    y2 = int(np.clip(np.ceil(det.y + det.h), y1 + 1, height))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 12 or patch.shape[1] < 8:
        return None
    return cv2.resize(patch, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


def build_split(numbers: list[int], split: str) -> dict:
    image_dir = OUT / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / f"{split}.jsonl"
    done_path = OUT / f"{split}.done.json"
    done = set(json.loads(done_path.read_text())) if done_path.is_file() else set()
    stats = {
        "clips": 0,
        "tracklets_kept": 0,
        "tracklets_rejected": 0,
        "crops": 0,
        "rejected_reasons": {"short": 0, "flip_or_merge": 0},
    }
    lines = manifest_path.read_text().splitlines() if manifest_path.is_file() else []
    with manifest_path.open("a") as handle:
        for clip in numbers:
            key = f"{clip:02d}"
            if key in done:
                continue
            stem = ART / "teacher_smoothed" / "round1" / f"clip_{clip:02d}"
            video = footage_path(clip)
            if not stem.with_suffix(".meta.json").is_file() or not video.is_file():
                print(f"skip clip {clip}: missing cache or video", flush=True)
                done.add(key)
                continue
            meta, rows, hist, motion, hud, _markers = load_cache(stem)
            params = params_for(clip)
            frames = build_frames(meta, rows, hist, hud, params)
            tracks = associate_clean(frames, motion, params)
            kept = []
            for track in tracks:
                team = track.team()
                for piece in split_pieces(track):
                    if reliable(piece):
                        kept.append((team, piece))
                    else:
                        stats["tracklets_rejected"] += 1
            print(f"clip {clip}: {len(tracks)} forward, {len(kept)} kept", flush=True)
            needed: dict[int, list] = {}
            for index, (team, piece) in enumerate(kept):
                for det in pick_frames(piece):
                    needed.setdefault(det.frame, []).append((index, det))
            if needed:
                cap = cv2.VideoCapture(str(video))
                index = 0
                last = max(needed)
                while index <= last:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    for track_index, det in needed.get(index, []):
                        patch = crop_box(frame, det)
                        if patch is None:
                            continue
                        team = kept[track_index][0]
                        name = f"c{clip:02d}_t{track_index:03d}_f{det.frame:04d}.jpg"
                        cv2.imwrite(str(image_dir / name), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        record = {
                            "split": split,
                            "clip": clip,
                            "track": track_index,
                            "identity": f"{clip:02d}_{track_index:03d}",
                            "team": team,
                            "frame": int(det.frame),
                            "cx": round(det.cx, 2),
                            "cy": round(det.cy, 2),
                            "h": round(det.h, 2),
                            "file": f"images/{split}/{name}",
                            "hist": [round(float(v), 5) for v in det.hist.tolist()],
                        }
                        handle.write(json.dumps(record) + "\n")
                        stats["crops"] += 1
                    index += 1
                cap.release()
            stats["clips"] += 1
            stats["tracklets_kept"] += len(kept)
            done.add(key)
            done_path.write_text(json.dumps(sorted(done)))
            handle.flush()
    stats["manifest_lines"] = sum(1 for line in manifest_path.read_text().splitlines() if line.strip())
    stats["prior_lines"] = len(lines)
    return stats


def main() -> int:
    splits = json.loads(SPLITS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "train": build_split(splits["train_clips"], "train"),
        "val": build_split(splits["val_clips"], "val"),
    }
    dest = OUT / "counts.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
