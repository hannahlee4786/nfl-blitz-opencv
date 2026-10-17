#!/usr/bin/env python3
"""Detect players on every frame and cache motion, appearance, and HUD cues.

The tracker reads these files and does not rerun the detector:

    $NFL_ML_CACHE/det/<clip>_<tag>.csv
    /tmp/ml_trk/det/<clip>_<tag>.hist.npy
    /tmp/ml_trk/det/<clip>_<tag>.motion.npy
    /tmp/ml_trk/det/<clip>_<tag>.markers.csv
    /tmp/ml_trk/det/<clip>_<tag>.hud.npy
    /tmp/ml_trk/det/<clip>_<tag>.meta.json

CSV columns match the scorer: frame,id,team,x,y,w,h,conf.
`motion[t]` is a 2x3 affine mapping frame t-1 onto frame t (identity at t=0).
`hist` rows line up with CSV data rows. `hud` is a quarter-resolution mask,
255 where a pixel stays still and is not grass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.paths import cache_root  # noqa: E402
from ml.detect import (  # noqa: E402
    CLIPS,
    GdinoDetector,
    YoloDetector,
    assign_team,
    clip_xyxy,
)


def torso_hist(bgr: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Normalized 12x4 hue-saturation histogram of the upper-middle box."""
    height, width = bgr.shape[:2]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    hist = np.zeros(48, dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return hist
    roi = bgr[y1:y2, x1:x2]
    top = int(roi.shape[0] * 0.12)
    bot = max(top + 1, int(roi.shape[0] * 0.62))
    torso = roi[top:bot]
    if torso.size == 0:
        return hist
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    counted = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    flat = counted.reshape(-1).astype(np.float32)
    total = float(flat.sum())
    if total > 0:
        flat /= total
    return flat


def estimate_affine(prev_gray: np.ndarray, curr_gray: np.ndarray, full_w: int, full_h: int) -> np.ndarray:
    """Partial affine from the previous frame onto the current frame, in full-res pixels."""
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    small_w = 480
    small_h = max(1, int(round(full_h * (small_w / full_w))))
    prev = cv2.resize(prev_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    curr = cv2.resize(curr_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    points0 = cv2.goodFeaturesToTrack(prev, maxCorners=350, qualityLevel=0.01, minDistance=7)
    if points0 is None or len(points0) < 12:
        return identity
    points1, status, _err = cv2.calcOpticalFlowPyrLK(
        prev,
        curr,
        points0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if points1 is None or status is None:
        return identity
    ok = status.reshape(-1) == 1
    if int(ok.sum()) < 12:
        return identity
    matrix, _inliers = cv2.estimateAffinePartial2D(
        points0[ok],
        points1[ok],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=400,
        confidence=0.98,
    )
    if matrix is None:
        return identity
    sx = full_w / small_w
    sy = full_h / small_h
    linear = matrix[:, :2].astype(np.float64)
    # Reject a broken fit (a scene-sized jump or a wild scale).
    scale = float(np.sqrt(max(1e-8, np.linalg.det(linear))))
    shift = float(np.hypot(matrix[0, 2] * sx, matrix[1, 2] * sy))
    if scale < 0.9 or scale > 1.1 or shift > 180:
        return identity
    return np.array(
        [
            [matrix[0, 0], matrix[0, 1] * (sx / sy), matrix[0, 2] * sx],
            [matrix[1, 0] * (sy / sx), matrix[1, 1], matrix[1, 2] * sy],
        ],
        dtype=np.float64,
    )


def marker_points(bgr: np.ndarray) -> list[tuple[float, float, float]]:
    """Bright blue UI blobs that are not the jersey: ring at the feet, arrow, or the '1' icon."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 95)
        & (hsv[:, :, 0] <= 130)
        & (hsv[:, :, 1] >= 120)
        & (hsv[:, :, 2] >= 100)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = bgr.shape[:2]
    points = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index, :5])
        if area < 80 or area > 9000:
            continue
        if w > 0.22 * width or h > 0.16 * height:
            continue
        if w < 8 or h < 8:
            continue
        if y < 0.015 * height or (y + h) > 0.975 * height:
            continue
        if x < 0.008 * width or (x + w) > 0.992 * width:
            continue
        aspect = w / max(h, 1)
        if aspect > 3.2 or aspect < 0.28:
            continue
        fill = area / float(w * h)
        if fill < 0.18:
            continue
        points.append((float(centroids[index][0]), float(centroids[index][1]), float(area)))
    return points


def build_hud(samples: list[np.ndarray]) -> np.ndarray:
    """Quarter-res mask of screen-fixed non-grass pixels (scoreboard, banners, logos, letterbox)."""
    stack = np.stack(samples).astype(np.float32)
    std = stack.std(axis=0).mean(axis=2)
    median = np.median(stack, axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    grass = (hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 28) & (hsv[:, :, 2] >= 25)
    static = std < 9.0
    hud = (static & ~grass).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    hud = cv2.morphologyEx(hud, cv2.MORPH_OPEN, kernel)
    # A static camera paints standing players into the mask. Keep only blobs that
    # touch the frame border, which is where the HUD and letterbox sit.
    if float((hud > 0).mean()) > 0.28:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(hud, 8)
        height, width = hud.shape
        kept = np.zeros_like(hud)
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index, :5])
            touches = x <= 1 or y <= 1 or (x + w) >= width - 1 or (y + h) >= height - 1
            if touches and area >= 20:
                kept[labels == index] = 255
        hud = kept
    hud = cv2.dilate(hud, np.ones((5, 5), np.uint8), iterations=1)
    return hud


def build_detector(args):
    if args.detector == "yolo":
        return YoloDetector(args.weights, args.imgsz, args.conf, sahi=False)
    if args.detector == "gdino":
        return GdinoDetector(args.weights, args.prompt, args.conf, args.text_threshold)
    raise ValueError(args.detector)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache every-frame detections plus motion and HUD.")
    parser.add_argument(
        "--clip",
        default=None,
        choices=sorted(CLIPS),
        help="Named clip. Required unless --video is set.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Video file. Overrides the named clip path. Without --clip, the cache stem uses the file name.",
    )
    parser.add_argument("--detector", required=True, choices=["yolo", "gdino"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--prompt", default="football player")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--tag", required=True, help="Cache name suffix, e.g. yolo11m or gdino_tiny.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=cache_root() / "det",
    )
    parser.add_argument("--limit", type=int, default=0, help="If >0, only the first N frames (debug).")
    args = parser.parse_args(argv)
    if args.clip is None and args.video is None:
        parser.error("pass --clip or --video")
    if args.video is None:
        video = CLIPS[args.clip]["video"]
        clip_name = args.clip
    else:
        video = args.video.expanduser().resolve()
        clip_name = args.clip if args.clip else video.stem

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{clip_name}_{args.tag}"
    meta_path = stem.with_suffix(".meta.json")
    if meta_path.is_file() and args.limit == 0:
        existing = json.loads(meta_path.read_text())
        if (
            existing.get("complete")
            and existing.get("conf_floor") == args.conf
            and existing.get("detector") == args.detector
            and int(existing.get("limit", 0)) == 0
        ):
            print(f"cache complete, skipping: {stem}", flush=True)
            return 0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.limit > 0:
        nframes = min(nframes, args.limit)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"loading {args.detector} {args.weights}", flush=True)
    detector = build_detector(args)
    ok, warmup = cap.read()
    if not ok:
        raise RuntimeError("empty video")
    started = time.perf_counter()
    detector.predict(warmup)
    warmup_sec = time.perf_counter() - started
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"warmup {warmup_sec:.2f}s  frames {nframes} {width}x{height}", flush=True)

    motion = np.zeros((nframes, 2, 3), dtype=np.float64)
    motion[0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hists: list[np.ndarray] = []
    rows: list[dict] = []
    marker_rows: list[tuple[int, float, float, float]] = []
    hud_samples: list[np.ndarray] = []
    sample_every = max(1, nframes // 28)
    prev_gray = None
    next_id = 1
    infer_times: list[float] = []
    decode_started = time.perf_counter()

    quarter = (max(1, width // 4), max(1, height // 4))
    for index in range(nframes):
        ok, frame = cap.read()
        if not ok:
            nframes = index
            motion = motion[:nframes]
            print(f"video ended early at frame {index}", flush=True)
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            motion[index] = estimate_affine(prev_gray, gray, width, height)
        prev_gray = gray
        if index % sample_every == 0:
            hud_samples.append(cv2.resize(frame, quarter, interpolation=cv2.INTER_AREA))
        for cx, cy, area in marker_points(frame):
            marker_rows.append((index, cx, cy, area))

        t0 = time.perf_counter()
        detections = detector.predict(frame)
        infer_times.append(time.perf_counter() - t0)
        kept = 0
        for det in detections:
            clipped = clip_xyxy(det["xyxy"], width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            box_w = x2 - x1
            box_h = y2 - y1
            if box_h < 12 or box_w < 8:
                continue
            team = assign_team(frame, x1, y1, box_w, box_h)
            hists.append(torso_hist(frame, x1, y1, box_w, box_h))
            rows.append(
                {
                    "frame": index,
                    "id": next_id,
                    "team": team,
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "w": round(box_w, 2),
                    "h": round(box_h, 2),
                    "conf": round(float(det["conf"]), 6),
                }
            )
            next_id += 1
            kept += 1
        if index % 10 == 0 or index == nframes - 1:
            elapsed = time.perf_counter() - decode_started
            print(
                f"frame {index}: {kept} boxes  {infer_times[-1]:.2f}s  elapsed {elapsed/60:.1f}min",
                flush=True,
            )

    cap.release()
    hud = build_hud(hud_samples) if hud_samples else np.zeros(quarter[::-1], np.uint8)
    hist_arr = np.stack(hists) if hists else np.zeros((0, 48), np.float32)

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "id", "team", "x", "y", "w", "h", "conf"])
        writer.writeheader()
        writer.writerows(rows)
    np.save(stem.with_suffix(".hist.npy"), hist_arr)
    np.save(stem.with_suffix(".motion.npy"), motion)
    np.save(stem.with_suffix(".hud.npy"), hud)
    with stem.with_suffix(".markers.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "area"])
        writer.writerows(marker_rows)

    steady = infer_times[1:] if len(infer_times) > 1 else infer_times
    meta = {
        "complete": True,
        "clip": clip_name,
        "video": str(video),
        "detector": args.detector,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "prompt": args.prompt,
        "conf_floor": args.conf,
        "text_threshold": args.text_threshold,
        "limit": args.limit,
        "nframes": nframes,
        "width": width,
        "height": height,
        "boxes": len(rows),
        "markers": len(marker_rows),
        "hud_fraction": round(float((hud > 0).mean()), 4),
        "warmup_sec": round(warmup_sec, 3),
        "sec_per_frame": round(float(np.mean(infer_times)) if infer_times else 0.0, 4),
        "sec_per_frame_steady": round(float(np.mean(steady)) if steady else 0.0, 4),
        "total_sec": round(time.perf_counter() - decode_started, 3),
        "csv": str(csv_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {csv_path} ({len(rows)} boxes), hud {meta['hud_fraction']:.3f}, "
        f"{meta['sec_per_frame_steady']:.3f}s/frame",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
