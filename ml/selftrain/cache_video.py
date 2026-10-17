#!/usr/bin/env python3
"""Write a track_assoc-compatible detection cache for any video.

Teacher mode: YOLO11 COCO person, HSV team (ml.detect.assign_team).
Student mode: fine-tuned 2-class detector, team from the class
(0 red_player, 1 blue_player), agnostic NMS so one body gets one team.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2
import numpy as np

from ml.detect import assign_team, clip_xyxy, ensure_yolo_weights
from ml.detect_all import build_hud, estimate_affine, marker_points, torso_hist
from ml.selftrain.common import ART, SPLITS, TEST, YOLO11M, clip_name, ensure_dirs, footage_path


def load_model(weights: Path):
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    path = ensure_yolo_weights(str(weights))
    return YOLO(str(path)), path


def predict_batch(model, frames: list[np.ndarray], imgsz: int, conf: float, mode: str) -> list[list[dict]]:
    kwargs = {
        "imgsz": imgsz,
        "conf": conf,
        "device": "cpu",
        "verbose": False,
        "max_det": 80,
    }
    if mode == "teacher":
        kwargs.update(iou=0.7, classes=[0], agnostic_nms=False)
    else:
        kwargs.update(iou=0.60, agnostic_nms=True)
    results = model.predict(frames, **kwargs)
    parsed: list[list[dict]] = []
    for result in results:
        boxes = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            for box, score, cls_id in zip(xyxy, confs, clss):
                boxes.append(
                    {
                        "xyxy": [float(v) for v in box],
                        "conf": float(score),
                        "cls": int(cls_id),
                    }
                )
        parsed.append(boxes)
    return parsed


def cache_is_complete(
    stem: Path, mode: str, weights: str, imgsz: int, conf: float, video: Path, team_source: str = ""
) -> bool:
    meta_path = stem.with_suffix(".meta.json")
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    if not meta.get("complete"):
        return False
    if meta.get("mode") != mode or int(meta.get("imgsz", -1)) != int(imgsz):
        return False
    if float(meta.get("conf_floor", -1)) != float(conf):
        return False
    if meta.get("weights") != str(weights):
        return False
    if int(meta.get("video_size", -1)) != video.stat().st_size:
        return False
    if team_source:
        stored = meta.get("team_source", "hsv" if meta.get("mode") == "teacher" else "class")
        if stored != team_source:
            return False
    return True


def cache_one(
    video: Path,
    stem: Path,
    mode: str,
    weights: Path,
    imgsz: int,
    conf: float,
    batch: int,
    model=None,
    resolved_weights: Path | None = None,
    team_source: str | None = None,
) -> dict:
    """Detect every frame and write the detect_all cache files. Returns meta."""
    if model is None:
        model, resolved_weights = load_model(weights)
    assert resolved_weights is not None
    if team_source is None:
        team_source = "hsv" if mode == "teacher" else "class"
    if cache_is_complete(stem, mode, str(resolved_weights), imgsz, conf, video, team_source):
        meta = json.loads(stem.with_suffix(".meta.json").read_text())
        print(f"cache complete, skipping {stem}", flush=True)
        return meta

    stem.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, warmup = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"empty video {video}")
    started_warm = time.perf_counter()
    predict_batch(model, [warmup], imgsz, conf, mode)
    warmup_sec = time.perf_counter() - started_warm
    cap.release()
    cap = cv2.VideoCapture(str(video))
    print(f"warmup {warmup_sec:.2f}s {video.name} {width}x{height} batch {batch}", flush=True)

    motion = np.zeros((1024, 2, 3), dtype=np.float64)
    hists: list[np.ndarray] = []
    rows: list[dict] = []
    marker_rows: list[tuple[int, float, float, float]] = []
    hud_samples: list[np.ndarray] = []
    prev_gray = None
    next_id = 1
    infer_times: list[float] = []
    quarter = (max(1, width // 4), max(1, height // 4))
    batch_frames: list[np.ndarray] = []
    batch_index: list[int] = []
    index = 0
    decode_started = time.perf_counter()

    def grow(frame_index: int) -> None:
        nonlocal motion
        if frame_index < len(motion):
            return
        extra = np.zeros((frame_index + 512, 2, 3), dtype=np.float64)
        motion = np.concatenate([motion, extra], axis=0)

    def consume(frame: np.ndarray, frame_index: int, detections: list[dict], per_frame: float) -> None:
        nonlocal next_id
        infer_times.append(per_frame)
        kept = 0
        for det in detections:
            if mode == "student" and det["cls"] not in (0, 1):
                continue
            clipped = clip_xyxy(det["xyxy"], width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            box_w = x2 - x1
            box_h = y2 - y1
            if box_h < 12 or box_w < 8:
                continue
            use_hsv = team_source == "hsv" or mode == "teacher"
            if use_hsv:
                team = assign_team(frame, x1, y1, box_w, box_h)
            else:
                team = "red" if det["cls"] == 0 else "blue"
            hists.append(torso_hist(frame, x1, y1, box_w, box_h))
            rows.append(
                {
                    "frame": frame_index,
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
        if frame_index % 20 == 0:
            elapsed = time.perf_counter() - decode_started
            print(
                f"frame {frame_index}: {kept} boxes  {per_frame:.3f}s  elapsed {elapsed / 60:.1f}min",
                flush=True,
            )

    def flush() -> None:
        if not batch_frames:
            return
        started = time.perf_counter()
        parsed = predict_batch(model, batch_frames, imgsz, conf, mode)
        per_frame = (time.perf_counter() - started) / len(batch_frames)
        for frame, frame_index, detections in zip(batch_frames, batch_index, parsed):
            consume(frame, frame_index, detections, per_frame)
        batch_frames.clear()
        batch_index.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grow(index)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if index == 0:
            motion[0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        else:
            motion[index] = estimate_affine(prev_gray, gray, width, height)
        prev_gray = gray
        if index % max(1, 30) == 0:
            hud_samples.append(cv2.resize(frame, quarter, interpolation=cv2.INTER_AREA))
        for cx, cy, area in marker_points(frame):
            marker_rows.append((index, cx, cy, area))
        batch_frames.append(frame)
        batch_index.append(index)
        index += 1
        if len(batch_frames) >= batch:
            flush()
    flush()
    cap.release()
    nframes = index
    motion = motion[:nframes]
    if nframes == 0:
        raise RuntimeError(f"no frames read from {video}")

    sample_target = max(1, nframes // 28)
    # detect_all samples ~28 frames spread over the clip. Recompute HUD from a
    # second pass only if the in-loop stride (every 30) undershoots a lot.
    # The every-30 sample is close enough for clips of a few hundred frames
    # and avoids a second decode. Rebuild if we stored fewer than 8.
    if len(hud_samples) < 8:
        cap = cv2.VideoCapture(str(video))
        hud_samples = []
        step = max(1, nframes // 28)
        cursor = 0
        while cursor < nframes:
            ok, frame = cap.read()
            if not ok:
                break
            if cursor % step == 0:
                hud_samples.append(cv2.resize(frame, quarter, interpolation=cv2.INTER_AREA))
            cursor += 1
        cap.release()
    hud = build_hud(hud_samples) if hud_samples else np.zeros((quarter[1], quarter[0]), np.uint8)
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

    steady = infer_times[2:] if len(infer_times) > 4 else infer_times
    meta = {
        "complete": True,
        "mode": mode,
        "clip": stem.name,
        "video": str(video),
        "video_size": video.stat().st_size,
        "detector": "yolo" if mode == "teacher" else "student",
        "weights": str(resolved_weights),
        "imgsz": imgsz,
        "prompt": "person" if mode == "teacher" else "red_player,blue_player",
        "conf_floor": conf,
        "text_threshold": None,
        "limit": 0,
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
        "batch": batch,
        "team_source": team_source,
        "csv": str(csv_path),
        "unused_sample_target": sample_target,
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {csv_path} ({len(rows)} boxes), hud {meta['hud_fraction']:.3f}, "
        f"{meta['sec_per_frame_steady']:.3f}s/frame, {meta['total_sec']:.0f}s",
        flush=True,
    )
    return meta


def bench(video: Path, weights: Path, imgsz: int, conf: float, frames_n: int, batches: list[int]) -> dict:
    model, resolved = load_model(weights)
    cap = cv2.VideoCapture(str(video))
    frames = []
    while len(frames) < frames_n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    predict_batch(model, frames[:1], imgsz, conf, "teacher")
    report = {"video": str(video), "weights": str(resolved), "imgsz": imgsz, "frames": len(frames), "batches": {}}
    for size in batches:
        started = time.perf_counter()
        seen = 0
        for start in range(0, len(frames), size):
            chunk = frames[start : start + size]
            predict_batch(model, chunk, imgsz, conf, "teacher")
            seen += len(chunk)
        elapsed = time.perf_counter() - started
        report["batches"][str(size)] = {
            "sec_per_frame": round(elapsed / max(seen, 1), 4),
            "total_sec": round(elapsed, 3),
        }
        print(f"batch {size}: {elapsed / max(seen, 1):.3f}s/frame", flush=True)
    return report


def jobs_from_args(args) -> list[tuple[str, Path, Path]]:
    """Return (stem_name, video, stem_path) jobs."""
    if args.which == "test":
        out_dir = ART / "student_cache" / f"round{args.round}_imgsz{args.imgsz}"
        jobs = []
        for name, spec in TEST.items():
            jobs.append((name, spec["video"], out_dir / name))
        return jobs
    splits = json.loads(SPLITS.read_text())
    if args.which == "train":
        numbers = splits["train_clips"]
    elif args.which == "val":
        numbers = splits["val_clips"]
    else:
        numbers = list(splits["train_clips"]) + list(splits["val_clips"])
    if args.only:
        wanted = set(args.only)
        numbers = [number for number in numbers if number in wanted]
    if args.mode == "teacher":
        out_dir = ART / "teacher" / "yolo11m"
    else:
        out_dir = ART / "student_cache" / f"round{args.round}_imgsz{args.imgsz}"
    return [(clip_name(number), footage_path(number), out_dir / clip_name(number)) for number in numbers]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache detections for footage or test clips.")
    parser.add_argument("--mode", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--which", choices=["train", "val", "trainval", "test"], default="trainval")
    parser.add_argument("--weights", type=Path, default=YOLO11M)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--bench", type=int, default=0, help="If >0, time this many frames and exit.")
    parser.add_argument("--bench-video", type=Path, default=None)
    parser.add_argument(
        "--team-source",
        choices=["hsv", "class"],
        default=None,
        help="Jersey label. Default hsv for the teacher and the detector class for the student.",
    )
    args = parser.parse_args(argv)
    ensure_dirs()
    if args.bench > 0:
        video = args.bench_video
        if video is None:
            video = footage_path(5)
        report = bench(video, args.weights, args.imgsz, args.conf, args.bench, [1, 4, 8])
        dest = ART / "logs" / "teacher_bench.json"
        dest.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {dest}", flush=True)
        return 0

    jobs = jobs_from_args(args)
    if not jobs:
        raise RuntimeError("no videos selected")
    model, resolved = load_model(args.weights)
    summaries = []
    for name, video, stem in jobs:
        print(f"=== {name} {video}", flush=True)
        meta = cache_one(
            video,
            stem,
            args.mode,
            args.weights,
            args.imgsz,
            args.conf,
            args.batch,
            model=model,
            resolved_weights=resolved,
            team_source=args.team_source,
        )
        summaries.append(
            {
                "name": name,
                "boxes": meta.get("boxes"),
                "nframes": meta.get("nframes"),
                "sec_per_frame_steady": meta.get("sec_per_frame_steady"),
                "total_sec": meta.get("total_sec"),
                "skipped": meta.get("total_sec") is None,
            }
        )
        progress = ART / "logs" / f"cache_{args.mode}_progress.json"
        progress.write_text(json.dumps(summaries, indent=2) + "\n")
    timed = [row for row in summaries if row.get("total_sec")]
    total_sec = sum(float(row["total_sec"]) for row in timed)
    total_frames = sum(int(row["nframes"]) for row in timed)
    print(
        f"done {len(summaries)} clips, timed frames {total_frames}, "
        f"sum of clip totals {total_sec / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
