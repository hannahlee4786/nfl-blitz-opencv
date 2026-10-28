#!/usr/bin/env python3
"""Draw track boxes and ids on selected frames."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.detect import CLIPS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Write review stills for a boxes CSV.")
    parser.add_argument("--clip", choices=sorted(CLIPS), help="Named clip. Required unless --video is set.")
    parser.add_argument("--video", type=Path, help="Video file. Stills are named after its stem.")
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated frame indices.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.clip is None and args.video is None:
        parser.error("pass --clip or --video")
    video = args.video if args.video is not None else CLIPS[args.clip]["video"]
    name = args.clip if args.clip else video.stem.replace(" ", "_")

    wanted = [int(piece) for piece in args.frames.split(",") if piece.strip()]
    by_frame: dict[int, list[dict]] = {frame: [] for frame in wanted}
    with args.boxes.open() as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame in by_frame:
                by_frame[frame].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(video)
    index = 0
    last = max(wanted)
    written = []
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index in by_frame:
            canvas = frame.copy()
            rows = sorted(by_frame[index], key=lambda row: int(row["id"]))
            for row in rows:
                x, y, w, h = (int(float(row[k])) for k in ("x", "y", "w", "h"))
                color = (40, 40, 230) if row["team"] == "red" else (230, 90, 20)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
                label = f"{row['id']} {row['team']}"
                cv2.putText(
                    canvas,
                    label,
                    (x, max(18, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            dest = args.out_dir / f"{name}_f{index:04d}.jpg"
            cv2.imwrite(str(dest), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            written.append(str(dest))
            print(f"{dest} boxes={len(rows)}", flush=True)
        index += 1
    cap.release()
    missing = [frame for frame in wanted if not (args.out_dir / f"{name}_f{frame:04d}.jpg").is_file()]
    if missing:
        raise RuntimeError(f"missing frames {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
