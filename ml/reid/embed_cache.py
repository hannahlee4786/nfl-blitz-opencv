#!/usr/bin/env python3
"""Embed every cached detection. Writes <stem>.reid.npy aligned with the CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.reid.model import Embedder  # noqa: E402
from ml.reid.train import MEAN, STD, torso  # noqa: E402

CROP_H = 128
CROP_W = 64


def crop(frame, x, y, w, h):
    height, width = frame.shape[:2]
    x1 = int(np.clip(np.floor(x), 0, width - 1))
    y1 = int(np.clip(np.floor(y), 0, height - 1))
    x2 = int(np.clip(np.ceil(x + w), x1 + 1, width))
    y2 = int(np.clip(np.ceil(y + h), y1 + 1, height))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        patch = np.zeros((CROP_H, CROP_W, 3), np.uint8)
    else:
        patch = cv2.resize(patch, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
        patch = torso(patch)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache re-id embeddings for a detection stem.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.cache.with_suffix(".csv").open()))
    out = args.cache.with_suffix(".reid.npy")
    meta_path = args.cache.with_suffix(".reid.json")
    if out.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("weights") == str(args.weights) and meta.get("boxes") == len(rows) and meta.get("complete"):
            print(f"reid cache complete, skipping {out}", flush=True)
            return 0
    torch.set_num_threads(6)
    model = Embedder()
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    by_frame: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_frame.setdefault(int(row["frame"]), []).append(index)
    emb = np.zeros((len(rows), 128), dtype=np.float32)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    frame_index = 0
    last = max(by_frame) if by_frame else -1
    pending_img = []
    pending_idx = []

    def flush() -> None:
        if not pending_img:
            return
        with torch.no_grad():
            pred = model(torch.stack(pending_img)).numpy()
        for slot, index in enumerate(pending_idx):
            emb[index] = pred[slot]
        pending_img.clear()
        pending_idx.clear()

    while frame_index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for index in by_frame.get(frame_index, []):
            row = rows[index]
            pending_img.append(crop(frame, float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])))
            pending_idx.append(index)
            if len(pending_img) >= args.batch:
                flush()
        if frame_index % 40 == 0:
            print(f"embed frame {frame_index}/{last}", flush=True)
        frame_index += 1
    flush()
    cap.release()
    np.save(out, emb)
    meta_path.write_text(
        json.dumps(
            {"complete": True, "weights": str(args.weights), "boxes": len(rows), "video": str(args.video), "dim": 128},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out} {emb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
