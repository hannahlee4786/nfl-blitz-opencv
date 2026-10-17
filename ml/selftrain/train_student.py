#!/usr/bin/env python3
"""Fine-tune YOLO11n (2 classes) on the pseudo-label dataset.

Epoch wall times are appended to logs/round{N}_epoch_times.jsonl.
Checkpoint choice is NOT done here; ml/selftrain/evaluate.py scores Clip 4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from ml.selftrain.common import ART, YOLO11N, ensure_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the student detector.")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    ensure_dirs()

    import torch

    torch.set_num_threads(8)
    data = ART / "dataset" / f"round{args.round}" / "data.yaml"
    if not data.is_file():
        raise RuntimeError(f"missing dataset yaml {data}")
    weights = args.weights or YOLO11N
    if not weights.is_file():
        raise RuntimeError(f"missing weights {weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    times_path = ART / "logs" / f"round{args.round}_epoch_times.jsonl"
    state = {"t": time.perf_counter()}

    def on_train_epoch_start(trainer):
        state["t"] = time.perf_counter()

    def on_train_epoch_end(trainer):
        elapsed = time.perf_counter() - state["t"]
        record = {"epoch": int(trainer.epoch), "sec": round(elapsed, 2)}
        with times_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"EPOCH {trainer.epoch} sec {elapsed:.1f}", flush=True)

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    train_kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ART / "runs"),
        name=f"round{args.round}",
        exist_ok=True,
        pretrained=True,
        patience=8,
        save=True,
        save_period=1,
        cache=False,
        amp=False,
        plots=False,
        seed=0,
        deterministic=False,
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=1.0,
        hsv_h=0.0,
        hsv_s=0.3,
        hsv_v=0.25,
        scale=0.5,
        mosaic=1.0,
        mixup=0.0,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.1,
        erasing=0.05,
        close_mosaic=5,
        val=True,
        verbose=True,
        resume=args.resume,
    )
    print(f"training {weights} on {data} batch {args.batch} imgsz {args.imgsz}", flush=True)
    try:
        model.train(**train_kwargs)
    except RuntimeError as exc:
        text = str(exc).lower()
        if args.batch > 4 and ("memory" in text or "alloc" in text):
            print(f"batch {args.batch} failed ({exc}); retrying batch 4", flush=True)
            train_kwargs["batch"] = 4
            model = YOLO(str(weights))
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.train(**train_kwargs)
        else:
            raise
    print(f"epoch times {times_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
