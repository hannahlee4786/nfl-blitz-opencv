#!/usr/bin/env python3
"""Zero-shot player detection on labeled frames.

Writes a boxes CSV the repo scorer understands:

    frame,id,team,x,y,w,h,conf

x,y is the top-left corner in source-video pixels. Each box gets a unique
integer id. Team is a fixed HSV jersey rule (red vs blue), not a learned
classifier. No training and no use of the label boxes.

Frames are decoded sequentially. Only frames listed under labels["frames"]
are passed to the detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "/Users/starfengsmac/Downloads/nfl-ml/models/hf")
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    "/Users/starfengsmac/Downloads/nfl-ml/models/hf/hub",
)
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    "/Users/starfengsmac/Downloads/nfl-ml/models/Ultralytics",
)

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = Path("/Users/starfengsmac/Downloads/nfl-ml/models")
CLIPS = {
    "clip4": {
        "video": REPO / "data/input/Clip_4 Camera Swing and Distance Change Test.mp4",
        "labels": REPO / "data/labels/clip4.json",
    },
    "clip64": {
        "video": REPO / "data/input/Clip 64.mov",
        "labels": REPO / "data/labels/clip64.json",
    },
}


def labeled_frame_ids(path: Path) -> list[int]:
    with path.open() as handle:
        data = json.load(handle)
    frames = data.get("frames")
    if not isinstance(frames, dict):
        raise ValueError(f"{path} has no frames object")
    return sorted(int(key) for key in frames)


def read_frames(video: Path, wanted: list[int]) -> tuple[dict[int, np.ndarray], float]:
    """Sequential decode. Returns frame index -> BGR image, and decode seconds."""
    need = set(wanted)
    if not need:
        return {}, 0.0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    start = time.perf_counter()
    found: dict[int, np.ndarray] = {}
    index = 0
    last = max(need)
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index in need:
            found[index] = frame
        index += 1
    cap.release()
    missing = sorted(need - set(found))
    if missing:
        raise RuntimeError(f"{video} ended before frames {missing}")
    return found, time.perf_counter() - start


def assign_team(bgr: np.ndarray, x: float, y: float, w: float, h: float) -> str:
    """Red vs blue from saturated jersey pixels in the upper-middle box.

    OpenCV hue: red wraps through 0, blue sits near 90-135. Green field,
    yellow lines, and white numbers are left out by hue or saturation.
    This rule is fixed; it is not fit on the labeled boxes.
    """
    height, width = bgr.shape[:2]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return "red"
    roi = bgr[y1:y2, x1:x2]
    rh = roi.shape[0]
    top = int(rh * 0.15)
    bot = max(top + 1, int(rh * 0.60))
    torso = roi[top:bot]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    def counts(min_sat: int, min_val: int) -> tuple[int, int]:
        colorful = (sat >= min_sat) & (val >= min_val)
        red = colorful & ((hue <= 15) | (hue >= 165))
        blue = colorful & (hue >= 90) & (hue <= 135)
        return int(red.sum()), int(blue.sum())

    red_count, blue_count = counts(60, 40)
    if red_count + blue_count < 10:
        red_count, blue_count = counts(30, 30)
    if red_count + blue_count >= 10:
        return "red" if red_count >= blue_count else "blue"
    mean = torso.reshape(-1, 3).astype(np.float32).mean(axis=0)
    blue_ch, _green, red_ch = (float(v) for v in mean)
    if blue_ch > red_ch + 8:
        return "blue"
    return "red"


def nms_xyxy(boxes: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy NMS. boxes are dicts with xyxy and conf."""
    if len(boxes) <= 1:
        return boxes
    order = sorted(range(len(boxes)), key=lambda i: boxes[i]["conf"], reverse=True)
    kept: list[dict] = []
    while order:
        current = order.pop(0)
        kept.append(boxes[current])
        cx1, cy1, cx2, cy2 = boxes[current]["xyxy"]
        remain = []
        for index in order:
            x1, y1, x2, y2 = boxes[index]["xyxy"]
            inter_w = max(0.0, min(cx2, x2) - max(cx1, x1))
            inter_h = max(0.0, min(cy2, y2) - max(cy1, y1))
            inter = inter_w * inter_h
            if inter <= 0:
                remain.append(index)
                continue
            area_c = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            area_o = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = area_c + area_o - inter
            iou = inter / union if union > 0 else 0.0
            if iou < iou_threshold:
                remain.append(index)
        order = remain
    return kept


def clip_xyxy(xyxy, width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


class YoloDetector:
    def __init__(self, weights: str, imgsz: int, conf: float, sahi: bool):
        from ultralytics import YOLO

        path = ensure_yolo_weights(weights)
        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.conf = conf
        self.sahi = sahi
        self.detail = {
            "family": "yolo",
            "weights": str(path),
            "imgsz": imgsz,
            "sahi": sahi,
            "classes": [0],
            "iou": 0.7,
        }

    def _predict_image(self, bgr: np.ndarray, origin_xy=(0, 0)) -> list[dict]:
        result = self.model.predict(
            bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.7,
            classes=[0],
            device="cpu",
            verbose=False,
            max_det=300,
        )[0]
        ox, oy = origin_xy
        boxes = []
        if result.boxes is None or len(result.boxes) == 0:
            return boxes
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, confs):
            boxes.append(
                {
                    "xyxy": [float(box[0] + ox), float(box[1] + oy), float(box[2] + ox), float(box[3] + oy)],
                    "conf": float(score),
                }
            )
        return boxes

    def predict(self, bgr: np.ndarray) -> list[dict]:
        if not self.sahi:
            return self._predict_image(bgr)
        height, width = bgr.shape[:2]
        tile = 640
        overlap = 0.2
        stride = max(1, int(tile * (1.0 - overlap)))
        boxes: list[dict] = []
        ys = list(range(0, max(1, height - tile + 1), stride))
        xs = list(range(0, max(1, width - tile + 1), stride))
        if not ys or ys[-1] != max(0, height - tile):
            ys.append(max(0, height - tile))
        if not xs or xs[-1] != max(0, width - tile):
            xs.append(max(0, width - tile))
        for top in ys:
            for left in xs:
                bottom = min(height, top + tile)
                right = min(width, left + tile)
                crop = bgr[top:bottom, left:right]
                boxes.extend(self._predict_image(crop, (left, top)))
        return nms_xyxy(boxes, 0.5)


class OwlDetector:
    def __init__(self, model_id: str, prompt: str, conf: float):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.model.eval()
        self.prompt = prompt
        self.conf = conf
        size = getattr(self.processor.image_processor, "size", None)
        self.detail = {
            "family": "owlv2",
            "weights": model_id,
            "prompt": prompt,
            "processor_size": size,
            "nms_iou": 0.5,
        }

    def predict(self, bgr: np.ndarray) -> list[dict]:
        torch = self.torch
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=[[self.prompt]], images=pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        target = torch.tensor([pil.size[::-1]])
        processed = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target,
            threshold=self.conf,
        )[0]
        boxes = []
        for box, score in zip(processed["boxes"], processed["scores"]):
            boxes.append({"xyxy": [float(v) for v in box.tolist()], "conf": float(score)})
        return nms_xyxy(boxes, 0.5)


class GdinoDetector:
    def __init__(self, model_id: str, prompt: str, conf: float, text_threshold: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.eval()
        self.prompt = prompt if prompt.endswith(".") else prompt + "."
        self.conf = conf
        self.text_threshold = text_threshold
        size = getattr(self.processor.image_processor, "size", None)
        self.detail = {
            "family": "grounding-dino",
            "weights": model_id,
            "prompt": self.prompt,
            "text_threshold": text_threshold,
            "processor_size": size,
            "nms_iou": 0.5,
        }

    def predict(self, bgr: np.ndarray) -> list[dict]:
        import inspect

        torch = self.torch
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil, text=self.prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        signature = inspect.signature(self.processor.post_process_grounded_object_detection)
        kwargs = {
            "outputs": outputs,
            "target_sizes": [pil.size[::-1]],
        }
        if "input_ids" in signature.parameters:
            kwargs["input_ids"] = inputs["input_ids"]
        if "threshold" in signature.parameters:
            kwargs["threshold"] = self.conf
        elif "box_threshold" in signature.parameters:
            kwargs["box_threshold"] = self.conf
        if "text_threshold" in signature.parameters:
            kwargs["text_threshold"] = self.text_threshold
        processed = self.processor.post_process_grounded_object_detection(**kwargs)[0]
        boxes = []
        for box, score in zip(processed["boxes"], processed["scores"]):
            xyxy = box.tolist() if hasattr(box, "tolist") else list(box)
            value = score.item() if hasattr(score, "item") else float(score)
            boxes.append({"xyxy": [float(v) for v in xyxy], "conf": float(value)})
        return nms_xyxy(boxes, 0.5)


def ensure_yolo_weights(weights: str) -> Path:
    candidate = Path(weights)
    if candidate.is_file():
        return candidate
    name = candidate.name
    dest = MODELS / name
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    MODELS.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    previous = os.getcwd()
    os.chdir(MODELS)
    try:
        YOLO(name)
    finally:
        os.chdir(previous)
    if not dest.is_file():
        raise FileNotFoundError(f"ultralytics did not download {name} into {MODELS}")
    return dest


def build_detector(args) -> YoloDetector | OwlDetector | GdinoDetector:
    if args.detector == "yolo":
        return YoloDetector(args.weights, args.imgsz, args.conf, args.sahi)
    if args.detector == "owlv2":
        return OwlDetector(args.weights, args.prompt, args.conf)
    if args.detector == "gdino":
        return GdinoDetector(args.weights, args.prompt, args.conf, args.text_threshold)
    raise ValueError(args.detector)


def frame_fingerprint(bgr: np.ndarray) -> dict:
    return {
        "shape": list(bgr.shape),
        "mean_bgr": [round(float(v), 4) for v in bgr.mean(axis=(0, 1))],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-shot detection on labeled frames.")
    parser.add_argument("--clip", choices=sorted(CLIPS), help="Known clip name.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--detector", required=True, choices=["yolo", "owlv2", "gdino"])
    parser.add_argument("--weights", required=True, help="YOLO filename/path or Hugging Face model id.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--prompt", default="football player")
    parser.add_argument("--conf", type=float, default=0.01, help="Score floor. Sweep thresholds later.")
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--sahi", action="store_true", help="Tiled inference, tile 640, overlap 0.2.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only the first N labeled frames.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    args = parser.parse_args(argv)

    if args.clip:
        spec = CLIPS[args.clip]
        video = spec["video"]
        labels = spec["labels"]
    else:
        if args.video is None or args.labels is None:
            parser.error("pass --clip or both --video and --labels")
        video = args.video
        labels = args.labels

    frame_ids = labeled_frame_ids(labels)
    if args.limit > 0:
        frame_ids = frame_ids[: args.limit]
    print(f"reading {len(frame_ids)} labeled frames from {video.name}", flush=True)
    frames, decode_sec = read_frames(video, frame_ids)
    print(f"frame0 {frame_fingerprint(frames[frame_ids[0]])}", flush=True)

    print(f"loading {args.detector} {args.weights}", flush=True)
    detector = build_detector(args)
    warmup_frame = frames[frame_ids[0]]
    warmup_started = time.perf_counter()
    detector.predict(warmup_frame)
    warmup_sec = time.perf_counter() - warmup_started
    print(f"warmup {warmup_sec:.3f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    next_id = 1
    infer_times = []
    rows = []
    for frame_id in frame_ids:
        image = frames[frame_id]
        started = time.perf_counter()
        detections = detector.predict(image)
        infer_times.append(time.perf_counter() - started)
        height, width = image.shape[:2]
        kept = 0
        for det in detections:
            clipped = clip_xyxy(det["xyxy"], width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            box_w = x2 - x1
            box_h = y2 - y1
            team = assign_team(image, x1, y1, box_w, box_h)
            rows.append(
                {
                    "frame": frame_id,
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
        print(
            f"frame {frame_id}: {kept} boxes in {infer_times[-1]:.3f}s",
            flush=True,
        )

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame", "id", "team", "x", "y", "w", "h", "conf"]
        )
        writer.writeheader()
        writer.writerows(rows)

    steady = infer_times[1:] if len(infer_times) > 1 else infer_times
    timing = {
        "clip": args.clip,
        "video": str(video),
        "labels": str(labels),
        "detector": detector.detail,
        "conf_floor": args.conf,
        "frames": len(frame_ids),
        "boxes": len(rows),
        "decode_sec": round(decode_sec, 4),
        "warmup_sec": round(warmup_sec, 4),
        "sec_per_frame": round(float(np.mean(infer_times)), 4),
        "sec_per_frame_steady": round(float(np.mean(steady)), 4),
        "sec_per_frame_min": round(float(np.min(infer_times)), 4),
        "sec_per_frame_max": round(float(np.max(infer_times)), 4),
        "frame0": frame_fingerprint(frames[frame_ids[0]]),
    }
    timing_path = args.timing
    if timing_path is None:
        timing_path = args.out.with_suffix(".timing.json")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    print(
        f"wrote {args.out} ({len(rows)} boxes), "
        f"sec/frame {timing['sec_per_frame_steady']:.3f} steady "
        f"({timing['sec_per_frame']:.3f} including first)",
        flush=True,
    )
    print(f"wrote {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
