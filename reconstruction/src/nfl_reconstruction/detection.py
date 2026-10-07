"""Replaceable player-detector adapters and dataset export."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .schemas import DetectionRecord


class PlayerDetector(Protocol):
    def detect(
        self, frame: NDArray[np.uint8], frame_index: int
    ) -> list[DetectionRecord]: ...


@dataclass(frozen=True)
class DatasetManifest:
    training_frames: tuple[int, ...]
    validation_frames: tuple[int, ...]
    source_sha256: dict[int, str]


class RecordedDetector:
    def __init__(self, path: str | Path):
        self._by_frame: dict[int, list[DetectionRecord]] = {}
        identifiers: set[tuple[int, str]] = set()
        for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = DetectionRecord.model_validate_json(line)
            key = (record.frame, record.detection_id)
            if key in identifiers:
                raise ValueError(
                    f"Found duplicate detection ID {record.detection_id!r} "
                    f"in frame {record.frame} at line {line_number}."
                )
            identifiers.add(key)
            if record.ground_x is None or record.ground_y is None:
                record = record.model_copy(
                    update={
                        "ground_x": (record.x1 + record.x2) / 2,
                        "ground_y": record.y2,
                    }
                )
            self._by_frame.setdefault(record.frame, []).append(record)

    def detect(
        self, frame: NDArray[np.uint8], frame_index: int
    ) -> list[DetectionRecord]:
        del frame
        return list(self._by_frame.get(frame_index, ()))


class UltralyticsDetector:
    def __init__(self, weights: str | Path, confidence: float = 0.25):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics support is optional; install it with "
                "pip install -e '.[detector]'."
            ) from error
        self._model = YOLO(str(weights))
        self._confidence = confidence

    def detect(
        self, frame: NDArray[np.uint8], frame_index: int
    ) -> list[DetectionRecord]:
        result = self._model.predict(frame, conf=self._confidence, verbose=False)[0]
        records = []
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            class_name = str(result.names[int(box.cls.item())]).lower()
            team = class_name if class_name in {"offense", "defense"} else "unknown"
            records.append(
                DetectionRecord(
                    frame=frame_index,
                    detection_id=f"{frame_index}-{index}",
                    team=team,
                    confidence=float(box.conf.item()),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    ground_x=(x1 + x2) / 2,
                    ground_y=y2,
                )
            )
        return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_yolo_dataset(
    labels: list[DetectionRecord],
    frames_dir: Path,
    output_dir: Path,
) -> DatasetManifest:
    by_frame: dict[int, list[DetectionRecord]] = {}
    for label in labels:
        by_frame.setdefault(label.frame, []).append(label)
    training, validation, hashes = [], [], {}
    class_ids = {"offense": 0, "defense": 1, "unknown": 2}
    for frame_index in sorted(by_frame):
        source = frames_dir / f"{frame_index:06d}.png"
        if not source.is_file():
            raise FileNotFoundError(f"Missing labeled frame: {source}")
        with Image.open(source) as image:
            width, height = image.size
        split = "val" if frame_index % 5 == 0 else "train"
        (validation if split == "val" else training).append(frame_index)
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for label in by_frame[frame_index]:
            if not (
                0 <= label.x1 < label.x2 <= width
                and 0 <= label.y1 < label.y2 <= height
            ):
                raise ValueError(
                    f"Detection {label.detection_id} is outside frame bounds."
                )
            center_x = (label.x1 + label.x2) / (2 * width)
            center_y = (label.y1 + label.y2) / (2 * height)
            box_width = (label.x2 - label.x1) / width
            box_height = (label.y2 - label.y1) / height
            rows.append(
                f"{class_ids[label.team]} {center_x:.8f} {center_y:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            )
        shutil.copy2(source, images_dir / source.name)
        (labels_dir / f"{frame_index:06d}.txt").write_text("\n".join(rows) + "\n")
        hashes[frame_index] = _sha256(source)
    return DatasetManifest(tuple(training), tuple(validation), hashes)
