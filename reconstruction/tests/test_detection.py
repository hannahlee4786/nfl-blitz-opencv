import json

import numpy as np
from PIL import Image
import pytest

from nfl_reconstruction.detection import RecordedDetector, export_yolo_dataset
from nfl_reconstruction.schemas import DetectionRecord


def write_detection_fixture(tmp_path, bbox=(100, 50, 140, 130), team="offense"):
    path = tmp_path / "detections.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame": 7,
                "detection_id": "7-0",
                "team": team,
                "confidence": 0.9,
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]
            }
        ) + "\n"
    )
    return path


def test_recorded_detector_returns_ground_contact_midpoint(tmp_path):
    path = write_detection_fixture(tmp_path)

    detection = RecordedDetector(path).detect(
        np.zeros((200, 300, 3), dtype=np.uint8), 7
    )[0]

    assert detection.ground_x == pytest.approx(120.0)
    assert detection.ground_y == pytest.approx(130.0)


def test_recorded_detector_rejects_duplicate_frame_ids(tmp_path):
    path = write_detection_fixture(tmp_path)
    path.write_text(path.read_text() * 2)

    with pytest.raises(ValueError, match="duplicate detection ID"):
        RecordedDetector(path)


def test_export_rejects_label_outside_frame(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (100, 80)).save(frames / "000000.png")
    label = DetectionRecord(
        frame=0, detection_id="0-0", team="offense", confidence=1,
        x1=-1, y1=10, x2=20, y2=30
    )

    with pytest.raises(ValueError, match="frame bounds"):
        export_yolo_dataset([label], frames, tmp_path / "dataset")


def test_export_uses_deterministic_frame_split(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    labels = []
    for frame in range(6):
        Image.new("RGB", (100, 80), (frame, 0, 0)).save(
            frames / f"{frame:06d}.png"
        )
        labels.append(
            DetectionRecord(
                frame=frame, detection_id=f"{frame}-0", team="defense",
                confidence=1, x1=10, y1=10, x2=30, y2=50
            )
        )

    manifest = export_yolo_dataset(labels, frames, tmp_path / "dataset")

    assert manifest.validation_frames == (0, 5)
    assert manifest.training_frames == (1, 2, 3, 4)
    assert len(manifest.source_sha256) == 6
