"""Confidence-aware player projection and short-occlusion association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .camera import unproject_image
from .field import NFLField
from .schemas import CameraFrame, DetectionRecord, TrackRecord


@dataclass(frozen=True)
class TrackingConfig:
    max_staleness: int = 3
    max_center_distance: float = 60.0
    reliable_camera_confidence: float = 0.25


@dataclass(frozen=True)
class ProjectedDetection:
    field_x: float | None
    field_y: float | None
    projection_status: str


@dataclass
class _Track:
    track_id: int
    team: str
    x: float
    y: float
    width: float
    height: float
    vx: float = 0.0
    vy: float = 0.0
    stale: int = 0
    confidence: float = 1.0


def project_detection(
    detection: DetectionRecord,
    camera: CameraFrame,
    field: NFLField,
    confidence_threshold: float = 0.25,
) -> ProjectedDetection:
    if (
        camera.confidence < confidence_threshold
        or camera.status != "reliable"
        or camera.homography is None
    ):
        return ProjectedDetection(None, None, "camera_uncertain")
    point = torch.tensor(
        [[detection.ground_x, detection.ground_y]], dtype=torch.float64
    )
    homography = torch.tensor(camera.homography, dtype=torch.float64)
    projected = unproject_image(point, homography)[0].detach().numpy()
    x, y = float(projected[0]), float(projected[1])
    status = "reliable" if field.contains([[x, y]])[0] else "out_of_bounds"
    return ProjectedDetection(x, y, status)


def _iou(track: _Track, detection: DetectionRecord) -> float:
    left = max(track.x - track.width / 2, detection.x1)
    right = min(track.x + track.width / 2, detection.x2)
    top = max(track.y - track.height, detection.y1)
    bottom = min(track.y, detection.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = track.width * track.height + (
        detection.x2 - detection.x1
    ) * (detection.y2 - detection.y1) - intersection
    return intersection / union if union > 0 else 0.0


def _record(
    track: _Track,
    frame: int,
    camera: CameraFrame,
    field: NFLField,
    detection: DetectionRecord | None,
    config: TrackingConfig,
) -> TrackRecord:
    if detection is None:
        synthetic = DetectionRecord(
            frame=frame,
            detection_id=f"prediction-{track.track_id}-{frame}",
            team=track.team,
            confidence=track.confidence,
            x1=track.x - track.width / 2,
            y1=track.y - track.height,
            x2=track.x + track.width / 2,
            y2=track.y,
            ground_x=track.x,
            ground_y=track.y,
        )
        projected = project_detection(
            synthetic, camera, field, config.reliable_camera_confidence
        )
        status = "prediction"
        detection_id = None
    else:
        projected = project_detection(
            detection, camera, field, config.reliable_camera_confidence
        )
        status = projected.projection_status
        detection_id = detection.detection_id
    return TrackRecord(
        frame=frame,
        track_id=track.track_id,
        detection_id=detection_id,
        team=track.team,
        confidence=track.confidence,
        screen_x=track.x,
        screen_y=track.y,
        field_x=projected.field_x,
        field_y=projected.field_y,
        camera_confidence=camera.confidence,
        projection_status=status,
    )


def track_players(
    detections: list[list[DetectionRecord]],
    cameras: list[CameraFrame],
    field: NFLField,
    config: TrackingConfig = TrackingConfig(),
) -> list[TrackRecord]:
    if len(detections) != len(cameras):
        raise ValueError("Detection and camera frame counts must match.")
    active: dict[int, _Track] = {}
    next_id = 0
    records = []
    for frame, (frame_detections, camera) in enumerate(
        zip(detections, cameras, strict=True)
    ):
        tracks = list(active.values())
        previous = {track.track_id: (track.x, track.y) for track in tracks}
        for track in tracks:
            track.x += track.vx
            track.y += track.vy
            track.stale += 1
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        if tracks and frame_detections:
            costs = np.full((len(tracks), len(frame_detections)), 1e6)
            for row, track in enumerate(tracks):
                for column, detection in enumerate(frame_detections):
                    if track.team != detection.team:
                        continue
                    x = detection.ground_x
                    y = detection.ground_y
                    distance = float(np.hypot(track.x - x, track.y - y))
                    if distance <= config.max_center_distance:
                        costs[row, column] = distance + 20.0 * (
                            1.0 - _iou(track, detection)
                        )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] >= 1e6:
                    continue
                track = tracks[row]
                detection = frame_detections[column]
                old_x, old_y = previous[track.track_id]
                track.vx = detection.ground_x - old_x
                track.vy = detection.ground_y - old_y
                track.x = detection.ground_x
                track.y = detection.ground_y
                track.width = detection.x2 - detection.x1
                track.height = detection.y2 - detection.y1
                track.stale = 0
                track.confidence = detection.confidence
                matched_tracks.add(track.track_id)
                matched_detections.add(column)
                records.append(_record(track, frame, camera, field, detection, config))
        for track in tracks:
            if track.track_id not in matched_tracks and track.stale <= config.max_staleness:
                records.append(_record(track, frame, camera, field, None, config))
            if track.stale > config.max_staleness:
                active.pop(track.track_id, None)
        for index, detection in enumerate(frame_detections):
            if index in matched_detections:
                continue
            track = _Track(
                track_id=next_id,
                team=detection.team,
                x=detection.ground_x,
                y=detection.ground_y,
                width=detection.x2 - detection.x1,
                height=detection.y2 - detection.y1,
                confidence=detection.confidence,
            )
            active[next_id] = track
            next_id += 1
            records.append(_record(track, frame, camera, field, detection, config))
    return records
