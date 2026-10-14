"""Quantitative diagnostics for a reconstruction run."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field
import numpy as np

from .line_score import get_tracking_status
from .schemas import CameraFrame, TrackRecord


class EvaluationReport(BaseModel):
    """Summary metrics used to compare and gate reconstruction runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    median_reprojection_error_px: float | None = None
    p95_reprojection_error_px: float | None = None
    median_line_error_px: float | None = None
    p95_line_error_px: float | None = None
    lost_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_smoothness: float | None = None
    detection_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    out_of_bounds_rate: float = Field(ge=0.0, le=1.0)
    track_fragments: int = Field(ge=0)


def evaluate_run(
    camera_frames: list[CameraFrame],
    tracks: list[TrackRecord],
    *,
    annotated_player_count: int = 0,
    matched_detection_count: int = 0,
    expected_track_count: int | None = None,
) -> EvaluationReport:
    """Compute camera, detector, and tracking quality metrics."""

    errors = np.asarray(
        [
            frame.reprojection_error_px
            for frame in camera_frames
            if frame.reprojection_error_px is not None
        ],
        dtype=float,
    )
    median_error = float(np.median(errors)) if errors.size else None
    p95_error = float(np.percentile(errors, 95)) if errors.size else None

    line_errors = np.asarray(
        [
            frame.line_error_px
            for frame in camera_frames
            if frame.line_error_px is not None
        ],
        dtype=float,
    )
    median_line_error = float(np.median(line_errors)) if line_errors.size else None
    p95_line_error = float(np.percentile(line_errors, 95)) if line_errors.size else None
    lost_fraction = None
    if line_errors.size:
        # Frames whose grid could not be measured count as lost.
        lost = sum(
            get_tracking_status(
                math.nan if frame.line_error_px is None else frame.line_error_px
            )
            == "LOST"
            for frame in camera_frames
        )
        lost_fraction = lost / len(camera_frames)

    ordered = sorted(camera_frames, key=lambda frame: frame.frame)
    smoothness = None
    if len(ordered) >= 3:
        if all(frame.homography is not None for frame in ordered):
            field_center = np.asarray([60.0, 26.6665, 1.0])
            positions = []
            for frame in ordered:
                projected = np.asarray(frame.homography) @ field_center
                positions.append(projected[:2] / projected[2])
            trajectory = np.asarray(positions)
        else:
            trajectory = np.asarray(
                [(frame.state.target_x, frame.state.target_y) for frame in ordered]
            )
        acceleration = np.diff(trajectory, n=2, axis=0)
        smoothness = float(np.mean(np.linalg.norm(acceleration, axis=1)))

    recall = None
    if annotated_player_count > 0:
        recall = min(1.0, matched_detection_count / annotated_player_count)

    out_of_bounds = sum(
        track.projection_status == "out_of_bounds"
        or (
            track.field_x is not None
            and track.field_y is not None
            and not (0.0 <= track.field_x <= 120.0 and 0.0 <= track.field_y <= 53.333)
        )
        for track in tracks
    )
    out_of_bounds_rate = out_of_bounds / len(tracks) if tracks else 0.0

    fragments = 0
    if expected_track_count is not None:
        observed_tracks = len({track.track_id for track in tracks})
        fragments = max(0, observed_tracks - expected_track_count)

    return EvaluationReport(
        median_reprojection_error_px=median_error,
        p95_reprojection_error_px=p95_error,
        median_line_error_px=median_line_error,
        p95_line_error_px=p95_line_error,
        lost_fraction=lost_fraction,
        camera_smoothness=smoothness,
        detection_recall=recall,
        out_of_bounds_rate=out_of_bounds_rate,
        track_fragments=fragments,
    )
