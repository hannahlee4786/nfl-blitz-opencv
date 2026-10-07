"""Quantitative diagnostics for a reconstruction run."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import numpy as np

from .schemas import CameraFrame, TrackRecord


class EvaluationReport(BaseModel):
    """Summary metrics used to compare and gate reconstruction runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    median_reprojection_error_px: float | None = None
    p95_reprojection_error_px: float | None = None
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

    ordered = sorted(camera_frames, key=lambda frame: frame.frame)
    smoothness = None
    if len(ordered) >= 3:
        targets = np.asarray(
            [(frame.state.target_x, frame.state.target_y) for frame in ordered]
        )
        acceleration = np.diff(targets, n=2, axis=0)
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
        camera_smoothness=smoothness,
        detection_recall=recall,
        out_of_bounds_rate=out_of_bounds_rate,
        track_fragments=fragments,
    )
