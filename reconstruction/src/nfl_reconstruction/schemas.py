"""Versioned records exchanged between reconstruction stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """Strict, immutable base for persisted workflow records."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1


class FieldPoint(SchemaModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class ImagePoint(SchemaModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class AnchorSet(SchemaModel):
    frame: int = Field(ge=0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    field_points: tuple[FieldPoint, ...]
    image_points: tuple[ImagePoint, ...]


class CameraState(SchemaModel):
    target_x: float = Field(allow_inf_nan=False)
    target_y: float = Field(allow_inf_nan=False)
    distance: float = Field(gt=0, allow_inf_nan=False)
    yaw_degrees: float = Field(allow_inf_nan=False)
    pitch_degrees: float = Field(allow_inf_nan=False)
    focal_px: float = Field(gt=0, allow_inf_nan=False)
    roll_degrees: float = Field(default=0.0, allow_inf_nan=False)
    # -1 when the calibration labels the sidelines so that the camera sits
    # on the -z side of the field plane, where z = x cross y.
    up_sign: Literal[1, -1] = 1


class CameraFrame(SchemaModel):
    frame: int = Field(ge=0)
    state: CameraState
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    status: Literal["reliable", "uncertain", "missing"] = "reliable"
    reprojection_error_px: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    homography: tuple[tuple[float, float, float], ...] | None = None
    anchor_source: int | None = Field(default=None, ge=0)
    line_error_px: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source: Literal["snapped", "predicted", "filled"] | None = None


class DetectionRecord(SchemaModel):
    frame: int = Field(ge=0)
    detection_id: str = Field(min_length=1)
    team: Literal["offense", "defense", "unknown"] = "unknown"
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    x1: float = Field(allow_inf_nan=False)
    y1: float = Field(allow_inf_nan=False)
    x2: float = Field(allow_inf_nan=False)
    y2: float = Field(allow_inf_nan=False)
    ground_x: float | None = Field(default=None, allow_inf_nan=False)
    ground_y: float | None = Field(default=None, allow_inf_nan=False)


class TrackRecord(SchemaModel):
    frame: int = Field(ge=0)
    track_id: int = Field(ge=0)
    detection_id: str | None = None
    team: Literal["offense", "defense", "unknown"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    screen_x: float = Field(allow_inf_nan=False)
    screen_y: float = Field(allow_inf_nan=False)
    field_x: float | None = Field(default=None, allow_inf_nan=False)
    field_y: float | None = Field(default=None, allow_inf_nan=False)
    camera_confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    projection_status: Literal[
        "reliable", "camera_uncertain", "out_of_bounds", "prediction"
    ]


class FrameRecord(SchemaModel):
    index: int = Field(ge=0)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["decoded", "missing", "error"] = "decoded"
    error: str | None = None


class VideoInfo(SchemaModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps_numerator: int = Field(gt=0)
    fps_denominator: int = Field(gt=0)
    frame_count: int = Field(ge=0)


class StageRecord(SchemaModel):
    name: str
    status: Literal["completed", "reused", "failed", "blocked"]
    input_hash: str
    output_path: str | None = None
    detail: str | None = None


class RunManifest(SchemaModel):
    run_id: str = Field(min_length=1)
    video: VideoInfo
    frames: tuple[FrameRecord, ...] = ()
    stages: dict[str, StageRecord] = {}

