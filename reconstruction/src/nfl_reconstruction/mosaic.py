"""Confidence-gated accumulation into a static top-down field mosaic."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray

from .field import NFLField
from .schemas import CameraFrame


@dataclass(frozen=True)
class MosaicResult:
    image: NDArray[np.uint8]
    coverage: NDArray[np.bool_]
    sample_count: NDArray[np.uint16]


@dataclass(frozen=True)
class FieldWarp:
    image: NDArray[np.uint8]
    coverage: NDArray[np.bool_]


def warp_frame_to_field(
    frame: NDArray[np.uint8],
    camera: CameraFrame,
    dynamic_mask: NDArray[np.bool_],
    field: NFLField,
    pixels_per_yard: int = 10,
) -> FieldWarp:
    if pixels_per_yard <= 0:
        raise ValueError("Pixels per yard must be positive.")
    if dynamic_mask.shape != frame.shape[:2]:
        raise ValueError(
            f"Dynamic mask shape {dynamic_mask.shape} does not match frame "
            f"shape {frame.shape[:2]}."
        )

    canvas_width = round(field.length_yards * pixels_per_yard)
    canvas_height = round(field.width_yards * pixels_per_yard)
    image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    coverage = np.zeros((canvas_height, canvas_width), dtype=bool)
    if camera.homography is None:
        return FieldWarp(image=image, coverage=coverage)

    rows, columns = np.indices((canvas_height, canvas_width))
    field_points = np.stack(
        (
            columns.ravel() / pixels_per_yard,
            rows.ravel() / pixels_per_yard,
            np.ones(canvas_height * canvas_width),
        ),
        axis=1,
    )
    projected = field_points @ np.asarray(camera.homography, dtype=float).T
    finite_scale = np.isfinite(projected[:, 2]) & (np.abs(projected[:, 2]) > 1e-9)
    screen = np.full((len(field_points), 2), np.nan, dtype=float)
    screen[finite_scale] = projected[finite_scale, :2] / projected[finite_scale, 2:3]
    pixels = np.zeros((len(field_points), 2), dtype=int)
    finite_screen = np.isfinite(screen).all(axis=1)
    pixels[finite_screen] = np.rint(screen[finite_screen]).astype(int)
    valid = (
        finite_screen
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < frame.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < frame.shape[0])
    )
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= ~dynamic_mask[
        pixels[valid_indices, 1], pixels[valid_indices, 0]
    ]
    valid_indices = np.flatnonzero(valid)
    image.reshape(-1, 3)[valid_indices] = frame[
        pixels[valid_indices, 1], pixels[valid_indices, 0]
    ]
    coverage.reshape(-1)[valid_indices] = True
    return FieldWarp(image=image, coverage=coverage)


def build_mosaic(
    frames: list[NDArray[np.uint8]],
    camera_frames: list[CameraFrame],
    dynamic_masks: list[NDArray[np.bool_]] | None,
    field: NFLField,
    pixels_per_yard: int = 10,
    confidence_threshold: float = 0.25,
) -> MosaicResult:
    if len(frames) != len(camera_frames):
        raise ValueError("Frames and camera records must have the same length.")
    if dynamic_masks is None:
        dynamic_masks = [np.zeros(frame.shape[:2], dtype=bool) for frame in frames]
    if len(dynamic_masks) != len(frames):
        raise ValueError("Dynamic mask count must match frame count.")
    if pixels_per_yard <= 0:
        raise ValueError("Pixels per yard must be positive.")

    canvas_width = round(field.length_yards * pixels_per_yard)
    canvas_height = round(field.width_yards * pixels_per_yard)
    samples = []
    valid_samples = []
    for frame, camera, mask in zip(
        frames, camera_frames, dynamic_masks, strict=True
    ):
        if camera.confidence < confidence_threshold or camera.homography is None:
            continue
        warped = warp_frame_to_field(
            frame, camera, mask, field, pixels_per_yard=pixels_per_yard
        )
        samples.append(warped.image)
        valid_samples.append(warped.coverage)

    if not samples:
        return MosaicResult(
            image=np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8),
            coverage=np.zeros((canvas_height, canvas_width), dtype=bool),
            sample_count=np.zeros((canvas_height, canvas_width), dtype=np.uint16),
        )
    values = np.stack(samples).astype(float)
    validity = np.stack(valid_samples)
    values[~validity, :] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        image = np.nanmedian(values, axis=0)
    image = np.nan_to_num(image, nan=0.0).astype(np.uint8)
    count = validity.sum(axis=0).astype(np.uint16)
    return MosaicResult(image=image, coverage=count > 0, sample_count=count)
