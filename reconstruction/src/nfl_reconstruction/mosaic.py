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
    rows, columns = np.indices((canvas_height, canvas_width))
    field_points = np.stack(
        [
            columns.ravel() / pixels_per_yard,
            rows.ravel() / pixels_per_yard,
            np.ones(canvas_height * canvas_width),
        ],
        axis=1,
    )
    samples = []
    valid_samples = []
    for frame, camera, mask in zip(
        frames, camera_frames, dynamic_masks, strict=True
    ):
        if mask.shape != frame.shape[:2]:
            raise ValueError(
                f"Dynamic mask shape {mask.shape} does not match frame "
                f"shape {frame.shape[:2]}."
            )
        if camera.confidence < confidence_threshold or camera.homography is None:
            continue
        homography = np.asarray(camera.homography, dtype=float)
        projected = field_points @ homography.T
        projected = projected[:, :2] / projected[:, 2:3]
        pixels = np.rint(projected).astype(int)
        valid = (
            np.isfinite(projected).all(axis=1)
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < frame.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < frame.shape[0])
        )
        indices = np.where(valid)[0]
        valid[indices] &= ~mask[pixels[indices, 1], pixels[indices, 0]]
        warped = np.zeros((len(field_points), 3), dtype=np.uint8)
        indices = np.where(valid)[0]
        warped[indices] = frame[pixels[indices, 1], pixels[indices, 0]]
        samples.append(warped.reshape(canvas_height, canvas_width, 3))
        valid_samples.append(valid.reshape(canvas_height, canvas_width))

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
