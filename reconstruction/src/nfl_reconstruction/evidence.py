"""Field-marking evidence with explicit HUD and foreground exclusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass(frozen=True)
class EvidenceMap:
    gradient: NDArray[np.float32]
    line_likelihood: NDArray[np.float32]
    distance: NDArray[np.float32]
    excluded: NDArray[np.bool_]


def build_exclusion_mask(
    image_size: tuple[int, int],
    rectangles: list[tuple[float, float, float, float]],
    dynamic_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    width, height = image_size
    mask = np.zeros((height, width), dtype=bool)
    if dynamic_mask is not None:
        if dynamic_mask.shape != mask.shape:
            raise ValueError(
                f"Dynamic mask shape {dynamic_mask.shape} does not match "
                f"image mask shape {mask.shape}."
            )
        mask |= dynamic_mask.astype(bool)
    for rectangle in rectangles:
        x1, y1, x2, y2 = rectangle
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Exclusion rectangles must be normalized and ordered.")
        left, right = int(np.floor(x1 * width)), int(np.ceil(x2 * width))
        top, bottom = int(np.floor(y1 * height)), int(np.ceil(y2 * height))
        mask[top:bottom, left:right] = True
    return mask


def _normalize(values: NDArray[np.float64], valid: NDArray[np.bool_]):
    result = np.zeros_like(values, dtype=np.float32)
    if valid.any():
        scale = float(np.percentile(values[valid], 99))
        if scale > 0:
            result[valid] = np.clip(values[valid] / scale, 0, 1)
    return result


def field_evidence(
    frame: NDArray[np.uint8],
    exclusion_mask: NDArray[np.bool_],
) -> EvidenceMap:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Field evidence expects an RGB image with three channels.")
    if exclusion_mask.shape != frame.shape[:2]:
        raise ValueError("Exclusion mask shape must match the frame dimensions.")
    rgb = frame.astype(np.float64) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    gradient_raw = np.hypot(
        ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1)
    )
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    brightness = rgb.max(axis=2)
    line_raw = brightness * np.clip(1.0 - saturation, 0.0, 1.0)
    valid = ~exclusion_mask
    gradient = _normalize(gradient_raw, valid)
    line_likelihood = _normalize(line_raw, valid)
    gradient[exclusion_mask] = 0
    line_likelihood[exclusion_mask] = 0
    distance = ndimage.distance_transform_edt(line_likelihood < 0.25).astype(
        np.float32
    )
    distance[exclusion_mask] = float(max(frame.shape[:2]))
    return EvidenceMap(gradient, line_likelihood, distance, exclusion_mask.copy())

