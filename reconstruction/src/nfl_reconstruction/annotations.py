"""Loading and validating manual field-to-image anchors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .field import NFLField
from .schemas import AnchorSet


def _coordinates(points) -> NDArray[np.float64]:
    return np.asarray([(point.x, point.y) for point in points], dtype=float)


def validate_anchor(
    anchor: AnchorSet,
    field: NFLField,
    image_size: tuple[int, int],
) -> None:
    """Reject anchors that cannot determine a field-to-image homography."""

    width, height = image_size
    field_points = _coordinates(anchor.field_points)
    image_points = _coordinates(anchor.image_points)

    if len(field_points) != len(image_points) or len(field_points) < 4:
        raise ValueError("An anchor requires at least four matching point pairs.")
    if len(np.unique(field_points, axis=0)) != len(field_points) or len(
        np.unique(image_points, axis=0)
    ) != len(image_points):
        raise ValueError("Anchor point pairs must be unique.")
    if not field.contains(field_points).all():
        raise ValueError("Anchor field points must be inside field bounds.")
    if (
        (image_points[:, 0] < 0)
        | (image_points[:, 0] >= width)
        | (image_points[:, 1] < 0)
        | (image_points[:, 1] >= height)
    ).any():
        raise ValueError("Anchor image points must be inside image bounds.")
    if np.linalg.matrix_rank(field_points - field_points.mean(axis=0)) < 2:
        raise ValueError("Anchor field points must be non-collinear.")
    if np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2:
        raise ValueError("Anchor image points must be non-collinear.")


def load_anchor(
    path: str | Path,
    image_size: tuple[int, int],
) -> AnchorSet:
    """Load the compact JSON representation used by annotation tools."""

    payload = json.loads(Path(path).read_text())
    payload["field_points"] = [
        {"x": point[0], "y": point[1]} for point in payload["field_points"]
    ]
    payload["image_points"] = [
        {"x": point[0], "y": point[1]} for point in payload["image_points"]
    ]
    anchor = AnchorSet.model_validate(payload)
    if (anchor.image_width, anchor.image_height) != image_size:
        raise ValueError(
            "Anchor image dimensions do not match the requested image size."
        )
    validate_anchor(anchor, NFLField.standard(), image_size)
    return anchor


def _normalize_points(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = np.linalg.norm(shifted, axis=1).mean()
    if mean_distance <= np.finfo(float).eps:
        raise ValueError("Anchor points do not span a usable area.")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def anchor_homography(anchor: AnchorSet) -> NDArray[np.float64]:
    """Estimate a normalized-DLT homography from field yards to pixels."""

    validate_anchor(
        anchor,
        NFLField.standard(),
        (anchor.image_width, anchor.image_height),
    )
    source, source_transform = _normalize_points(
        _coordinates(anchor.field_points)
    )
    target, target_transform = _normalize_points(
        _coordinates(anchor.image_points)
    )
    rows = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows)
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("Anchor pairs do not determine a unique homography.")
    _, _, vectors = np.linalg.svd(design)
    normalized_homography = vectors[-1].reshape(3, 3)
    homography = (
        np.linalg.inv(target_transform)
        @ normalized_homography
        @ source_transform
    )
    if abs(homography[2, 2]) <= np.finfo(float).eps:
        raise ValueError("Anchor homography has an invalid scale.")
    homography /= homography[2, 2]
    return homography

