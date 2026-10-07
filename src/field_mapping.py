# Converts between gameplay-screen pixels and field coordinates (yards)
import cv2
import numpy as np

from src.field_model import FIELD_LENGTH, FIELD_WIDTH


def invert_homography(homography):
    """Return screen -> field, or None if the homography is degenerate."""

    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(inverse)):
        return None

    return inverse


def screen_to_field(screen_points, inverse_homography):
    """Map (N, 2) screen pixels to (N, 2) field yards."""

    points = np.asarray(screen_points, dtype=np.float32).reshape(-1, 2)

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),
        inverse_homography,
    ).reshape(-1, 2)


def field_to_screen(field_points, homography):
    """Map (N, 2) field yards to (N, 2) screen pixels."""

    points = np.asarray(field_points, dtype=np.float32).reshape(-1, 2)

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)


def pixels_per_yard(homography, field_points):
    """
    Local image scale at each field point: how many pixels one yard
    spans there. Lets size filters be written in yards, so they hold
    across zoom levels and clips.
    """

    points = np.asarray(field_points, dtype=np.float64).reshape(-1, 2)
    count = len(points)

    stacked = np.vstack(
        [points, points + [1.0, 0.0], points + [0.0, 1.0]]
    ).astype(np.float32)

    projected = cv2.perspectiveTransform(
        stacked.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)

    origin = projected[:count]
    along_x = projected[count:2 * count] - origin
    along_y = projected[2 * count:] - origin

    area = np.abs(
        along_x[:, 0] * along_y[:, 1] - along_x[:, 1] * along_y[:, 0]
    )

    return np.sqrt(area)


def attach_field_positions(detections, inverse_homography, margin=3.0):
    """
    Set detection.field_xy from each detection's foot point and drop
    detections that land outside the field (plus a margin in yards),
    such as crowd or sideline blobs.
    """

    if not detections:
        return []

    feet = np.array([d.foot for d in detections], dtype=np.float32)
    field_points = screen_to_field(feet, inverse_homography)

    kept = []

    for detection, point in zip(detections, field_points):
        if not np.all(np.isfinite(point)):
            continue

        if not (-margin <= point[0] <= FIELD_LENGTH + margin):
            continue

        if not (-margin <= point[1] <= FIELD_WIDTH + margin):
            continue

        detection.field_xy = point.astype(np.float64)
        kept.append(detection)

    return kept