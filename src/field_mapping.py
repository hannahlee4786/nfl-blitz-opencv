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


def step_vectors(homography, field_points):
    """
    Screen vectors (pixels) of a one-yard step along field x and along
    field y, starting at each field point. Returns (along_x, along_y).
    """

    points = np.asarray(field_points, dtype=np.float64).reshape(-1, 2)
    count = len(points)

    if count == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty

    stacked = np.vstack(
        [points, points + [1.0, 0.0], points + [0.0, 1.0]]
    ).astype(np.float32)

    projected = cv2.perspectiveTransform(
        stacked.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)

    origin = projected[:count]

    return (
        projected[count:2 * count] - origin,
        projected[2 * count:] - origin,
    )


def pixels_per_yard(homography, field_points):
    """
    Ground-plane scale: sqrt of the screen area of a 1 x 1 yard ground
    cell. Fine for ground distances; for the height of an upright
    player use upright_scale instead.
    """

    along_x, along_y = step_vectors(homography, field_points)

    area = np.abs(
        along_x[:, 0] * along_y[:, 1] - along_x[:, 1] * along_y[:, 0]
    )

    return np.sqrt(area)


def upright_scale(homography, field_points):
    """
    Pixels per yard for an upright object standing at each field point.

    A standing player is scaled like the ground direction that runs
    sideways on screen (across the field when the camera looks down the
    field), not like the depth direction, which the camera tilt
    squashes. The more horizontal of the two ground steps is used, so
    this also holds if the camera is rotated.
    """

    along_x, along_y = step_vectors(homography, field_points)

    length_x = np.linalg.norm(along_x, axis=1)
    length_y = np.linalg.norm(along_y, axis=1)

    horizontal_x = np.abs(along_x[:, 0]) / np.maximum(length_x, 1e-9)
    horizontal_y = np.abs(along_y[:, 0]) / np.maximum(length_y, 1e-9)

    return np.where(horizontal_x >= horizontal_y, length_x, length_y)


def front_sign(homography, width, height):
    """
    Sign of the projective w for points in front of the camera, taken
    from the point seen at the image center (same rule as draw_field_grid).
    """

    inverse = invert_homography(homography)

    if inverse is None:
        return 1.0

    center = inverse @ np.array([width / 2.0, height / 2.0, 1.0])

    if abs(center[2]) < 1e-12:
        return 1.0

    center = center / center[2]
    sign = np.sign((homography @ center)[2])

    return float(sign) if sign != 0 else 1.0


def field_coordinate_grid(homography, width, height, step=4):
    """
    Field coordinates of a coarse grid of screen pixels (every `step`
    pixels). Returns (field_x, field_y, valid), each shaped like the
    grid. `valid` is False for pixels that do not land on the ground in
    front of the camera (above the horizon).
    """

    inverse = invert_homography(homography)

    if inverse is None:
        return None

    xs = np.arange(step // 2, width, step, dtype=np.float64)
    ys = np.arange(step // 2, height, step, dtype=np.float64)

    u, v = np.meshgrid(xs, ys)

    pixels = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    mapped = inverse @ pixels

    center = inverse @ np.array([width / 2.0, height / 2.0, 1.0])
    front = np.sign(center[2]) if center[2] != 0 else 1.0

    w = mapped[2]
    valid = (w * front) > 1e-6 * abs(center[2])
    safe_w = np.where(valid, w, 1.0)

    field_x = (mapped[0] / safe_w).reshape(u.shape)
    field_y = (mapped[1] / safe_w).reshape(u.shape)

    return field_x, field_y, valid.reshape(u.shape)


def attach_field_positions(detections, inverse_homography, margin=1.0):
    """
    Set detection.field_xy from each detection's foot point and drop
    detections that land outside the field (plus a margin in yards).
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