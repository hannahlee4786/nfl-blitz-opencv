"""Turns field homographies into a pinhole camera and back.

Ported from src/camera_model.py on the camera-tracking branch.

The game renders through a virtual pinhole camera: square pixels, no
lens distortion, and the principal point at the center of the picture.
That leaves a focal length plus a position and three angles per frame.

Poses are solved in the field frame of field.py with z = x cross y, so
(x, y, z) is right-handed. Whether z points up or into the ground depends
on how the calibration labeled the two sidelines, so each clip finds out
which side of the field its camera is on (the "up" sign below).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .field import NFLField

_FIELD = NFLField.standard()

# Field points used to compare a camera with a homography: every 5 yards
# along the field and ten steps across it.
GRID_POINTS = np.array(
    [
        (x, y)
        for x in np.arange(0.0, _FIELD.length_yards + 0.1, 5.0)
        for y in np.linspace(0.0, _FIELD.width_yards, 11)
    ],
    dtype=np.float64,
)

# A pose needs this many grid points in view.
MIN_GRID_POINTS = 8

# Hold everything but the focal length fixed in calibrateCamera.
FOCAL_ONLY_FLAGS = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    | cv2.CALIB_FIX_PRINCIPAL_POINT
    | cv2.CALIB_FIX_ASPECT_RATIO
    | cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K1
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
)


def fov_from_focal(focal: float, picture_width: float) -> float:
    """Return the horizontal field of view, in degrees."""

    return float(np.degrees(2 * np.arctan(picture_width / (2 * focal))))


def focal_from_fov(fov: float, picture_width: float) -> float:
    """Return the focal length, in pixels, for a horizontal field of view."""

    return picture_width / (2 * np.tan(np.radians(fov) / 2))


def camera_matrix(
    focal: float,
    principal_point: tuple[float, float],
) -> NDArray[np.float64]:
    cx, cy = principal_point

    return np.array(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ]
    )


def visible_grid_points(
    homography: NDArray[np.float64],
    frame_size: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return the grid points that the homography puts inside the frame,
    as (field_points, screen_points).
    """

    width, height = frame_size

    projected = np.hstack(
        [GRID_POINTS, np.ones((len(GRID_POINTS), 1))]
    ) @ homography.T

    # Points in front of the camera share the sign of their projective w
    # with the field point seen at the image center.
    image_center = np.array([width / 2, height / 2, 1.0])
    center_on_field = np.linalg.inv(homography) @ image_center
    center_on_field /= center_on_field[2]
    front_sign = np.sign((homography @ center_on_field)[2])

    with np.errstate(divide="ignore", invalid="ignore"):
        screen_points = projected[:, :2] / projected[:, 2:3]

        visible = (
            (projected[:, 2] * front_sign > 0)
            & (screen_points[:, 0] >= 0)
            & (screen_points[:, 0] < width)
            & (screen_points[:, 1] >= 0)
            & (screen_points[:, 1] < height)
        )

    return GRID_POINTS[visible], screen_points[visible]


def to_object_points(field_points: NDArray[np.float64]) -> NDArray[np.float32]:
    """Add z = 0 to field points, the format OpenCV expects."""

    return np.hstack(
        [field_points, np.zeros((len(field_points), 1))]
    ).astype(np.float32)


def fit_shared_focal(
    homographies: NDArray[np.float64],
    frame_size: tuple[int, int],
    principal_point: tuple[float, float],
    focal: float,
) -> tuple[float, float]:
    """
    Fit one focal length to many frames at once, starting from `focal`.

    Returns (focal, rms), where rms is the reprojection error in pixels.
    Frames with too little of the field in view are skipped.
    """

    object_points = []
    image_points = []

    for homography in homographies:
        field_points, screen_points = visible_grid_points(
            homography,
            frame_size,
        )

        if len(field_points) < MIN_GRID_POINTS:
            continue

        object_points.append(to_object_points(field_points))
        image_points.append(screen_points.astype(np.float32))

    rms, matrix, _, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        frame_size,
        camera_matrix(focal, principal_point),
        np.zeros(5),
        flags=FOCAL_ONLY_FLAGS,
    )

    return float(matrix[0, 0]), float(rms)


def fit_pose(
    homography: NDArray[np.float64],
    matrix: NDArray[np.float64],
    frame_size: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
    """
    Return the camera pose (rvec, tvec) that best reproduces the
    homography, and its reprojection error in pixels. Return None when
    too little of the field is in view.
    """

    field_points, screen_points = visible_grid_points(
        homography,
        frame_size,
    )

    if len(field_points) < MIN_GRID_POINTS:
        return None

    object_points = to_object_points(field_points)
    image_points = screen_points.astype(np.float32)

    # IPPE is made for points on a plane.
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        matrix,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        matrix,
        None,
        rvec,
        tvec,
    )

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        matrix,
        None,
    )

    error = np.sqrt(
        np.mean(
            np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)
        )
    )

    return rvec.reshape(3), tvec.reshape(3), float(error)


def camera_center(
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    return -rotation.T @ translation


def find_up_sign(
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> float:
    """
    Return +1 when the camera is on the +z side of the field, else -1.
    The camera is always above the field, so that side is up.
    """

    return 1.0 if camera_center(rotation, translation)[2] > 0 else -1.0


def pose_to_parameters(
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    up_sign: float,
) -> NDArray[np.float64]:
    """
    Return (x, y, height, pan, tilt, roll).

    x and y are the camera's spot on the field and height is above it,
    all in yards. Angles are in degrees. Pan is the viewing direction on
    the field map (0 toward +x, 90 toward +y), tilt is how far below
    level it looks, and roll is 0 when the horizon is level.
    """

    down = np.array([0.0, 0.0, -up_sign])
    right, image_down, forward = rotation

    x, y, z = camera_center(rotation, translation)

    pan = np.degrees(np.arctan2(forward[1], forward[0]))
    tilt = np.degrees(np.arcsin(np.clip(forward @ down, -1.0, 1.0)))
    roll = np.degrees(np.arctan2(right @ down, image_down @ down))

    return np.array([x, y, up_sign * z, pan, tilt, roll])


def parameters_to_pose(
    parameters: NDArray[np.float64],
    up_sign: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of pose_to_parameters. Returns (rotation, translation)."""

    x, y, height, pan, tilt, roll = parameters
    pan, tilt, roll = np.radians([pan, tilt, roll])

    down = np.array([0.0, 0.0, -up_sign])

    forward = np.array(
        [
            np.cos(tilt) * np.cos(pan),
            np.cos(tilt) * np.sin(pan),
            0.0,
        ]
    ) + np.sin(tilt) * down

    # With no roll, the image's down direction is the world's down,
    # minus the part along the viewing direction.
    level_down = down - (down @ forward) * forward
    level_down /= np.linalg.norm(level_down)
    level_right = np.cross(level_down, forward)

    right = np.cos(roll) * level_right + np.sin(roll) * level_down
    image_down = -np.sin(roll) * level_right + np.cos(roll) * level_down

    rotation = np.array([right, image_down, forward])
    center = np.array([x, y, up_sign * height])

    return rotation, -rotation @ center


def pose_homography(
    matrix: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the field-to-screen homography of a camera."""

    homography = matrix @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation]
    )

    return homography / homography[2, 2]


def smooth_series(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    strength: float,
) -> NDArray[np.float64]:
    """
    Smooth evenly spaced values, trusting each one by its weight.

    Minimizes the weighted squared distance to the values plus
    `strength` times the squared second differences (a Whittaker
    smoother). Frames with weight 0 are filled in from their neighbors.
    """

    count = len(values)

    if count < 3:
        return np.asarray(values, dtype=np.float64)

    second_difference = np.diff(np.eye(count), 2, axis=0)

    system = (
        np.diag(weights)
        + strength * second_difference.T @ second_difference
    )

    return np.linalg.solve(system, weights * np.asarray(values))
