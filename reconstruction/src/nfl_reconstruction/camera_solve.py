"""Solves a pinhole camera for every tracked frame and smooths its path.

Ported from src/solve_camera.py on the camera-tracking branch. Each tracked
grid gives a pose. The clip gets one focal length unless the fit shows the
lens changing, and the poses are smoothed over time, trusting each frame by
how its grid was found.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .camera_model import (
    camera_matrix,
    find_up_sign,
    fit_pose,
    fit_shared_focal,
    focal_from_fov,
    fov_from_focal,
    parameters_to_pose,
    pose_homography,
    pose_to_parameters,
    smooth_series,
)
from .schemas import CameraState

# Starting guess for the focal length. Every clip so far is 44 to 47
# degrees wide.
INITIAL_FOV = 46.0

# Frames per window when checking whether the focal length changes.
FOCAL_WINDOW = 30

# Let the focal length change only when per-window focal lengths cut the
# camera's fit error by more than this fraction. More freedom always
# helps a little: clips with a fixed lens gain 12 to 18 percent.
ZOOM_IMPROVEMENT = 0.3

# How strongly to smooth the camera path. Higher is smoother but cuts
# corners on fast moves.
SMOOTHING = 2.0

# Where tracking is lost at the start or end of a clip, carry the camera's
# motion on for at most this many frames before holding it still.
COAST_FRAMES = 10

# How much each frame counts when smoothing, by how it was tracked.
# Frames with weight 0 are filled in from their neighbors.
SOURCE_WEIGHTS = {
    "snapped": 1.0,
    "predicted": 0.2,
    "filled": 0.0,
}


@dataclass(frozen=True)
class CameraSolution:
    """The smoothed camera of every frame, and what it was solved from."""

    sources: NDArray[np.str_]
    focals: NDArray[np.float64]
    fovs: NDArray[np.float64]
    window_fovs: list[float]
    lens_changes: bool
    principal_point: tuple[float, float]
    up_sign: float
    raw_parameters: NDArray[np.float64]
    fit_errors: NDArray[np.float64]
    parameters: NDArray[np.float64]
    homographies: NDArray[np.float64]
    raw_homographies: list[NDArray[np.float64] | None]


def find_sources(statuses: ArrayLike, snapped: ArrayLike) -> NDArray[np.str_]:
    """Label each frame by how far its homography can be trusted."""

    statuses = np.asarray(statuses)
    snapped = np.asarray(snapped, dtype=bool)

    return np.where(
        statuses == "LOST",
        "filled",
        np.where(snapped & (statuses == "GOOD"), "snapped", "predicted"),
    )


def fit_focal_lengths(
    homographies: NDArray[np.float64],
    sources: NDArray[np.str_],
    frame_size: tuple[int, int],
    principal_point: tuple[float, float],
    picture_width: float,
) -> tuple[NDArray[np.float64], list[float], bool]:
    """
    Return (focals, window_fovs, lens_changes): a focal length for every
    frame, the field of view fitted to each window, and whether the
    focal length had to change over the clip.
    """

    trusted = np.flatnonzero(sources == "snapped")

    shared_focal, _ = fit_shared_focal(
        homographies[trusted],
        frame_size,
        principal_point,
        focal_from_fov(INITIAL_FOV, picture_width),
    )

    window_centers = []
    window_focals = []

    for start in range(0, len(homographies), FOCAL_WINDOW):
        frames = trusted[
            (trusted >= start) & (trusted < start + FOCAL_WINDOW)
        ]

        if len(frames) < 5:
            continue

        window_focal, _ = fit_shared_focal(
            homographies[frames],
            frame_size,
            principal_point,
            shared_focal,
        )

        window_centers.append(frames.mean())
        window_focals.append(window_focal)

    window_fovs = [
        fov_from_focal(focal, picture_width) for focal in window_focals
    ]

    if len(window_focals) < 2:
        return np.full(len(homographies), shared_focal), window_fovs, False

    frame_numbers = np.arange(len(homographies))
    varying_focals = np.interp(frame_numbers, window_centers, window_focals)

    # Compare how well each choice reproduces the trusted frames.
    def fit_error(focals):
        errors = []

        for frame in trusted:
            pose = fit_pose(
                homographies[frame],
                camera_matrix(focals[frame], principal_point),
                frame_size,
            )

            if pose is not None:
                errors.append(pose[2])

        return np.sqrt(np.mean(np.square(errors)))

    shared_focals = np.full(len(homographies), shared_focal)
    lens_changes = bool(
        fit_error(varying_focals)
        < (1 - ZOOM_IMPROVEMENT) * fit_error(shared_focals)
    )

    if not lens_changes:
        return shared_focals, window_fovs, False

    return (
        smooth_series(varying_focals, np.ones(len(homographies)), SMOOTHING),
        window_fovs,
        True,
    )


def solve_poses(
    homographies: NDArray[np.float64],
    focals: NDArray[np.float64],
    sources: NDArray[np.str_],
    frame_size: tuple[int, int],
    principal_point: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Return the raw camera parameters of every frame, (x, y, height, pan,
    tilt, roll) per row, the fit error of each, and the up sign.
    Frames whose pose cannot be solved are marked "filled" in `sources`.
    """

    count = len(homographies)
    parameters = np.zeros((count, 6))
    errors = np.full(count, np.nan)
    poses = [None] * count

    for frame in range(count):
        pose = fit_pose(
            homographies[frame],
            camera_matrix(focals[frame], principal_point),
            frame_size,
        )

        if pose is None:
            sources[frame] = "filled"
            continue

        rvec, tvec, errors[frame] = pose
        poses[frame] = (cv2.Rodrigues(rvec)[0], tvec)

    # The camera stays on one side of the field for the whole clip.
    up_signs = [
        find_up_sign(*poses[frame])
        for frame in np.flatnonzero(sources == "snapped")
        if poses[frame] is not None
    ]

    up_sign = 1.0 if np.mean(up_signs) > 0 else -1.0

    for frame, pose in enumerate(poses):
        if pose is not None:
            parameters[frame] = pose_to_parameters(*pose, up_sign)

    return parameters, errors, up_sign


def smooth_parameters(
    parameters: NDArray[np.float64],
    sources: NDArray[np.str_],
) -> NDArray[np.float64]:
    weights = np.array([SOURCE_WEIGHTS[source] for source in sources])
    used = weights > 0

    smoothed = np.zeros_like(parameters)

    for column in range(parameters.shape[1]):
        values = parameters[:, column].copy()

        # Pan wraps around at +-180 degrees. Unwrap it so smoothing does
        # not average 179 and -179 into 0.
        if column == 3:
            values[used] = np.degrees(np.unwrap(np.radians(values[used])))

        smoothed[:, column] = smooth_series(
            np.where(used, values, 0.0),
            weights,
            SMOOTHING,
        )

    smoothed[:, 3] = (smoothed[:, 3] + 180.0) % 360.0 - 180.0

    # Before the first and after the last trusted frame there is nothing to
    # fill in between, and the smoother carries the camera's motion on in
    # a straight line. That suits a short gap, but over a long one it
    # drifts off the field, or under it if the camera was coming down
    # (Clip 13). Coast a few frames, then hold.
    trusted = np.flatnonzero(used)

    if trusted.size:
        hold_edge(smoothed, trusted[-1], 1)
        hold_edge(smoothed, trusted[0], -1)

    return smoothed


def hold_edge(
    parameters: NDArray[np.float64],
    trusted: int,
    step: int,
) -> None:
    """
    Past the trusted frame, going `step` (1 forward, -1 back), keep the
    smoothed camera for at most COAST_FRAMES frames while it stays above
    the field looking down, then hold the last of those.
    """

    held = trusted
    frame = trusted + step

    while 0 <= frame < len(parameters) and abs(frame - trusted) <= COAST_FRAMES:
        height, tilt = parameters[frame, 2], parameters[frame, 4]

        if height <= 0 or tilt <= 0:
            break

        held = frame
        frame += step

    if step > 0:
        parameters[held + 1 :] = parameters[held]
    else:
        parameters[:held] = parameters[held]


def solve_camera(
    homographies: ArrayLike,
    statuses: ArrayLike,
    snapped: ArrayLike,
    frame_size: tuple[int, int],
    active_area: tuple[int, int],
) -> CameraSolution:
    """
    Fit a pinhole camera to the tracked grid of every frame: one focal
    length for the clip, unless the fit shows it changing, plus the
    camera's position, pan, tilt, and roll on every frame, smoothed over
    time.
    """

    homographies = np.asarray(homographies, dtype=np.float64)
    height = frame_size[1]
    left, right = active_area

    # The principal point is the middle of the game picture, in OpenCV's
    # pixel-center coordinates.
    principal_point = ((left + right) / 2 - 0.5, height / 2 - 0.5)
    picture_width = right - left

    sources = find_sources(statuses, snapped)

    if not (sources == "snapped").any():
        raise RuntimeError(
            "No frame was snapped to the painted lines, so there is "
            "nothing reliable to solve the camera from."
        )

    focals, window_fovs, lens_changes = fit_focal_lengths(
        homographies,
        sources,
        frame_size,
        principal_point,
        picture_width,
    )

    raw_parameters, fit_errors, up_sign = solve_poses(
        homographies,
        focals,
        sources,
        frame_size,
        principal_point,
    )

    parameters = smooth_parameters(raw_parameters, sources)

    poses = [parameters_to_pose(row, up_sign) for row in parameters]

    camera_homographies = np.array(
        [
            pose_homography(
                camera_matrix(focal, principal_point),
                rotation,
                translation,
            )
            for focal, (rotation, translation) in zip(focals, poses)
        ]
    )

    # The unsmoothed camera, to see what smoothing costs.
    raw_homographies = [
        None if source == "filled" else pose_homography(
            camera_matrix(focal, principal_point),
            *parameters_to_pose(row, up_sign),
        )
        for focal, row, source in zip(focals, raw_parameters, sources)
    ]

    return CameraSolution(
        sources=sources,
        focals=focals,
        fovs=np.array(
            [fov_from_focal(focal, picture_width) for focal in focals]
        ),
        window_fovs=window_fovs,
        lens_changes=lens_changes,
        principal_point=principal_point,
        up_sign=up_sign,
        raw_parameters=raw_parameters,
        fit_errors=fit_errors,
        parameters=parameters,
        homographies=camera_homographies,
        raw_homographies=raw_homographies,
    )


def describe_solution(
    solution: CameraSolution,
    tracked_errors: NDArray[np.float64],
    raw_errors: NDArray[np.float64],
    line_errors: NDArray[np.float64],
) -> list[str]:
    """Summarize the lens, the fit, and what smoothing costs."""

    trusted = solution.sources == "snapped"
    window_text = " ".join(f"{fov:.0f}" for fov in solution.window_fovs)

    if solution.lens_changes:
        lens = (
            f"Lens: changes from {solution.fovs.max():.1f} to "
            f"{solution.fovs.min():.1f} degrees wide "
            f"(per {FOCAL_WINDOW} frames: {window_text})"
        )
    else:
        lens = (
            f"Lens: fixed, {solution.fovs[0]:.1f} degrees wide, focal "
            f"length {solution.focals[0]:.0f} px "
            f"(per {FOCAL_WINDOW} frames: {window_text})"
        )

    def describe(errors):
        if np.isnan(errors).all():
            return "unmeasured"

        return (
            f"{np.nanmedian(errors):.1f} / "
            f"{np.nanpercentile(errors, 95):.1f} px"
        )

    lines = [
        lens,
        "Camera vs. tracked grid: median "
        f"{np.nanmedian(solution.fit_errors[trusted]):.1f} px on snapped "
        "frames. Roll: median "
        f"{np.median(solution.raw_parameters[trusted, 5]):+.2f} degrees",
        "Line error, median / 95th percentile: tracked grid "
        f"{describe(tracked_errors)}, camera {describe(raw_errors)}, "
        f"smoothed camera {describe(line_errors)}",
    ]

    filled = int((solution.sources == "filled").sum())

    if filled:
        lines.append(
            f"{filled} frames had no usable grid and were filled in from "
            "their neighbors"
        )

    return lines


def pose_to_state(
    parameters: NDArray[np.float64],
    focal: float,
    up_sign: float,
) -> CameraState:
    """
    Describe a solved camera as a CameraState, which aims the camera at
    the field point in the middle of the picture: pitch is the tilt,
    distance runs from the camera to that point, and yaw points from that
    point back toward the camera.
    """

    x, y, height, pan, tilt, roll = (float(value) for value in parameters)

    if height <= 0 or tilt <= 0:
        raise ValueError(
            "A CameraState needs a camera above the field looking down at it."
        )

    pan_radians, tilt_radians = np.radians([pan, tilt])

    # How far along the field the middle of the picture is.
    reach = height / np.tan(tilt_radians)

    return CameraState(
        target_x=float(x + reach * np.cos(pan_radians)),
        target_y=float(y + reach * np.sin(pan_radians)),
        distance=float(height / np.sin(tilt_radians)),
        yaw_degrees=float(
            np.degrees(np.arctan2(-np.cos(pan_radians), -np.sin(pan_radians)))
        ),
        pitch_degrees=tilt,
        focal_px=float(focal),
        roll_degrees=roll,
        up_sign=1 if up_sign > 0 else -1,
    )


def export_pose_json(
    path: Path,
    video_path: str,
    frame_size: tuple[int, int],
    fps: float,
    frame_indices: list[int],
    solution: CameraSolution,
) -> None:
    """
    Save the camera of every frame for other tools: the focal length, the
    position in yards (x and y as in field.py, plus height above the
    field), and pan, tilt, and roll in degrees.
    """

    frames = [
        {
            "frame_index": int(frame_index),
            "time": float(frame_index / fps),
            "source": str(solution.sources[index]),
            "focal_px": float(solution.focals[index]),
            "fov_deg": float(solution.fovs[index]),
            "position_yards": [
                float(value) for value in solution.parameters[index, :3]
            ],
            "pan_deg": float(solution.parameters[index, 3]),
            "tilt_deg": float(solution.parameters[index, 4]),
            "roll_deg": float(solution.parameters[index, 5]),
        }
        for index, frame_index in enumerate(frame_indices)
    ]

    data = {
        "video": video_path,
        "fps": fps,
        "image_width": frame_size[0],
        "image_height": frame_size[1],
        "up_sign": int(solution.up_sign),
        "frames": frames,
    }

    path.write_text(json.dumps(data, indent=1))
