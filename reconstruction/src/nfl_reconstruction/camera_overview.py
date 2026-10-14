"""Render a fixed, behind-the-camera overview of solved camera motion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .camera_model import (
    camera_center,
    camera_matrix,
    fov_from_focal,
    parameters_to_pose,
)
from .field import LOWER_NUMBER_Y, UPPER_NUMBER_Y, NFLField
from .schemas import CameraFrame, CameraState

OVERVIEW_FOV = 45.0
OVERVIEW_ELEVATION = 30.0
MAX_VIEW_DISTANCE = 120.0
SURROUND = 15.0
FRAME_MARGIN = 0.06

# RGB colors. OpenCV drawing functions preserve the tuple order supplied.
BACKGROUND_COLOR = (30, 30, 30)
SURROUND_COLOR = (40, 55, 40)
FIELD_COLOR = (45, 100, 45)
END_ZONE_COLOR = (35, 75, 35)
LINE_COLOR = (225, 225, 225)
CAMERA_COLOR = (255, 210, 0)
UNCERTAIN_CAMERA_COLOR = (155, 155, 155)
VIEW_COLOR = (60, 190, 255)
PATH_COLOR = (255, 0, 255)
SHADOW_COLOR = (25, 60, 25)
WARNING_COLOR = (255, 180, 35)

OverviewProjection = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]


@dataclass(frozen=True)
class CameraOverview:
    camera_frames: tuple[CameraFrame, ...]
    frame_size: tuple[int, int]
    size: tuple[int, int]
    field: NFLField
    matrices: tuple[NDArray[np.float64], ...]
    rotations: NDArray[np.float64]
    translations: NDArray[np.float64]
    camera_centers: NDArray[np.float64]
    views: tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]
    projection: OverviewProjection
    field_image: NDArray[np.uint8]
    camera_size: float
    shadows: NDArray[np.float64]


def _state_parameters(state: CameraState) -> NDArray[np.float64]:
    pitch = np.radians(state.pitch_degrees)
    yaw = np.radians(state.yaw_degrees)
    height = state.distance * np.sin(pitch)
    reach = state.distance * np.cos(pitch)
    camera_x = state.target_x + reach * np.sin(yaw)
    camera_y = state.target_y + reach * np.cos(yaw)
    pan = np.degrees(
        np.arctan2(state.target_y - camera_y, state.target_x - camera_x)
    )
    return np.asarray(
        (
            camera_x,
            camera_y,
            height,
            pan,
            state.pitch_degrees,
            state.roll_degrees,
        ),
        dtype=float,
    )


def _look_at(
    eye: NDArray[np.float64],
    target: NDArray[np.float64],
    up: NDArray[np.float64],
) -> NDArray[np.float64]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return np.asarray((right, np.cross(forward, right), forward))


def _project(
    projection: OverviewProjection,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    matrix, rotation, eye = projection
    camera_points = (rotation @ (np.asarray(points, dtype=float) - eye).T).T
    in_front = camera_points[:, 2] > 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = (matrix @ camera_points.T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]
    return pixels, in_front


def project_overview(
    overview: CameraOverview,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Project three-dimensional field points into the overview panel."""

    return _project(overview.projection, points)


def _ground_rectangle(field: NFLField, margin: float) -> NDArray[np.float64]:
    return np.asarray(
        (
            (-margin, -margin, 0.0),
            (field.length_yards + margin, -margin, 0.0),
            (field.length_yards + margin, field.width_yards + margin, 0.0),
            (-margin, field.width_yards + margin, 0.0),
        )
    )


def _create_projection(
    centers: NDArray[np.float64],
    rotations: NDArray[np.float64],
    up: NDArray[np.float64],
    field: NFLField,
    size: tuple[int, int],
) -> OverviewProjection:
    width, height = size
    scene = np.vstack((_ground_rectangle(field, 0.0), centers))
    target = (scene.min(axis=0) + scene.max(axis=0)) / 2
    target[2] = 0.0

    # A single clip-wide direction keeps the field stationary. Placing the
    # overview eye behind that direction puts the solved camera at the bottom
    # of the panel, facing upward into the field like the gameplay footage.
    heading = np.mean(rotations[:, 2], axis=0)
    heading -= (heading @ up) * up
    heading /= np.linalg.norm(heading)

    elevation = np.radians(OVERVIEW_ELEVATION)
    focal = (width / 2) / np.tan(np.radians(OVERVIEW_FOV / 2))
    matrix = camera_matrix(focal, (width / 2, height / 2))
    low = np.asarray((width, height)) * FRAME_MARGIN
    high = np.asarray((width, height)) * (1 - FRAME_MARGIN)
    distance = 10.0

    for _ in range(250):
        eye = (
            target
            - distance * np.cos(elevation) * heading
            + distance * np.sin(elevation) * up
        )
        projection = (matrix, _look_at(eye, target, up), eye)
        pixels, in_front = _project(projection, scene)
        if in_front.all() and (pixels > low).all() and (pixels < high).all():
            return projection
        distance *= 1.05
    raise RuntimeError("Could not frame the field and camera path in the overview.")


def _follow_view_rays(
    matrix: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    frame_size: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    width, height = frame_size
    corners = np.asarray(
        ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0))
    )
    border = np.vstack(
        tuple(
            np.linspace(corners[index], corners[(index + 1) % 4], 20, endpoint=False)
            for index in range(4)
        )
    )
    center = camera_center(rotation, translation)
    inverse_matrix = np.linalg.inv(matrix)

    def follow(screen_points: NDArray[np.float64]):
        homogeneous = np.column_stack((screen_points, np.ones(len(screen_points))))
        rays = (rotation.T @ (inverse_matrix @ homogeneous.T)).T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            steps = -center[2] / rays[:, 2]
        hits = (steps > 0) & (steps < MAX_VIEW_DISTANCE)
        steps = np.where(hits, steps, MAX_VIEW_DISTANCE)
        return center + steps[:, None] * rays, hits

    corner_ends, _ = follow(corners)
    border_ends, border_hits = follow(border)
    return corner_ends, border_ends[border_hits]


def _to_pixel(point: NDArray[np.float64]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _draw_polyline(
    panel: NDArray[np.uint8],
    projection: OverviewProjection,
    points: NDArray[np.float64],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    pixels, in_front = _project(projection, points)
    if in_front.all():
        cv2.polylines(
            panel,
            [np.round(pixels).astype(np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_field(
    projection: OverviewProjection,
    field: NFLField,
    size: tuple[int, int],
) -> NDArray[np.uint8]:
    width, height = size
    panel = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)

    def fill(corners: NDArray[np.float64], color: tuple[int, int, int]) -> None:
        pixels, in_front = _project(projection, corners)
        if in_front.all():
            cv2.fillPoly(panel, [np.round(pixels).astype(np.int32)], color)

    def strip(x_start: float, x_end: float) -> NDArray[np.float64]:
        return np.asarray(
            (
                (x_start, 0.0, 0.0),
                (x_end, 0.0, 0.0),
                (x_end, field.width_yards, 0.0),
                (x_start, field.width_yards, 0.0),
            )
        )

    fill(_ground_rectangle(field, SURROUND), SURROUND_COLOR)
    fill(strip(0.0, field.length_yards), FIELD_COLOR)
    fill(strip(0.0, 10.0), END_ZONE_COLOR)
    fill(strip(field.length_yards - 10.0, field.length_yards), END_ZONE_COLOR)

    lines = [
        ((float(x), 0.0), (float(x), field.width_yards))
        for x in range(0, round(field.length_yards) + 1, 5)
    ]
    lines.extend(
        (
            ((0.0, 0.0), (field.length_yards, 0.0)),
            ((0.0, field.width_yards), (field.length_yards, field.width_yards)),
        )
    )
    for start, end in lines:
        steps = np.linspace(0.0, 1.0, 30)[:, None]
        points = np.asarray(start) + steps * (np.asarray(end) - np.asarray(start))
        width_px = 2 if start[0] % 10 == 0 or start[1] == end[1] else 1
        _draw_polyline(
            panel,
            projection,
            np.column_stack((points, np.zeros(len(points)))),
            LINE_COLOR,
            width_px,
        )

    for x in range(20, 101, 10):
        label = str(50 - abs(x - 60))
        for y in (UPPER_NUMBER_Y, LOWER_NUMBER_Y):
            pixel, in_front = _project(projection, np.asarray(((x, y, 0.0),)))
            if in_front[0]:
                cv2.putText(
                    panel,
                    label,
                    _to_pixel(pixel[0] - (10, -6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    LINE_COLOR,
                    1,
                    cv2.LINE_AA,
                )
    return panel


def _draw_camera(
    panel: NDArray[np.uint8],
    overview: CameraOverview,
    index: int,
    color: tuple[int, int, int],
) -> None:
    width, height = overview.frame_size
    corners = np.asarray(
        ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0))
    )
    triangle = np.asarray(
        (
            (0.25 * width, 0.0),
            (0.75 * width, 0.0),
            (0.5 * width, -0.3 * height),
        )
    )
    rotation = overview.rotations[index]
    center = overview.camera_centers[index]
    inverse_matrix = np.linalg.inv(overview.matrices[index])

    def rays(screen_points: NDArray[np.float64]) -> NDArray[np.float64]:
        homogeneous = np.column_stack((screen_points, np.ones(len(screen_points))))
        return (rotation.T @ (inverse_matrix @ homogeneous.T)).T

    picture = center + overview.camera_size * rays(corners)
    marker = center + overview.camera_size * rays(triangle)
    pixels, in_front = _project(
        overview.projection, np.vstack(((center,), picture, marker))
    )
    if not in_front.all():
        return
    apex, picture_pixels, marker_pixels = pixels[0], pixels[1:5], pixels[5:]
    for corner in picture_pixels:
        cv2.line(panel, _to_pixel(apex), _to_pixel(corner), color, 2, cv2.LINE_AA)
    cv2.polylines(
        panel,
        [np.round(picture_pixels).astype(np.int32)],
        True,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.fillPoly(panel, [np.round(marker_pixels).astype(np.int32)], color)


def _draw_label(
    panel: NDArray[np.uint8],
    lines: tuple[str, ...],
    color: tuple[int, int, int] = LINE_COLOR,
) -> None:
    text_width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        for line in lines
    )
    cv2.rectangle(panel, (8, 8), (32 + text_width, 20 + 28 * len(lines)), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (20, 36 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )


def build_camera_overview(
    camera_frames: list[CameraFrame],
    frame_size: tuple[int, int],
    field: NFLField,
    *,
    size: tuple[int, int] = (960, 720),
) -> CameraOverview:
    """Prepare one fixed overview projection for an entire camera sequence."""

    if not camera_frames:
        raise ValueError("Camera overview requires at least one camera frame.")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Camera overview dimensions must be positive.")
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Source frame dimensions must be positive.")

    parameters = np.stack([_state_parameters(frame.state) for frame in camera_frames])
    poses = [
        parameters_to_pose(values, camera_frames[index].state.up_sign)
        for index, values in enumerate(parameters)
    ]
    rotations = np.stack([pose[0] for pose in poses])
    translations = np.stack([pose[1] for pose in poses])
    centers = np.stack(
        [camera_center(rotation, translation) for rotation, translation in poses]
    )
    matrices = tuple(
        camera_matrix(
            frame.state.focal_px,
            ((frame_width - 1) / 2, (frame_height - 1) / 2),
        )
        for frame in camera_frames
    )
    views = tuple(
        _follow_view_rays(matrix, rotation, translation, frame_size)
        for matrix, rotation, translation in zip(
            matrices, rotations, translations, strict=True
        )
    )
    up_sign = camera_frames[0].state.up_sign
    if any(frame.state.up_sign != up_sign for frame in camera_frames):
        raise ValueError("Every camera frame in a clip must use the same up sign.")
    up = np.asarray((0.0, 0.0, float(up_sign)))
    projection = _create_projection(centers, rotations, up, field, size)
    field_image = _draw_field(projection, field, size)
    camera_size = 0.06 * np.linalg.norm(projection[2] - centers.mean(axis=0))
    shadows = centers - np.outer(centers @ up, up)
    return CameraOverview(
        camera_frames=tuple(camera_frames),
        frame_size=frame_size,
        size=size,
        field=field,
        matrices=matrices,
        rotations=rotations,
        translations=translations,
        camera_centers=centers,
        views=views,
        projection=projection,
        field_image=field_image,
        camera_size=float(camera_size),
        shadows=shadows,
    )


def render_camera_overview(
    overview: CameraOverview,
    index: int,
) -> NDArray[np.uint8]:
    """Render the camera, its footprint, and its path over the fixed field."""

    if index < 0 or index >= len(overview.camera_frames):
        raise IndexError("Camera overview frame index is out of range.")
    panel = overview.field_image.copy()
    frame = overview.camera_frames[index]
    corner_ends, outline = overview.views[index]
    clipped = None
    if len(outline) >= 3:
        _, clipped = cv2.intersectConvexConvex(
            outline[:, :2].astype(np.float32),
            _ground_rectangle(overview.field, SURROUND)[:, :2].astype(np.float32),
        )
    if clipped is not None and len(clipped) >= 3:
        clipped = clipped.reshape(-1, 2)
        pixels, in_front = _project(
            overview.projection,
            np.column_stack((clipped, np.zeros(len(clipped)))),
        )
        if in_front.all():
            polygon = np.round(pixels).astype(np.int32)
            shaded = panel.copy()
            cv2.fillPoly(shaded, [polygon], VIEW_COLOR)
            panel = cv2.addWeighted(shaded, 0.35, panel, 0.65, 0)
            cv2.polylines(panel, [polygon], True, VIEW_COLOR, 2, cv2.LINE_8)

    if index > 0:
        _draw_polyline(
            panel,
            overview.projection,
            overview.shadows[: index + 1],
            SHADOW_COLOR,
            2,
        )
        _draw_polyline(
            panel,
            overview.projection,
            overview.camera_centers[: index + 1],
            PATH_COLOR,
            2,
        )
    color = (
        CAMERA_COLOR
        if frame.status == "reliable" and frame.source != "filled"
        else UNCERTAIN_CAMERA_COLOR
    )
    _draw_polyline(
        panel,
        overview.projection,
        np.asarray((overview.camera_centers[index], overview.shadows[index])),
        color,
        1,
    )
    for end in corner_ends:
        _draw_polyline(
            panel,
            overview.projection,
            np.asarray((overview.camera_centers[index], end)),
            VIEW_COLOR,
            1,
        )
    _draw_camera(panel, overview, index, color)

    parameters = _state_parameters(frame.state)
    fov = fov_from_focal(frame.state.focal_px, overview.frame_size[0])
    status_color = LINE_COLOR if frame.status == "reliable" else WARNING_COLOR
    _draw_label(
        panel,
        (
            f"Frame {frame.frame} | lens {fov:.1f} deg | height {parameters[2]:.1f} yd",
            f"pan {parameters[3]:+.0f} | tilt {parameters[4]:.0f} | "
            f"roll {parameters[5]:+.1f} deg | {frame.status} / {frame.source or 'missing'}",
        ),
        status_color,
    )
    return panel
