"""Diagnostic overlays for camera and player reconstruction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .field import NFLField
from .schemas import CameraFrame, DetectionRecord, TrackRecord


def _project(point: tuple[float, float], homography: NDArray[np.float64]):
    homogeneous = homography @ np.asarray((*point, 1.0), dtype=float)
    if abs(homogeneous[2]) < 1e-9:
        return None
    return tuple((homogeneous[:2] / homogeneous[2]).tolist())


def render_overlay(
    frame: NDArray[np.uint8],
    field: NFLField,
    camera: CameraFrame,
    detections: Iterable[DetectionRecord],
    tracks: Iterable[TrackRecord],
) -> NDArray[np.uint8]:
    """Draw projected markings and player diagnostics on a source frame."""

    canvas = Image.fromarray(np.asarray(frame, dtype=np.uint8).copy(), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    line_color = (0, 255, 0) if camera.status == "reliable" else (255, 190, 0)

    if camera.homography is not None:
        homography = np.asarray(camera.homography, dtype=float)
        for start, end in field.marking_segments():
            projected_start = _project(start, homography)
            projected_end = _project(end, homography)
            if projected_start is not None and projected_end is not None:
                draw.line((projected_start, projected_end), fill=line_color, width=2)

    team_colors = {
        "offense": (255, 70, 70),
        "defense": (70, 120, 255),
        "unknown": (220, 220, 220),
    }
    for detection in detections:
        draw.rectangle(
            (detection.x1, detection.y1, detection.x2, detection.y2),
            outline=team_colors[detection.team],
            width=2,
        )

    for track in tracks:
        color = team_colors[track.team]
        x, y = track.screen_x, track.screen_y
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text((x + 4, y - 8), str(track.track_id), fill=color)

    return np.asarray(canvas)


def render_minimap(
    field: NFLField,
    tracks: Iterable[TrackRecord],
    *,
    size: tuple[int, int] = (720, 320),
) -> NDArray[np.uint8]:
    """Render field-coordinate player positions on a regulation minimap."""

    width, height = size
    canvas = Image.new("RGB", size, (35, 115, 55))
    draw = ImageDraw.Draw(canvas)

    def screen(point: tuple[float, float]) -> tuple[int, int]:
        return (
            round(point[0] / field.length_yards * (width - 1)),
            round(point[1] / field.width_yards * (height - 1)),
        )

    for start, end in field.marking_segments():
        draw.line((screen(start), screen(end)), fill=(235, 235, 235), width=1)

    colors = {
        "offense": (230, 45, 45),
        "defense": (45, 90, 230),
        "unknown": (245, 210, 60),
    }
    for track in tracks:
        if track.field_x is None or track.field_y is None:
            continue
        x, y = screen((track.field_x, track.field_y))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[track.team])

    return np.asarray(canvas)
