"""Side-by-side diagnostics for source and fixed-field reconstruction views."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .camera_overview import build_camera_overview, render_camera_overview
from .field import NFLField
from .schemas import CameraFrame
from .visualization import render_overlay

GRID_COLOR = (230, 70, 255)
RELIABLE_COLOR = (40, 210, 90)
WARNING_COLOR = (255, 180, 35)


def _fit_image(image: NDArray[np.uint8], size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    source = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    source.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (12, 12, 12))
    offset = (
        (target_width - source.width) // 2,
        (target_height - source.height) // 2,
    )
    canvas.paste(source, offset)
    return canvas


def _status(camera: CameraFrame) -> tuple[tuple[int, int, int], str]:
    reliable = camera.status == "reliable" and camera.source == "snapped"
    color = RELIABLE_COLOR if reliable else WARNING_COLOR
    source = camera.source or "no transform"
    error = "n/a" if camera.line_error_px is None else f"{camera.line_error_px:.2f}px"
    return color, (
        f"frame {camera.frame} | {camera.status} | {source} | "
        f"confidence {camera.confidence:.2f} | line error {error}"
    )


def render_preview_frame(
    frame: NDArray[np.uint8],
    camera: CameraFrame,
    field: NFLField,
    overview_panel: NDArray[np.uint8],
    *,
    size: tuple[int, int] = (1920, 720),
) -> NDArray[np.uint8]:
    """Compose solved-grid footage and the fixed camera overview."""

    width, height = size
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Preview size must contain positive even dimensions.")
    if overview_panel.ndim != 3 or overview_panel.shape[2] != 3:
        raise ValueError("Camera overview panel must be an RGB image.")

    pane_width = width // 2
    overlay = render_overlay(
        frame,
        field,
        camera,
        [],
        [],
        line_color=GRID_COLOR,
    )
    left = np.asarray(_fit_image(overlay, (pane_width, height)), dtype=np.uint8).copy()
    right = np.asarray(
        _fit_image(overview_panel, (pane_width, height)), dtype=np.uint8
    ).copy()
    color, status_text = _status(camera)
    label_lines = ("Footage with the solved camera's grid", status_text)
    label_width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        for line in label_lines
    )
    cv2.rectangle(left, (8, 8), (32 + label_width, 76), (0, 0, 0), -1)
    cv2.putText(
        left,
        label_lines[0],
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        left,
        label_lines[1],
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1,
        cv2.LINE_AA,
    )
    return np.hstack((left, right))


def write_preview_video(
    path: str | Path,
    frames: list[NDArray[np.uint8]],
    cameras: list[CameraFrame],
    field: NFLField,
    *,
    fps: Fraction,
    size: tuple[int, int] = (1920, 720),
) -> None:
    """Encode a side-by-side diagnostic video at the source frame rate."""

    if not frames:
        raise ValueError("Preview requires at least one frame.")
    if len(frames) != len(cameras):
        raise ValueError("Frames and camera records must have the same length.")
    if fps <= 0:
        raise ValueError("Preview frame rate must be positive.")
    width, height = size
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Preview size must contain positive even dimensions.")
    frame_size = (frames[0].shape[1], frames[0].shape[0])
    if any(frame.shape[:2] != frames[0].shape[:2] for frame in frames):
        raise ValueError("Every preview source frame must have the same dimensions.")
    overview = build_camera_overview(
        cameras,
        frame_size,
        field,
        size=(width // 2, height),
    )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index, (frame, camera) in enumerate(zip(frames, cameras, strict=True)):
            rendered = render_preview_frame(
                frame,
                camera,
                field,
                render_camera_overview(overview, index),
                size=size,
            )
            video_frame = av.VideoFrame.from_ndarray(rendered, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
