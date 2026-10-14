"""Side-by-side diagnostics for source and fixed-field reconstruction views."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .field import NFLField
from .mosaic import warp_frame_to_field
from .schemas import CameraFrame
from .visualization import render_overlay


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
    color = (40, 210, 90) if reliable else (255, 180, 35)
    source = camera.source or "no transform"
    error = "n/a" if camera.line_error_px is None else f"{camera.line_error_px:.2f}px"
    return color, (
        f"frame {camera.frame} | {camera.status} | {source} | "
        f"confidence {camera.confidence:.2f} | line error {error}"
    )


def render_preview_frame(
    frame: NDArray[np.uint8],
    camera: CameraFrame,
    dynamic_mask: NDArray[np.bool_],
    field: NFLField,
    *,
    size: tuple[int, int] = (1280, 720),
    pixels_per_yard: int = 4,
) -> NDArray[np.uint8]:
    """Compose labeled source and canonical-field panes for one frame."""

    width, height = size
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Preview size must contain positive even dimensions.")
    if dynamic_mask.shape != frame.shape[:2]:
        raise ValueError("Dynamic mask shape must match the source frame.")

    pane_width = width // 2
    header_height = min(64, max(36, height // 9))
    content_size = (pane_width, height - header_height)
    overlay = render_overlay(frame, field, camera, [], [])
    warped = warp_frame_to_field(
        frame, camera, dynamic_mask, field, pixels_per_yard=pixels_per_yard
    )
    canvas = Image.new("RGB", size, (12, 12, 12))
    canvas.paste(_fit_image(overlay, content_size), (0, header_height))
    canvas.paste(_fit_image(warped.image, content_size), (pane_width, header_height))

    draw = ImageDraw.Draw(canvas)
    color, status_text = _status(camera)
    draw.rectangle((0, 0, width - 1, header_height - 1), fill=(20, 20, 20))
    draw.text((12, 8), "SOURCE + PROJECTED FIELD", fill=(235, 235, 235))
    draw.text(
        (pane_width + 12, 8),
        "FIXED FIELD COORDINATES",
        fill=(235, 235, 235),
    )
    draw.text((12, 30), status_text, fill=color)
    draw.line((pane_width, 0, pane_width, height), fill=color, width=2)
    draw.rectangle((0, 0, width - 1, height - 1), outline=color, width=4)
    return np.asarray(canvas, dtype=np.uint8)
