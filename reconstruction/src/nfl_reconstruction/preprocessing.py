"""Black-bar detection and the grass mask used to score and snap the grid.

Ported from src/preprocessing.py on the camera-tracking branch. Frames here
are RGB, as decoded by the prepare stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

# Columns that never get brighter than this are black bars, not game.
BAR_BRIGHTNESS = 40

# Grass, and the white paint on it, in OpenCV's HSV ranges.
LOWER_GREEN = np.array([25, 30, 30])
UPPER_GREEN = np.array([100, 255, 255])
LOWER_WHITE = np.array([0, 0, 150])
UPPER_WHITE = np.array([180, 70, 255])


def find_active_area(
    frames: Sequence[NDArray[np.uint8]],
    sample_count: int = 10,
) -> tuple[int, int]:
    """
    Return the first column of the game picture and the column just past
    its end.

    Widescreen captures pad the 4:3 picture with black bars on the left
    and right. Columns that stay dark in every sampled frame are bars.
    """

    if not frames:
        raise ValueError("Black-bar detection needs at least one frame.")

    brightest = None

    for index in np.linspace(0, len(frames) - 1, sample_count):
        gray = cv2.cvtColor(frames[int(index)], cv2.COLOR_RGB2GRAY)
        column_max = gray.max(axis=0)

        if brightest is None:
            brightest = column_max
        else:
            brightest = np.maximum(brightest, column_max)

    columns = np.flatnonzero(brightest > BAR_BRIGHTNESS)

    if len(columns) == 0:
        return 0, len(brightest)

    return int(columns[0]), int(columns[-1]) + 1


def find_field_mask(
    frame: NDArray[np.uint8],
    excluded: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """
    Return 255 on grass and painted lines and 0 elsewhere, including the
    excluded HUD boxes and black bars.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Painted lines are white, not green. Include them so their edges
    # sit inside the mask instead of on its border.
    white_mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    field_mask = cv2.bitwise_or(green_mask, white_mask)
    field_mask[excluded] = 0

    return field_mask
