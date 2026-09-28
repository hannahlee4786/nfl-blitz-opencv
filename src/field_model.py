# Defines the fixed football-field coordinate system
import numpy as np


# Full field including two 10-yard end zones.
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.333

# Approximate NFL hash-mark positions measured from the upper sideline.
# These can be adjusted later if NFL Blitz uses different proportions.
UPPER_HASH_Y = 23.58
LOWER_HASH_Y = 29.75


# IMPORTANT:
# Edit the x-coordinates below to match the specific yard lines visible
# in your chosen initialization frame.
#
# Coordinate convention:
#   (0, 0) = upper-left corner of the entire field
#   x increases toward the right end zone
#   y increases toward the lower sideline
#
# These six points represent three yard lines intersecting two hash rows.
CALIBRATION_POINTS = [
    {
        "label": "20-yard line, left hash",
        "coordinate": (30.0, UPPER_HASH_Y),
    },
    {
        "label": "20-yard line, right hash",
        "coordinate": (30.0, LOWER_HASH_Y),
    },
    {
        "label": "30-yard line, left hash",
        "coordinate": (40.0, UPPER_HASH_Y),
    },
    {
        "label": "30-yard line, right hash",
        "coordinate": (40.0, LOWER_HASH_Y),
    },
    {
        "label": "40-yard line, left hash",
        "coordinate": (50.0, UPPER_HASH_Y),
    },
    {
        "label": "40-yard line, right hash",
        "coordinate": (50.0, LOWER_HASH_Y),
    },
]


def get_calibration_field_points():
    """Return the permanent field coordinates used during initialization."""

    return np.array(
        [point["coordinate"] for point in CALIBRATION_POINTS],
        dtype=np.float32,
    )


def create_field_grid(spacing=10.0):
    grid_lines = []

    # x = 10 is the left goal line.
    # x = 110 is the right goal line.
    x = 10.0

    while x <= 110.0:
        grid_lines.append(
            (
                (x, 0.0),
                (x, FIELD_WIDTH),
            )
        )

        x += spacing

    # Field sidelines, including both end zones.
    grid_lines.append(
        (
            (0.0, 0.0),
            (FIELD_LENGTH, 0.0),
        )
    )

    grid_lines.append(
        (
            (0.0, FIELD_WIDTH),
            (FIELD_LENGTH, FIELD_WIDTH),
        )
    )

    # Show only the locally calibrated part of the hash rows.
    grid_lines.append(
        (
            (25.0, UPPER_HASH_Y),
            (55.0, UPPER_HASH_Y),
        )
    )

    grid_lines.append(
        (
            (25.0, LOWER_HASH_Y),
            (55.0, LOWER_HASH_Y),
        )
    )

    return grid_lines