# Defines the fixed football-field coordinate system
import numpy as np


# Full field including two 10-yard end zones.
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.333

# Approximate NFL hash-mark positions measured from the upper sideline.
# These can be adjusted later if NFL Blitz uses different proportions.
UPPER_HASH_Y = 23.58
LOWER_HASH_Y = 29.75


# Coordinate convention:
#   (0, 0) = upper-left corner of the entire field
#   x increases toward the right end zone
#   y increases toward the lower sideline


def build_calibration_points(yard_lines, field_half):
    """Build field points for three visible yard lines and both hash rows."""

    yard_lines = list(yard_lines)

    if len(yard_lines) != 3:
        raise ValueError("Exactly three yard lines are required.")

    if len(set(yard_lines)) != 3:
        raise ValueError("The three yard lines must be distinct.")

    if any(yard_line < 0 or yard_line > 50 for yard_line in yard_lines):
        raise ValueError("Yard-line labels must be between 0 and 50.")

    if field_half not in {"left", "right"}:
        raise ValueError("Field half must be 'left' or 'right'.")

    points = []

    for yard_line in yard_lines:
        if field_half == "left":
            field_x = 10.0 + yard_line
        else:
            field_x = 110.0 - yard_line

        points.extend(
            [
                {
                    "label": f"{yard_line}-yard line, upper hash",
                    "coordinate": (field_x, UPPER_HASH_Y),
                },
                {
                    "label": f"{yard_line}-yard line, lower hash",
                    "coordinate": (field_x, LOWER_HASH_Y),
                },
            ]
        )

    return points


CALIBRATION_POINTS = build_calibration_points([20, 30, 40], "left")


def get_calibration_field_points(calibration_points=CALIBRATION_POINTS):
    """Return the permanent field coordinates used during initialization."""

    return np.array(
        [point["coordinate"] for point in calibration_points],
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
