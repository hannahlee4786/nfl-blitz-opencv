# Defines the fixed football-field coordinate system
import numpy as np


# Full field including two 10-yard end zones.
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.333

# Approximate NFL hash-mark positions measured from the upper sideline.
# These can be adjusted later if NFL Blitz uses different proportions.
UPPER_HASH_Y = 23.58
LOWER_HASH_Y = 29.75

# Centers of the painted yard numbers, measured from the upper sideline.
# NFL Blitz paints them 15.8 yards in from each sideline (measured on
# gameplay3), not about 8 yards as on a real NFL field.
UPPER_NUMBER_Y = 15.8
LOWER_NUMBER_Y = FIELD_WIDTH - UPPER_NUMBER_Y


# Coordinate convention:
#   (0, 0) = upper-left corner of the entire field
#   x increases toward the right end zone
#   y increases toward the lower sideline
#
# The x = 0 end zone is the COWBOYS end zone in the current clips.


def yard_line_x(yard_number, far_half=False):
    """
    Return the field x of a painted yard number.

    The near half is the one next to the x = 0 end zone.
    """

    if far_half:
        return 110.0 - yard_number

    return 10.0 + yard_number


def create_calibration_points(yard_numbers=(20, 30, 40), far_half=False):
    """
    Return the landmarks to click: the center of each painted yard number,
    between its two digits, on both sides of the field.

    The numbers sit about 22 yards apart, so the homography is pinned down
    across the field better than with the hash rows, which are only about
    6 yards apart.

    "Left number" assumes the camera faces the x = 0 end zone, as it does
    at the start of every current clip. Facing the other way swaps left and
    right.
    """

    points = []

    for yard_number in yard_numbers:
        x = yard_line_x(yard_number, far_half)

        points.append(
            {
                "label": f"{yard_number}-yard line, left number",
                "coordinate": (x, UPPER_NUMBER_Y),
            }
        )

        points.append(
            {
                "label": f"{yard_number}-yard line, right number",
                "coordinate": (x, LOWER_NUMBER_Y),
            }
        )

    return points


# The 20, 30 and 40-yard lines on the near half.
CALIBRATION_POINTS = create_calibration_points()


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