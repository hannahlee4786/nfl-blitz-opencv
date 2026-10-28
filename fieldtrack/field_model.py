# Defines the fixed football-field coordinate system
import numpy as np


# Full field including two 10-yard end zones.
FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.333

# The playing field's visible boundary is the inner edge of the white sideline
# band. The first version of this model (from the player-tracking branch)
# put its edges 2-3 yd outside that: hand-labeled band edges on two clips and
# five frames land at y = 2.1-3.2 on the near side and 50.9-52.1 on the far
# side of that model, steady across the clips. Everything below is expressed
# with the boundary at y = 0 and y = FIELD_WIDTH, by shifting those old
# values by BOUNDARY_NEAR and stretching so the boundaries are FIELD_WIDTH
# apart. This assumes a yard is the same length across the field as along it.
_OLD_BOUNDARY_NEAR = 2.8
_OLD_BOUNDARY_FAR = 51.5
_SCALE = FIELD_WIDTH / (_OLD_BOUNDARY_FAR - _OLD_BOUNDARY_NEAR)


def from_old_model_y(y):
    """Convert a y in the old (player-tracking) frame to the boundary frame."""
    return (np.asarray(y, dtype=np.float64) - _OLD_BOUNDARY_NEAR) * _SCALE


# Hash-mark (dotted column) positions. Ridge response along the calibrated
# field on three clips peaks at 23.5 and 30.5 in the old frame; converted,
# the two rows are symmetric about the middle of the field to within 0.2 yd.
UPPER_HASH_Y = float(from_old_model_y(23.5))
LOWER_HASH_Y = float(from_old_model_y(30.5))

# Centers of the painted yard numbers (15.8 and 37.53 in the old frame).
UPPER_NUMBER_Y = float(from_old_model_y(15.8))
LOWER_NUMBER_Y = float(from_old_model_y(37.53))


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


VALID_YARD_NUMBERS = (10, 20, 30, 40, 50)


def parse_yard_line(token):
    """
    "30" -> (30, False)   unprefixed: side follows the far_half flag
    "F30" -> (30, True)   always the far half
    "N30" -> (30, False)  always the near half
    Returns (yard_number, side) where side is True / False / None.
    """

    text = str(token).strip().upper()
    side = None

    if text.startswith("F"):
        side = True
        text = text[1:]
    elif text.startswith("N"):
        side = False
        text = text[1:]

    number = int(text)

    if number not in VALID_YARD_NUMBERS:
        raise ValueError(
            f"Yard number {number} is not painted. "
            f"Use one of {VALID_YARD_NUMBERS}."
        )

    return number, side


def create_calibration_points(yard_numbers=(20, 30, 40), far_half=False,
                              swap_sides=False):
    """
    Return the landmarks to click: the center of each painted yard number,
    between its two digits, on both sides of the field.

    Each entry may be a number or a string such as "F40" (see
    parse_yard_line). swap_sides=True is for a camera facing the +x end
    zone, where the screen-left number is the lower-sideline one.
    """

    left_y, right_y = UPPER_NUMBER_Y, LOWER_NUMBER_Y

    if swap_sides:
        left_y, right_y = right_y, left_y

    points = []

    for token in yard_numbers:
        yard_number, side = parse_yard_line(token)
        x = yard_line_x(yard_number, far_half if side is None else side)

        label = f"{token}-yard line" if side is not None else (
            f"{yard_number}-yard line"
        )

        points.append(
            {
                "label": f"{label}, left number",
                "coordinate": (x, left_y),
            }
        )

        points.append(
            {
                "label": f"{label}, right number",
                "coordinate": (x, right_y),
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