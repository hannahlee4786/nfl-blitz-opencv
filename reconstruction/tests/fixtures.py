import numpy as np

from nfl_reconstruction.camera_model import (
    camera_matrix,
    parameters_to_pose,
    pose_homography,
)
from nfl_reconstruction.field import LOWER_NUMBER_Y, UPPER_NUMBER_Y
from nfl_reconstruction.line_score import LINE_SAMPLES, project_field_points
from nfl_reconstruction.schemas import CameraState


def synthetic_state(**changes):
    values = {
        "target_x": 55.0,
        "target_y": 26.6665,
        "distance": 55.0,
        "yaw_degrees": 0.0,
        "pitch_degrees": 48.0,
        "focal_px": 720.0,
        "roll_degrees": 0.0,
    }
    values.update(changes)
    return CameraState(**values)


# A view like the clips: behind the play at the 73, looking toward the
# x = 0 end zone. (x, y, height, pan, tilt, roll) in yards and degrees.
BLITZ_PARAMETERS = (83.0, 26.67, 15.7, 180.0, 27.0, 0.0)
FRAME_SIZE = (678, 508)
FOCAL = 800.0

GRASS = (40, 120, 50)
DARK_GRASS = (20, 70, 28)
PAINT = (235, 235, 235)
STANDS = (60, 60, 70)

# Dark specks in the grass, about one per square yard, give the point
# tracker something to follow. Darker than the grass, they never count as
# paint.
SPECK_RADIUS = 0.2


def blitz_homography(
    parameters=BLITZ_PARAMETERS,
    up_sign=1.0,
    focal=FOCAL,
    frame_size=FRAME_SIZE,
):
    width, height = frame_size
    matrix = camera_matrix(focal, ((width - 1) / 2, (height - 1) / 2))
    rotation, translation = parameters_to_pose(
        np.asarray(parameters, dtype=float), up_sign
    )
    return pose_homography(matrix, rotation, translation)


def field_shift(dx, dy):
    """Return a homography that moves field points by (dx, dy) yards."""

    return np.array(((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0)))


def across_line_errors(truth, homography, frame_size=FRAME_SIZE):
    """Return how far, in yards along the field, each yard-line sample
    the homography puts on screen lands from its own line."""

    width, height = frame_size
    screen, in_front = project_field_points(homography, LINE_SAMPLES, (height, width))
    visible = (
        in_front
        & (screen[:, 0] >= 0)
        & (screen[:, 0] < width)
        & (screen[:, 1] >= 0)
        & (screen[:, 1] < height)
    )
    back = np.column_stack([screen[visible], np.ones(visible.sum())])
    back = back @ np.linalg.inv(truth).T
    return np.abs(back[:, 0] / back[:, 2] - LINE_SAMPLES[visible, 0])


def _repeatable_fraction(first, second):
    """Return a pseudo-random number in [0, 1) for each pair of values."""

    value = np.sin(first * 12.9898 + second * 78.233) * 43758.5453
    return value - np.floor(value)


def render_field(homography, frame_size=FRAME_SIZE, line_width=0.3, numbers=False):
    """Draw speckled grass with painted 5-yard lines and sidelines, seen
    through the homography, as an RGB frame. With `numbers`, also paint a
    stand-in for each yard number: a bar beside the line, longer the
    farther it is from the nearer goal line, so no two look alike."""

    width, height = frame_size
    rows, columns = np.indices((height, width))
    pixels = np.stack(
        [columns.ravel(), rows.ravel(), np.ones(width * height)]
    )
    field = np.linalg.inv(homography) @ pixels

    # Pixels above the horizon map to points behind the camera, whose w
    # has the opposite sign of the ground at the bottom of the picture.
    bottom = np.linalg.inv(homography) @ np.array([width / 2, height - 1, 1.0])
    in_front = np.sign(field[2]) == np.sign(bottom[2])

    with np.errstate(divide="ignore", invalid="ignore"):
        x = field[0] / field[2]
        y = field[1] / field[2]

    half = line_width / 2
    on_field = in_front & (x >= 0) & (x <= 120) & (y >= 0) & (y <= 53.333)
    nearest_line = 10 + 5 * np.round((x - 10) / 5)
    on_yard_line = (
        (np.abs(x - nearest_line) < half) & (x > 10 - half) & (x < 110 + half)
    )
    on_sideline = (y < half) | (y > 53.333 - half)

    with np.errstate(invalid="ignore"):
        cell_x, cell_y = np.floor(x), np.floor(y)
        speck_x = cell_x + 0.2 + 0.6 * _repeatable_fraction(cell_x, cell_y)
        speck_y = cell_y + 0.2 + 0.6 * _repeatable_fraction(cell_y, cell_x)
        on_speck = (x - speck_x) ** 2 + (y - speck_y) ** 2 < SPECK_RADIUS**2

    on_number = np.zeros(width * height, dtype=bool)

    if numbers:
        with np.errstate(invalid="ignore"):
            ten = 10 + 10 * np.round((x - 10) / 10)
            blocks = np.minimum(ten - 10, 110 - ten) / 10
            beside = np.abs(x - (ten - 1.5)) < 0.5

            for row in (UPPER_NUMBER_Y, LOWER_NUMBER_Y):
                on_number |= beside & (np.abs(y - row) < 0.5 + 0.4 * blocks)

    frame = np.empty((width * height, 3), dtype=np.uint8)
    frame[:] = STANDS
    frame[on_field] = GRASS
    frame[on_field & on_speck] = DARK_GRASS
    frame[on_field & (on_yard_line | on_sideline | on_number)] = PAINT
    return frame.reshape(height, width, 3)
