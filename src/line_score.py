# Measures how far the projected field grid sits from the painted lines
import cv2
import numpy as np

from src.field_model import FIELD_WIDTH


# A frame is GOOD when the projected yard lines sit within this many
# pixels of painted lines, FAIR up to the second limit, and LOST beyond
# it. On Clip_4, about 3 px is a small visible offset and 4.5 px or more
# is clearly off the lines. Shifting a correct grid by half a yard-line
# spacing (2.5 yards) scores 5 to 16 px.
GOOD_LINE_ERROR = 1.5
FAIR_LINE_ERROR = 4.0

# With fewer samples than this on the visible field, the grid has left
# the field, so the frame is LOST.
MIN_LINE_SAMPLES = 30

# How much brighter than its surroundings a pixel must be to count as
# part of a painted line.
LINE_BRIGHTNESS = 12


def create_line_samples():
    """
    Return points every half yard along every 5-yard line, goal line to
    goal line.
    """

    # Stop a yard short of each sideline. Close to the sideline, little
    # else of the field is in view, so every sample counts.
    samples = [
        (x, y)
        for x in np.arange(10.0, 110.1, 5.0)
        for y in np.arange(1.0, FIELD_WIDTH - 0.99, 0.5)
    ]

    return np.array(samples, dtype=np.float64)


LINE_SAMPLES = create_line_samples()


def project_field_points(homography, field_points, frame_shape):
    """
    Return the screen position of each field point and whether it is in
    front of the camera.
    """

    height, width = frame_shape[:2]

    # Field points in front of the camera share the sign of their
    # projective w with the point seen at the image center.
    image_center = np.array([width / 2, height / 2, 1.0])
    center_on_field = np.linalg.inv(homography) @ image_center
    center_on_field /= center_on_field[2]
    front_sign = np.sign((homography @ center_on_field)[2])

    projected = np.hstack(
        [field_points, np.ones((len(field_points), 1))]
    ) @ homography.T

    in_front = projected[:, 2] * front_sign > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        screen_points = projected[:, :2] / projected[:, 2:3]

    return screen_points, in_front


def find_painted_lines(gray, field_mask):
    """
    Return how much brighter each pixel is than the grass around it.
    Painted lines are thin, so they stand out; pixels off the field are 0.
    """

    background = cv2.medianBlur(gray, 31)
    painted = cv2.subtract(gray, background)
    painted[field_mask == 0] = 0

    return painted


def create_distance_map(painted):
    """Return the distance from every pixel to the nearest painted line."""

    return cv2.distanceTransform(
        np.where(painted > LINE_BRIGHTNESS, 0, 1).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )


def measure_line_error(distance_map, field_mask, homography):
    """
    Return the median distance, in pixels, from the projected yard lines
    to the nearest painted line. Return NaN when too few of them land on
    the visible field.
    """

    height, width = distance_map.shape[:2]

    screen_points, in_front = project_field_points(
        homography,
        LINE_SAMPLES,
        distance_map.shape,
    )

    with np.errstate(invalid="ignore"):
        columns = np.round(screen_points[:, 0])
        rows = np.round(screen_points[:, 1])

        visible = (
            in_front
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )

    columns = columns[visible].astype(int)
    rows = rows[visible].astype(int)

    # Skip samples hidden by the HUD, players, or the stands.
    on_field = field_mask[rows, columns] > 0

    if on_field.sum() < MIN_LINE_SAMPLES:
        return float("nan")

    return float(
        np.median(distance_map[rows[on_field], columns[on_field]])
    )


def get_tracking_status(line_error):
    if np.isnan(line_error) or line_error >= FAIR_LINE_ERROR:
        return "LOST"

    if line_error >= GOOD_LINE_ERROR:
        return "FAIR"

    return "GOOD"


def describe_line_errors(line_errors, statuses):
    """Summarize a clip's line errors in one line."""

    line_errors = np.asarray(line_errors, dtype=np.float64)
    measured = line_errors[~np.isnan(line_errors)]

    lost_percent = 100.0 * np.mean(np.asarray(statuses) == "LOST")

    if len(measured) == 0:
        return (
            "Line error: no frame could be measured, "
            f"LOST on {lost_percent:.0f}% of frames"
        )

    return (
        f"Line error: median {np.median(measured):.1f} px, "
        f"95th percentile {np.percentile(measured, 95):.1f} px, "
        f"LOST on {lost_percent:.0f}% of frames"
    )
