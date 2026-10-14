import math

import numpy as np
import pytest

from nfl_reconstruction.grid_tracking import find_frame_lines
from nfl_reconstruction.line_score import (
    FAIR_LINE_ERROR,
    GOOD_LINE_ERROR,
    describe_line_errors,
    get_tracking_status,
    measure_line_error,
)
from tests.fixtures import FRAME_SIZE, blitz_homography, field_shift, render_field


def painted_lines(homography):
    frame = render_field(homography)
    return find_frame_lines(frame, np.zeros(frame.shape[:2], dtype=bool))


def test_grid_on_the_painted_lines_is_good():
    homography = blitz_homography()
    lines = painted_lines(homography)

    error = measure_line_error(lines.distance_map, lines.field_mask, homography)

    assert error < GOOD_LINE_ERROR
    assert get_tracking_status(error) == "GOOD"


def test_grid_half_a_line_spacing_off_is_lost():
    homography = blitz_homography()
    lines = painted_lines(homography)

    error = measure_line_error(
        lines.distance_map, lines.field_mask, homography @ field_shift(2.5, 0.0)
    )

    assert error >= FAIR_LINE_ERROR
    assert get_tracking_status(error) == "LOST"


def test_grid_outside_the_picture_cannot_be_measured():
    homography = blitz_homography()
    lines = painted_lines(homography)
    away = np.array(((1.0, 0.0, 5 * FRAME_SIZE[0]), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    error = measure_line_error(lines.distance_map, lines.field_mask, away @ homography)

    assert math.isnan(error)
    assert get_tracking_status(error) == "LOST"


@pytest.mark.parametrize(
    ("line_error", "status"),
    [(0.0, "GOOD"), (1.49, "GOOD"), (1.5, "FAIR"), (3.99, "FAIR"), (4.0, "LOST")],
)
def test_tracking_status_thresholds(line_error, status):
    assert get_tracking_status(line_error) == status


def test_line_error_description_counts_lost_frames():
    text = describe_line_errors([0.0, 1.0, math.nan, 2.0], ["GOOD", "GOOD", "LOST", "FAIR"])

    assert text == "Line error: median 1.0 px, 95th percentile 1.9 px, LOST on 25% of frames"
