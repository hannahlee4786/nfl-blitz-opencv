import numpy as np
import pytest

from nfl_reconstruction.grid_tracking import find_frame_lines
from nfl_reconstruction.line_score import measure_line_error
from nfl_reconstruction.line_snapping import snap_to_lines
from tests.fixtures import (
    FRAME_SIZE,
    across_line_errors,
    blitz_homography,
    field_shift,
    render_field,
)


@pytest.mark.parametrize("shift", [0.3, -0.6, 1.0])
def test_snapping_puts_a_shifted_grid_back_on_the_painted_lines(shift):
    truth = blitz_homography()
    frame = render_field(truth)
    lines = find_frame_lines(frame, np.zeros(frame.shape[:2], dtype=bool))
    shifted = truth @ field_shift(shift, 0.0)

    snapped = snap_to_lines(shifted, lines.painted, lines.distance_map, lines.field_mask)

    assert snapped is not None
    assert measure_line_error(lines.distance_map, lines.field_mask, snapped) == 0.0
    assert np.median(across_line_errors(truth, shifted)) == pytest.approx(abs(shift))
    assert np.median(across_line_errors(truth, snapped)) < 0.06
    assert np.max(across_line_errors(truth, snapped)) < 0.15


def test_snapping_refuses_a_grid_half_a_line_spacing_off():
    truth = blitz_homography()
    frame = render_field(truth)
    lines = find_frame_lines(frame, np.zeros(frame.shape[:2], dtype=bool))

    snapped = snap_to_lines(
        truth @ field_shift(2.5, 0.0), lines.painted, lines.distance_map, lines.field_mask
    )

    assert snapped is None


def test_snapping_needs_painted_lines():
    truth = blitz_homography()
    lines = find_frame_lines(
        render_field(truth), np.ones((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=bool)
    )

    assert snap_to_lines(truth, lines.painted, lines.distance_map, lines.field_mask) is None
