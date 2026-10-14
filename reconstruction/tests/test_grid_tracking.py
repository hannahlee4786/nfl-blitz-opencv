import math

import numpy as np
import pytest

from nfl_reconstruction.grid_tracking import measure_line_errors, track_grid
from nfl_reconstruction.line_score import LINE_SAMPLES, project_field_points
from tests.fixtures import (
    FRAME_SIZE,
    STANDS,
    across_line_errors,
    blitz_homography,
    field_shift,
    render_field,
)

NOTHING_EXCLUDED = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=bool)


def dolly_clip(count):
    """A camera backing up the field and drifting toward one sideline."""

    truths = [
        blitz_homography((83.0 - 0.8 * frame, 26.67 + 0.3 * frame, 15.7, 180.0, 27.0, 0.0))
        for frame in range(count)
    ]
    return truths, [render_field(truth) for truth in truths]


def screen_gaps(truth, homography):
    """Return how far apart, in pixels, the two homographies put the
    yard-line samples that the truth shows on screen."""

    shape = (FRAME_SIZE[1], FRAME_SIZE[0])
    expected, in_front = project_field_points(truth, LINE_SAMPLES, shape)
    tracked, _ = project_field_points(homography, LINE_SAMPLES, shape)
    visible = (
        in_front
        & (expected[:, 0] >= 0)
        & (expected[:, 0] < FRAME_SIZE[0])
        & (expected[:, 1] >= 0)
        & (expected[:, 1] < FRAME_SIZE[1])
    )
    return np.linalg.norm(expected[visible] - tracked[visible], axis=1)


def test_tracking_follows_the_camera_both_ways_from_the_anchor():
    truths, frames = dolly_clip(8)
    # The clicks are a quarter yard off across the yard lines.
    anchor = truths[3] @ field_shift(0.25, 0.0)

    grid = track_grid(frames, NOTHING_EXCLUDED, 3, anchor)

    assert grid.snapped.all()
    assert grid.statuses.tolist() == ["GOOD"] * 8
    for truth, homography in zip(truths, grid.homographies):
        assert np.median(across_line_errors(truth, homography)) < 0.06
        assert np.median(screen_gaps(truth, homography)) < 1.0


def test_tracking_keeps_the_last_grid_when_no_field_is_visible():
    truths, _ = dolly_clip(3)
    frames = [np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), STANDS, dtype=np.uint8)] * 3

    grid = track_grid(frames, NOTHING_EXCLUDED, 0, truths[0])

    assert not grid.snapped.any()
    assert grid.statuses.tolist() == ["LOST"] * 3
    assert all(np.allclose(homography, truths[0]) for homography in grid.homographies)


def test_tracking_needs_the_anchor_inside_the_clip():
    truths, frames = dolly_clip(2)

    with pytest.raises(ValueError, match="Anchor"):
        track_grid(frames, NOTHING_EXCLUDED, 2, truths[0])


def test_line_errors_score_each_camera_on_every_frame():
    truths, frames = dolly_clip(2)
    off_the_lines = truths[1] @ field_shift(2.5, 0.0)

    errors = measure_line_errors(
        frames, NOTHING_EXCLUDED, [truths, [None, off_the_lines]]
    )

    assert errors.shape == (2, 2)
    assert (errors[0] < 1.5).all()
    assert math.isnan(errors[1, 0])
    assert errors[1, 1] >= 4.0
