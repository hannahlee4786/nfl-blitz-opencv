import cv2
import numpy as np
import pytest

from nfl_reconstruction.annotations import load_anchor
from nfl_reconstruction.calibration import (
    LOWER_NUMBER_Y,
    UPPER_NUMBER_Y,
    anchor_field_points,
    build_anchor,
    frame_offset_for_key,
    move_frame_index,
)

FRAME_SIZE = (1356, 1016)
TRUTH = np.array(
    [[-20.0, 2.0, 1300.0], [0.5, 20.0, 0.0], [0.0, 0.0, 1.0]]
)


def _clicks(field_points):
    return cv2.perspectiveTransform(
        field_points.reshape(-1, 1, 2).astype(np.float32), TRUTH
    ).reshape(-1, 2)


def _pairs(heights, left_x=300.0, right_x=1000.0):
    return np.array([[(left_x, y), (right_x, y)] for y in heights]).reshape(-1, 2)


def test_twenty_nearest_the_camera_is_on_the_offense_own_half():
    clicks = _pairs([316.0, 174.0, 86.0])

    field_points = anchor_field_points((20, 30, 40), clicks)

    assert field_points.tolist() == [
        [30.0, 37.533], [30.0, 15.8],
        [40.0, 37.533], [40.0, 15.8],
        [50.0, 37.533], [50.0, 15.8],
    ]


def test_twenty_farthest_from_the_camera_is_on_the_attacked_half():
    clicks = _pairs([200.0, 305.0, 454.0])

    field_points = anchor_field_points((20, 30, 40), clicks)

    assert field_points[:, 0].tolist() == [90.0, 90.0, 80.0, 80.0, 70.0, 70.0]
    assert (UPPER_NUMBER_Y, LOWER_NUMBER_Y) == (15.8, 37.533)


def test_left_and_right_follow_the_screen_not_the_click_order():
    clicks = _pairs([200.0, 305.0, 454.0])
    clicks[[0, 1]] = clicks[[1, 0]]

    field_points = anchor_field_points((20, 30, 40), clicks)

    assert field_points[0].tolist() == [90.0, 15.8]
    assert field_points[1].tolist() == [90.0, 37.533]


def test_build_anchor_drops_a_wrong_click(tmp_path):
    field_points = anchor_field_points((20, 30, 40), _pairs([316.0, 174.0, 86.0]))
    clicks = _clicks(field_points)
    clicks[3] += (40.0, -25.0)

    anchor, error = build_anchor(field_points, clicks, 12, FRAME_SIZE)

    assert anchor["frame"] == 12
    assert len(anchor["field_points"]) == 5
    assert [40.0, 15.8] not in anchor["field_points"]
    assert error < 0.1
    path = tmp_path / "anchor.json"
    path.write_text(__import__("json").dumps(anchor))
    load_anchor(path, FRAME_SIZE)


def test_build_anchor_rejects_collinear_clicks():
    field_points = anchor_field_points((20, 30, 40), _pairs([316.0, 174.0, 86.0]))
    clicks = np.stack([np.linspace(100, 600, 6), np.full(6, 300.0)], axis=1)

    with pytest.raises(ValueError):
        build_anchor(field_points, clicks, 0, FRAME_SIZE)


def test_move_frame_index_stays_inside_the_video():
    assert move_frame_index(60, 30, 186) == 90
    assert move_frame_index(60, -30, 186) == 30
    assert move_frame_index(5, -30, 186) == 0
    assert move_frame_index(180, 30, 186) == 185
    with pytest.raises(ValueError):
        move_frame_index(0, 1, 0)


def test_frame_browser_keys_step_a_frame_or_a_second():
    assert frame_offset_for_key(ord("a"), 30.0) == -1
    assert frame_offset_for_key(ord("d"), 30.0) == 1
    assert frame_offset_for_key(ord("["), 29.97) == -30
    assert frame_offset_for_key(ord("]"), 30.0) == 30
    assert frame_offset_for_key(ord("x"), 30.0) is None
