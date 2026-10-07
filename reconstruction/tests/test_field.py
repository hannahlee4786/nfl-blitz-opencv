import numpy as np
import pytest

from nfl_reconstruction.field import NFLField


def test_standard_field_contains_regulation_points():
    field = NFLField.standard()

    assert field.length_yards == 120.0
    assert field.width_yards == pytest.approx(53.333)
    assert field.contains(np.array([[0.0, 0.0], [120.0, 53.333]])).all()
    assert not field.contains(
        np.array([[-0.01, 20.0], [60.0, 54.0]])
    ).any()


def test_standard_field_marks_goal_yard_hash_and_side_lines():
    segments = NFLField.standard().marking_segments()

    assert ((10.0, 0.0), (10.0, 53.333)) in segments
    assert ((110.0, 0.0), (110.0, 53.333)) in segments
    assert ((20.0, 0.0), (20.0, 53.333)) in segments
    assert ((0.0, 0.0), (120.0, 0.0)) in segments
    assert ((0.0, 53.333), (120.0, 53.333)) in segments
    assert ((10.0, 23.58), (110.0, 23.58)) in segments
    assert ((10.0, 29.75), (110.0, 29.75)) in segments

