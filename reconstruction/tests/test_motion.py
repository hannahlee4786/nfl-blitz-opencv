import numpy as np

from nfl_reconstruction.motion import motion_is_sane, relative_to_anchor


def test_relative_motion_is_rebased_at_anchor():
    first = np.array(((1, 0, 5), (0, 1, 2), (0, 0, 1)), dtype=float)
    second = np.array(((1, 0, 3), (0, 1, -1), (0, 0, 1)), dtype=float)

    result = relative_to_anchor([first, second], anchor_index=1)

    assert np.allclose(result[1], np.eye(3))
    assert np.allclose(result[2], second)


def test_motion_sanity_rejects_extreme_adjacent_zoom():
    ordinary = np.array(((1.02, 0, 3), (0, 1.02, -2), (0, 0, 1)), dtype=float)
    extreme = np.array(((5, 0, 0), (0, 5, 0), (0, 0, 1)), dtype=float)

    assert motion_is_sane(ordinary, (864, 614))
    assert not motion_is_sane(extreme, (864, 614))
