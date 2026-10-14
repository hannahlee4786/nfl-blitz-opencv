import numpy as np

from nfl_reconstruction.preprocessing import find_active_area, find_field_mask
from tests.fixtures import GRASS, PAINT, STANDS


def test_active_area_skips_black_bars():
    frame = np.zeros((20, 60, 3), dtype=np.uint8)
    frame[:, 10:50] = GRASS

    assert find_active_area([frame]) == (10, 50)


def test_active_area_combines_bright_columns_across_frames():
    first = np.zeros((20, 60, 3), dtype=np.uint8)
    second = np.zeros((20, 60, 3), dtype=np.uint8)
    first[:, 10:30] = GRASS
    second[:, 25:52] = GRASS

    assert find_active_area([first, second]) == (10, 52)


def test_active_area_without_bars_is_the_whole_frame():
    bright = np.full((20, 60, 3), 200, dtype=np.uint8)
    dark = np.zeros((20, 60, 3), dtype=np.uint8)

    assert find_active_area([bright]) == (0, 60)
    assert find_active_area([dark]) == (0, 60)


def test_field_mask_keeps_grass_and_paint_only():
    frame = np.array([[GRASS, PAINT, STANDS, (200, 30, 30), GRASS]], dtype=np.uint8)
    excluded = np.zeros((1, 5), dtype=bool)
    excluded[0, 4] = True

    mask = find_field_mask(frame, excluded)

    assert mask.tolist() == [[255, 255, 0, 0, 0]]
