import unittest

import numpy as np

from src.field_model import LOWER_HASH_Y, UPPER_HASH_Y, build_calibration_points


class BuildCalibrationPointsTests(unittest.TestCase):
    def test_builds_left_half_coordinates_from_visible_yard_labels(self):
        points = build_calibration_points([30, 40, 50], "left")

        self.assertEqual(
            [point["label"] for point in points],
            [
                "30-yard line, upper hash",
                "30-yard line, lower hash",
                "40-yard line, upper hash",
                "40-yard line, lower hash",
                "50-yard line, upper hash",
                "50-yard line, lower hash",
            ],
        )
        np.testing.assert_allclose(
            [point["coordinate"] for point in points],
            [
                (40.0, UPPER_HASH_Y),
                (40.0, LOWER_HASH_Y),
                (50.0, UPPER_HASH_Y),
                (50.0, LOWER_HASH_Y),
                (60.0, UPPER_HASH_Y),
                (60.0, LOWER_HASH_Y),
            ],
        )

    def test_builds_right_half_coordinates_from_visible_yard_labels(self):
        points = build_calibration_points([30, 40, 50], "right")

        np.testing.assert_allclose(
            [point["coordinate"] for point in points],
            [
                (80.0, UPPER_HASH_Y),
                (80.0, LOWER_HASH_Y),
                (70.0, UPPER_HASH_Y),
                (70.0, LOWER_HASH_Y),
                (60.0, UPPER_HASH_Y),
                (60.0, LOWER_HASH_Y),
            ],
        )

    def test_rejects_invalid_yard_line_sets(self):
        invalid_cases = (
            ([20, 30], "left"),
            ([20, 30, 60], "left"),
            ([30, 30, 40], "left"),
            ([20, 30, 40], "middle"),
        )

        for yard_lines, field_half in invalid_cases:
            with self.subTest(yard_lines=yard_lines, field_half=field_half):
                with self.assertRaises(ValueError):
                    build_calibration_points(yard_lines, field_half)


if __name__ == "__main__":
    unittest.main()
