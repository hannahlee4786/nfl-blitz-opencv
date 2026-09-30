import unittest

import numpy as np

from src.initialize_homography import validate_calibration_quality


class CalibrationQualityTests(unittest.TestCase):
    def test_accepts_six_consistent_calibration_points(self):
        validate_calibration_quality(
            np.ones((6, 1), dtype=np.uint8),
            np.array([0.4, 0.8, 1.2, 0.6, 0.5, 0.9]),
        )

    def test_rejects_a_calibration_with_only_four_inliers(self):
        with self.assertRaisesRegex(ValueError, "inliers"):
            validate_calibration_quality(
                np.array([[1], [1], [1], [1], [0], [0]], dtype=np.uint8),
                np.array([0.2, 0.4, 0.7, 0.8, 8.0, 9.0]),
            )

    def test_rejects_large_average_reprojection_error(self):
        with self.assertRaisesRegex(ValueError, "reprojection"):
            validate_calibration_quality(
                np.ones((6, 1), dtype=np.uint8),
                np.array([4.0, 4.2, 4.5, 4.3, 4.1, 4.4]),
            )


if __name__ == "__main__":
    unittest.main()
