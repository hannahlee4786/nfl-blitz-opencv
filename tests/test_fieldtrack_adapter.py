import unittest

import cv2
import numpy as np

from fieldtrack.adapter import screen_to_world


class ScreenToWorldTests(unittest.TestCase):
    def test_goal_line_is_the_origin_of_x(self):
        # Field -> screen: a plain scale and shift, so the inverse is easy to check.
        H = np.array([[10.0, 0.0, 100.0], [0.0, 10.0, 50.0], [0.0, 0.0, 1.0]])
        A = screen_to_world(H)
        # The 50-yard line is field x = 60; the upper sideline is y = 0.
        screen = np.array([[[10.0 * 60 + 100.0, 10.0 * 20 + 50.0]]], np.float32)
        x, y = cv2.perspectiveTransform(screen, A.astype(np.float32))[0, 0]
        self.assertAlmostEqual(float(x), 50.0, places=3)
        self.assertAlmostEqual(float(y), 20.0, places=3)


if __name__ == "__main__":
    unittest.main()
