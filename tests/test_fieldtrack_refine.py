import unittest

import cv2
import numpy as np

from fieldtrack.refine import LINE_XS, refine


def synthetic_field(H, size=(1016, 1356)):
    """A green picture with white yard lines drawn through the true fit H."""
    h, w = size
    img = np.full((h, w, 3), (90, 110, 40), np.uint8)
    for x in LINE_XS:
        pts = np.array([[x, y] for y in np.linspace(0, 53.33, 80)], np.float32)
        s = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)[:, 0, :]
        cv2.polylines(img, [s.astype(np.int32)], False, (235, 235, 235), 4)
    return img


class RefineTests(unittest.TestCase):
    def setUp(self):
        # Field -> screen for a camera looking down the field: field x runs
        # up the picture, field y runs across it, as in the game footage.
        src = np.array([[10, 0], [110, 0], [110, 53.33], [10, 53.33]], np.float32)
        dst = np.array([[60, 900], [500, 120], [860, 120], [1300, 900]], np.float32)
        self.truth = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        # Shift the view so about a dozen yard lines are on screen.
        self.truth = self.truth @ np.array([[1, 0, 25.0], [0, 1, 0], [0, 0, 1]])

    def screen_error(self, a, b, shape=(1016, 1356)):
        pts = np.array([[x, y] for x in (40, 50, 60) for y in (10, 25, 40)], np.float32)
        pa = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), a)[:, 0, :]
        pb = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), b)[:, 0, :]
        return float(np.median(np.linalg.norm(pa - pb, axis=1)))

    def test_pulls_a_drifted_fit_back_onto_the_lines(self):
        img = synthetic_field(self.truth)
        # Yard lines only pin the downfield direction, so that is the drift to correct.
        drift = np.array([[1.02, 0, 1.6], [0, 1.0, 0.0], [0, 0, 1.0]])
        bad = self.truth @ drift
        before = self.screen_error(bad, self.truth)
        fixed, residual, lines = refine(bad, img)
        after = self.screen_error(fixed, self.truth)
        self.assertGreater(before, 8.0)
        self.assertLess(after, before / 2)
        self.assertGreater(lines, 60)

    def test_leaves_a_correct_fit_alone(self):
        img = synthetic_field(self.truth)
        fixed, residual, lines = refine(self.truth, img)
        self.assertLess(self.screen_error(fixed, self.truth), 2.0)

    def test_blank_frame_does_not_raise(self):
        blank = np.zeros((1016, 1356, 3), np.uint8)
        fixed, residual, lines = refine(self.truth, blank)
        self.assertEqual(fixed.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
