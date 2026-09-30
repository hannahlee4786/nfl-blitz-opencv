import unittest

import cv2
import numpy as np

from src.tracking import (
    build_feature_mask,
    detect_field_features,
    estimate_descriptor_camera_motion,
    estimate_camera_motion,
    filter_features_by_mask,
    replenish_features,
)


class FeatureBasedMotionTests(unittest.TestCase):
    def setUp(self):
        self.previous = np.zeros((160, 220), dtype=np.uint8)
        for x, y in ((30, 30), (70, 40), (110, 80), (160, 60), (190, 120)):
            cv2.rectangle(self.previous, (x - 5, y - 5), (x + 5, y + 5), 255, -1)
        self.mask = np.full_like(self.previous, 255)
        self.current = cv2.warpAffine(
            self.previous,
            np.float32([[1, 0, 7], [0, 1, -4]]),
            (self.previous.shape[1], self.previous.shape[0]),
        )

    def test_detects_multiple_features_inside_the_field_mask(self):
        features = detect_field_features(self.previous, self.mask, max_features=30)

        self.assertGreaterEqual(len(features), 5)

    def test_estimates_camera_translation_from_tracked_field_features(self):
        features = detect_field_features(self.previous, self.mask, max_features=30)

        motion, current_features, inlier_mask, errors = estimate_camera_motion(
            self.previous,
            self.current,
            features,
        )

        self.assertIsNotNone(motion)
        self.assertGreaterEqual(int(inlier_mask.sum()), 4)
        self.assertGreaterEqual(len(current_features), 4)
        self.assertLess(float(errors[inlier_mask].mean()), 0.5)
        self.assertAlmostEqual(motion[0, 2], 7.0, delta=0.5)
        self.assertAlmostEqual(motion[1, 2], -4.0, delta=0.5)

    def test_descriptor_matching_estimates_a_larger_camera_translation(self):
        previous = np.zeros((220, 280), dtype=np.uint8)
        cv2.putText(previous, "A", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
        cv2.putText(previous, "B", (130, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 190, 2)
        cv2.circle(previous, (220, 170), 18, 220, 2)
        cv2.line(previous, (35, 185), (90, 150), 150, 3)
        current = cv2.warpAffine(
            previous,
            np.float32([[1, 0, 28], [0, 1, -16]]),
            (previous.shape[1], previous.shape[0]),
        )
        mask = np.full_like(previous, 255)

        motion, match_count, inlier_count, alignment_error = (
            estimate_descriptor_camera_motion(
            previous,
            current,
            mask,
            mask,
            )
        )

        self.assertIsNotNone(motion)
        self.assertGreaterEqual(match_count, 4)
        self.assertGreaterEqual(inlier_count, 4)
        self.assertLess(alignment_error, 1.0)
        self.assertAlmostEqual(motion[0, 2], 28.0, delta=1.0)
        self.assertAlmostEqual(motion[1, 2], -16.0, delta=1.0)

    def test_replenishes_features_without_adding_nearby_duplicates(self):
        features = replenish_features(
            np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float32),
            np.array(
                [[11.0, 11.0], [70.0, 70.0], [100.0, 100.0]],
                dtype=np.float32,
            ),
            max_features=4,
            minimum_distance=8.0,
        )

        np.testing.assert_allclose(
            features,
            np.array(
                [[10.0, 10.0], [40.0, 40.0], [70.0, 70.0], [100.0, 100.0]],
                dtype=np.float32,
            ),
        )

    def test_feature_mask_removes_pixels_near_non_field_boundaries(self):
        field_mask = np.zeros((60, 60), dtype=np.uint8)
        field_mask[10:50, 10:50] = 255

        feature_mask = build_feature_mask(field_mask, margin=5)

        self.assertEqual(feature_mask[14, 30], 0)
        self.assertEqual(feature_mask[15, 30], 255)
        self.assertEqual(feature_mask[30, 30], 255)

    def test_filters_features_outside_the_safe_field_mask(self):
        feature_mask = np.zeros((60, 60), dtype=np.uint8)
        feature_mask[10:50, 10:50] = 255

        kept = filter_features_by_mask(
            np.array([[9.0, 20.0], [20.0, 20.0], [51.0, 30.0]], dtype=np.float32),
            feature_mask,
        )

        np.testing.assert_allclose(
            kept,
            np.array([[20.0, 20.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
