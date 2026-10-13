import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.team_classifier import (
    build_color_prototype,
    classify_jersey_region,
    crop_region,
    prototype_distance,
)
from src.classify_teams import load_calibration_report
from pipeline.track_clip4 import _jersey_box, _with_team_classification


def solid_bgr(color, width=20, height=20):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


class TeamClassifierTests(unittest.TestCase):
    def setUp(self):
        self.red = build_color_prototype(solid_bgr((0, 0, 220)), "RED")
        self.blue = build_color_prototype(solid_bgr((220, 40, 20)), "BLUE")

    def test_classifies_both_team_colors(self):
        red_result = classify_jersey_region(
            solid_bgr((15, 25, 190)), self.red, self.blue
        )
        blue_result = classify_jersey_region(
            solid_bgr((190, 55, 30)), self.red, self.blue
        )

        self.assertEqual(red_result.label, "RED")
        self.assertEqual(blue_result.label, "BLUE")
        self.assertGreater(red_result.confidence, 0.95)
        self.assertGreater(blue_result.confidence, 0.95)

    def test_ignores_white_and_black_uniform_details(self):
        region = solid_bgr((0, 0, 210))
        region[:5, :] = (255, 255, 255)
        region[5:8, :] = (5, 5, 5)

        result = classify_jersey_region(region, self.red, self.blue)

        self.assertEqual(result.label, "RED")
        self.assertEqual(result.valid_pixel_count, 240)

    def test_returns_unknown_when_no_jersey_color_matches(self):
        result = classify_jersey_region(
            solid_bgr((0, 220, 0)), self.red, self.blue
        )

        self.assertEqual(result.label, "UNKNOWN")
        self.assertEqual(result.matched_pixel_count, 0)

    def test_rejects_small_team_colored_patch_inside_unrelated_region(self):
        region = solid_bgr((0, 200, 0), width=100, height=100)
        region[:5, :5] = (220, 40, 20)

        result = classify_jersey_region(region, self.red, self.blue)

        self.assertEqual(result.label, "UNKNOWN")
        self.assertEqual(result.confidence, 0.0)
        self.assertLess(result.matched_fraction, 0.01)
        self.assertEqual(result.vote_agreement, 1.0)

    def test_reports_team_confidence_and_color_coverage_separately(self):
        region = solid_bgr((0, 200, 0), width=20, height=20)
        region[:, :10] = (220, 40, 20)

        result = classify_jersey_region(region, self.red, self.blue)

        self.assertEqual(result.label, "BLUE")
        self.assertAlmostEqual(result.confidence, 1.0)
        self.assertAlmostEqual(result.matched_fraction, 0.5)

    def test_team_prototypes_are_well_separated(self):
        self.assertGreater(prototype_distance(self.red, self.blue), 0.5)

    def test_crop_region_clips_to_frame(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cropped = crop_region(frame, (-2, -3, 7, 8))
        self.assertEqual(cropped.shape, (5, 5, 3))

    def test_hue_distance_wraps_around_red(self):
        red_at_upper_hue = cv2.cvtColor(
            np.array([[[179, 230, 220]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )
        region = np.tile(red_at_upper_hue, (20, 20, 1))

        result = classify_jersey_region(region, self.red, self.blue)

        self.assertEqual(result.label, "RED")

    def test_loads_saved_team_color_calibration(self):
        report = {
            "team_a": {
                "label": "Tampa",
                "hsv": [176.68, 179.0, 110.0],
                "bgr": [41, 33, 110],
                "sample_pixel_count": 573,
            },
            "team_b": {
                "label": "Dallas",
                "hsv": [120.33, 199.0, 104.0],
                "bgr": [104, 23, 23],
                "sample_pixel_count": 361,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "calibration.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            team_a, team_b = load_calibration_report(report_path)

        self.assertEqual(team_a.label, "Tampa")
        self.assertEqual(team_b.label, "Dallas")
        self.assertAlmostEqual(team_a.hsv[0], 176.68)
        self.assertEqual(team_b.sample_pixel_count, 361)

    def test_automatically_classifies_a_tracked_player_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        player_box = (20, 10, 30, 60)
        x, y, width, height = _jersey_box(player_box)
        frame[y:y + height, x:x + width] = (10, 20, 210)
        tracks = [{"id": 4, "team": "red", "bbox": player_box}]

        classified = _with_team_classification(
            frame, tracks, (self.red, self.blue)
        )

        self.assertEqual(classified[0]["team_name"], "RED")
        self.assertGreater(classified[0]["team_conf"], 0.95)
        self.assertGreater(classified[0]["color_coverage"], 0.95)


if __name__ == "__main__":
    unittest.main()
