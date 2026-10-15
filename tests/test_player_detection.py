import unittest

import numpy as np

from pipeline.track_clip4 import (
    _claim_side_marker_body,
    _collapse_handoff_tracks,
    _collapse_split_tracks,
    _fill_main_track_gaps,
    _jersey_bodies,
    _on_football,
    _tight_qb_body,
    _without_field_lines,
    _without_head_color,
    _without_head_fragments,
    _without_player_markers,
)


class PlayerDetectionTests(unittest.TestCase):
    @staticmethod
    def _track(box, confidence=0.8):
        return {
            "bbox": box,
            "conf": confidence,
            "pos": (0.0, 0.0),
            "world": (0.0, 0.0),
        }

    def test_drops_small_aligned_helmet_fragment(self):
        helmet = {"bbox": (82, 28, 36, 28), "area": 520}
        jersey = {"bbox": (52, 92, 92, 82), "area": 3100}

        kept = _without_head_fragments([helmet, jersey])

        self.assertEqual(kept, [jersey])

    def test_keeps_two_full_players_aligned_in_depth(self):
        far_player = {"bbox": (80, 28, 42, 58), "area": 1400}
        near_player = {"bbox": (52, 102, 92, 82), "area": 3100}

        kept = _without_head_fragments([far_player, near_player])

        self.assertEqual(kept, [far_player, near_player])

    def test_removes_helmet_pixels_from_live_tracking_mask(self):
        mask = np.zeros((220, 220), dtype=bool)
        mask[28:56, 82:118] = True
        mask[92:174, 52:144] = True

        clean = _without_head_color(mask)

        self.assertFalse(clean[28:56, 82:118].any())
        self.assertTrue(clean[92:174, 52:144].all())

    def test_removes_ring_and_arrow_but_preserves_jersey(self):
        mask = np.zeros((180, 220), dtype=bool)
        mask[25:55, 90:130] = True  # arrow
        mask[68:108, 82:138] = True  # jersey
        mask[128:158, 45:175] = True  # selection ring
        ring = {
            "cx": 110,
            "cy": 143,
            "r": 30,
            "bbox": (45, 128, 130, 30),
        }
        arrow = (110, 40, 30)

        clean = _without_player_markers(mask, ring, arrow)

        self.assertFalse(clean[25:55, 90:130].any())
        self.assertFalse(clean[128:158, 45:175].any())
        self.assertTrue(clean[68:108, 82:138].all())

    def test_removes_horizontal_field_line_but_preserves_player(self):
        mask = np.zeros((180, 260), dtype=bool)
        mask[92:99, 12:248] = True
        mask[55:132, 108:154] = True

        clean = _without_field_lines(mask)

        self.assertFalse(clean[92:99, 12:100].any())
        self.assertTrue(clean[60:86, 114:148].all())
        self.assertTrue(clean[106:126, 114:148].all())

    def test_sideways_jersey_is_still_a_body_without_legs_below(self):
        hsv = np.zeros((240, 280, 3), dtype=np.uint8)
        hsv[:] = (82, 170, 95)
        hsv[92:150, 92:174] = (2, 180, 115)
        mask = np.zeros((240, 280), dtype=bool)
        mask[92:150, 92:174] = True

        bodies = _jersey_bodies(mask, hsv.shape, hsv)

        self.assertTrue(any(
            x <= 133 <= x + width and y <= 121 <= y + height
            for x, y, width, height, _fx, _fy in bodies
        ))

    def test_airborne_qb_box_uses_jersey_instead_of_ground_circle(self):
        bodies = [
            (772, 416, 28, 109, 786.0, 525.0),  # separated leg
            (799, 378, 70, 137, 834.0, 515.0),  # QB torso
            (141, 447, 110, 214, 196.0, 661.0),
            (588, 576, 28, 63, 602.0, 639.0),
        ]

        result = _tight_qb_body(
            bodies, (797.0, 548.0), (1016, 1356, 3),
            mode="arrow", allow_below=False)

        self.assertIsNotNone(result)
        box, foot = result
        self.assertLess(box[1] + box[3], 548)
        self.assertLess(foot[1], 548)
        self.assertLessEqual(box[0], 772)
        self.assertGreaterEqual(box[0] + box[2], 869)

    def test_qb_ring_selects_full_body_not_helmet_fragment(self):
        bodies = [
            (842, 236, 50, 188, 866.5, 424.0),  # helmet fragment
            (785, 317, 116, 220, 843.0, 537.0),  # full QB body
        ]

        result = _tight_qb_body(
            bodies, (835.6, 592.7), (1016, 1356, 3), mode="ring")

        self.assertIsNotNone(result)
        box, _foot = result
        self.assertGreater(box[1], 280)
        self.assertLess(box[1] + box[3], 555)

    def test_opening_side_marker_claims_qb_jersey_not_lower_fragment(self):
        full_qb = {
            "team": "blue", "bbox": (639, 521, 78, 197), "main": False,
        }
        lower_fragment = {
            "team": "blue", "bbox": (639, 626, 98, 157), "main": False,
        }
        lineman = {
            "team": "blue", "bbox": (493, 529, 100, 135), "main": False,
        }

        claimed = _claim_side_marker_body(
            [lower_fragment, full_qb, lineman], (493, 696, 58, 50))

        self.assertIs(claimed, full_qb)
        self.assertTrue(full_qb["main"])
        self.assertFalse(lower_fragment["main"])

    def test_rejects_warm_football_oval_on_grass(self):
        hsv = np.zeros((180, 220, 3), dtype=np.uint8)
        hsv[:] = (82, 170, 95)
        yy, xx = np.ogrid[:180, :220]
        football = ((xx - 110) ** 2 / 31 ** 2
                    + (yy - 118) ** 2 / 19 ** 2) <= 1.0
        hsv[football] = (7, 125, 132)

        self.assertTrue(_on_football(hsv, (70, 58, 80, 82)))

    def test_keeps_red_jersey_with_pants_below(self):
        hsv = np.zeros((180, 220, 3), dtype=np.uint8)
        hsv[:] = (82, 170, 95)
        hsv[55:108, 82:138] = (176, 145, 90)
        hsv[108:150, 88:132] = (0, 35, 155)

        self.assertFalse(_on_football(hsv, (72, 40, 76, 120)))

    def test_rejects_ball_seed_when_box_grows_around_blue_player(self):
        hsv = np.zeros((220, 240, 3), dtype=np.uint8)
        hsv[:] = (82, 170, 95)
        hsv[42:142, 96:154] = (118, 145, 92)
        yy, xx = np.ogrid[:220, :240]
        football = ((xx - 125) ** 2 / 31 ** 2
                    + (yy - 154) ** 2 / 18 ** 2) <= 1.0
        hsv[football] = (7, 125, 132)

        self.assertTrue(_on_football(hsv, (72, 24, 106, 170)))

    def test_collapses_temporally_supported_airborne_body_fragments(self):
        fused = [
            {
                8: self._track((10, 20, 28, 100), 0.7),
                9: self._track((8, 10, 70, 112), 0.9),
            },
            {8: self._track((30, 30, 70, 120), 0.8)},
            {
                8: self._track((50, 90, 28, 110), 0.9),
                9: self._track((80, 20, 90, 100), 0.6),
            },
            {
                8: self._track((100, 40, 28, 90), 0.7),
                9: self._track((96, 35, 72, 98), 0.9),
            },
        ]

        _collapse_split_tracks(fused, {8: "blue", 9: "blue"})

        self.assertEqual([list(frame) for frame in fused], [[9], [9], [9], [9]])
        self.assertEqual(fused[2][9]["bbox"], (50, 20, 120, 180))

    def test_keeps_adjacent_full_player_tracks_separate(self):
        fused = [{
            7: self._track((20, 30, 54, 120)),
            8: self._track((78, 32, 58, 118)),
        }]

        _collapse_split_tracks(fused, {7: "blue", 8: "blue"})

        self.assertEqual(set(fused[0]), {7, 8})

    def test_collapses_overlapping_torso_and_foot_boxes(self):
        fused = [{
            1: self._track((528, 524, 34, 109)),
            2: self._track((527, 568, 48, 62)),
        }]

        _collapse_split_tracks(fused, {1: "red", 2: "red"})

        self.assertEqual(len(fused[0]), 1)
        only = next(iter(fused[0].values()))
        self.assertEqual(only["bbox"], (527, 524, 48, 109))

    def test_removes_brief_track_id_handoff_overlap(self):
        fused = []
        for frame in range(8):
            row = {}
            if frame <= 4:
                row[9] = self._track((100 + frame, 40, 70, 140))
                row[9]["pos"] = (float(frame), 10.0)
            if frame >= 3:
                row[8] = self._track((103 + frame, 42, 72, 138))
                row[8]["pos"] = (float(frame) + 0.2, 10.1)
            fused.append(row)

        _collapse_handoff_tracks(fused, {8: "blue", 9: "blue"})

        self.assertEqual(set(fused[3]), {8})
        self.assertEqual(set(fused[4]), {8})

    def test_fills_short_airborne_main_player_gap(self):
        fused = [
            {9: self._track((100, 20, 60, 120), 0.9)},
            {},
            {},
            {9: self._track((130, 26, 66, 126), 0.6)},
        ]
        fused[0][9]["pos"] = (10.0, 20.0)
        fused[0][9]["world"] = (1.0, 2.0)
        fused[3][9]["pos"] = (13.0, 23.0)
        fused[3][9]["world"] = (4.0, 5.0)

        _fill_main_track_gaps(fused, 9)

        self.assertEqual(fused[1][9]["bbox"], (110, 22, 62, 122))
        self.assertEqual(fused[2][9]["bbox"], (120, 24, 64, 124))
        self.assertEqual(fused[1][9]["pos"], (11.0, 21.0))


if __name__ == "__main__":
    unittest.main()
