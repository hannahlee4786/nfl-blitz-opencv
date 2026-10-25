import unittest

import numpy as np

from ml.track_assoc import (
    DEFAULTS,
    Det,
    per_frame_suppress,
    repair_bad_handoffs,
    repair_merged_occlusions,
    suppress_interpolated_cross_team_duplicates,
    split_sustained_team_drift,
    suppress_duplicates,
)


def detection(uid, frame, x, team="red", width=30, height=60, conf=0.9):
    return Det(
        uid,
        frame,
        x,
        20,
        width,
        height,
        conf,
        team,
        np.zeros(48, dtype=np.float32),
    )


class TrackAssociationTests(unittest.TestCase):
    def test_still_suppresses_normal_partial_duplicate(self):
        body = detection(1, 0, 10, width=32, height=80, conf=0.9)
        torso = detection(2, 0, 12, width=28, height=40, conf=0.7)

        kept = suppress_duplicates([body, torso], 0.70)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].uid, 1)

    def test_repairs_group_box_and_following_detector_gap(self):
        steady = {
            frame: detection(frame, frame, 0, team="red") for frame in range(5)
        }
        occluded = {
            0: detection(10, 0, 10, team="blue"),
            1: detection(11, 1, 0, team="red", width=80, height=70),
            4: detection(14, 4, 10, team="blue"),
        }
        motion = np.repeat(
            np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
            5,
            axis=0,
        )
        paths = [steady, occluded]

        repaired = repair_merged_occlusions(paths, motion, dict(DEFAULTS))

        self.assertEqual(repaired, 3)
        self.assertEqual(sorted(occluded), [0, 1, 2, 3, 4])
        for frame in (1, 2, 3):
            self.assertEqual(occluded[frame].uid, -2)
            self.assertAlmostEqual(occluded[frame].x, 10)
            self.assertAlmostEqual(occluded[frame].w, 30)

        removed = per_frame_suppress(paths, ["red", "blue"])

        self.assertEqual(removed, 0)
        self.assertIn(1, steady)
        self.assertIn(1, occluded)

    def test_repairs_late_reid_handoff_after_bad_suffix(self):
        original = np.array([1.0, 0.0], dtype=np.float32)
        wrong = np.array([0.0, 1.0], dtype=np.float32)
        main = {}
        for frame in range(66):
            det = detection(frame, frame, frame * 0.2, team="blue")
            det.emb = original if frame < 60 else wrong
            main[frame] = det
        continuation = {}
        for frame in range(60, 90):
            det = detection(1000 + frame, frame, frame * 0.2 + 2, team="blue")
            det.emb = np.array([0.98, 0.2], dtype=np.float32)
            continuation[frame] = det
        paths = [main, continuation]

        repaired = repair_bad_handoffs(paths, ["blue", "blue"])

        self.assertEqual(repaired, 1)
        self.assertEqual(sorted(main), list(range(90)))
        self.assertGreater(main[60].uid, 1000)
        self.assertEqual(continuation, {})

    def test_drops_only_interpolated_cross_team_duplicate(self):
        red_path = {
            frame: detection(frame, frame, 20, team="red", width=60, height=100)
            for frame in range(10, 14)
        }
        blue_path = {
            frame: detection(-1, frame, 25, team="blue", width=45, height=90)
            for frame in range(10, 14)
        }
        paths = [red_path, blue_path]

        removed = suppress_interpolated_cross_team_duplicates(
            paths, ["red", "blue"]
        )

        self.assertEqual(removed, 4)
        self.assertEqual(sorted(paths[0]), [10, 11, 12, 13])
        self.assertEqual(paths[1], {})

    def test_keeps_brief_interpolated_opponent_crossing(self):
        red = detection(1, 10, 20, team="red", width=60, height=100)
        blue_interpolation = detection(
            -1, 10, 25, team="blue", width=45, height=90
        )
        paths = [{10: red}, {10: blue_interpolation}]

        removed = suppress_interpolated_cross_team_duplicates(
            paths, ["red", "blue"]
        )

        self.assertEqual(removed, 0)
        self.assertIn(10, paths[0])
        self.assertIn(10, paths[1])

    def test_splits_sustained_low_similarity_opponent_handoff(self):
        blue = np.array([1.0, 0.0], dtype=np.float32)
        red = np.array([0.0, 1.0], dtype=np.float32)
        path = {}
        for frame in range(40):
            det = detection(frame, frame, frame, team="blue")
            det.emb = blue
            path[frame] = det
        for frame in range(40, 70):
            det = detection(frame, frame, frame, team="red")
            det.emb = red
            path[frame] = det
        for frame in range(70, 74):
            det = detection(frame, frame, frame, team="blue")
            det.emb = blue
            path[frame] = det
        for frame in range(74, 78):
            det = detection(frame, frame, frame, team="red")
            det.emb = red
            path[frame] = det

        paths = [path]
        teams = ["blue"]
        moved, new_tracks = split_sustained_team_drift(paths, teams)

        self.assertEqual(moved, 34)
        self.assertEqual(new_tracks, 2)
        self.assertNotIn(40, path)
        self.assertNotIn(77, path)
        self.assertIn(39, path)
        self.assertIn(70, path)
        self.assertEqual(teams, ["blue", "red", "red"])
        self.assertEqual(sorted(paths[1]), list(range(40, 70)))
        self.assertEqual(sorted(paths[2]), list(range(74, 78)))

    def test_keeps_short_opponent_color_flicker(self):
        blue = np.array([1.0, 0.0], dtype=np.float32)
        red = np.array([0.0, 1.0], dtype=np.float32)
        path = {}
        for frame in range(40):
            det = detection(frame, frame, frame, team="blue")
            det.emb = blue
            path[frame] = det
        for frame in range(40, 56):
            det = detection(frame, frame, frame, team="red")
            det.emb = red
            path[frame] = det

        paths = [path]
        teams = ["blue"]
        moved, new_tracks = split_sustained_team_drift(paths, teams)

        self.assertEqual(moved, 0)
        self.assertEqual(new_tracks, 0)
        self.assertEqual(teams, ["blue"])
        self.assertEqual(sorted(path), list(range(56)))


if __name__ == "__main__":
    unittest.main()
