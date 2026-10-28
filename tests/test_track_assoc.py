import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ml.detect_all import ICON_AFTER as DETECT_ICON_AFTER
from ml.detect_all import ICON_BEFORE as DETECT_ICON_BEFORE
from ml.detect_all import ICON_NONE, icon_level, is_icon_box
from ml.track_assoc import (
    DEFAULTS,
    ICON_AFTER,
    ICON_BEFORE,
    Det,
    Tlet,
    Track,
    build_frames,
    jersey_margin,
    limit_roster,
    load_icons,
    remove_cone_runs,
    strip_icon_chains,
    pair_cost,
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


def torso_hist(red=0.0, blue=0.0):
    """A 12x4 hue-saturation histogram with saturated red and blue mass."""
    counts = np.zeros((12, 4), dtype=np.float32)
    counts[0, 3] = red
    counts[7, 3] = blue
    counts[4, 0] = max(0.0, 1.0 - red - blue)
    return counts.reshape(-1)


class OpponentJerseyTests(unittest.TestCase):
    def test_jersey_margin_separates_clear_and_mixed_torsos(self):
        self.assertAlmostEqual(jersey_margin(Det(1, 0, 0, 0, 10, 20, 0.9, "red", torso_hist(red=0.5))), 1.0)
        self.assertLess(jersey_margin(Det(2, 0, 0, 0, 10, 20, 0.9, "red", torso_hist(red=0.3, blue=0.25))), 0.6)
        self.assertEqual(jersey_margin(Det(3, 0, 0, 0, 10, 20, 0.9, "red", torso_hist())), 0.0)

    def _blue_track(self):
        track = Track(Det(0, 0, 100, 20, 30, 60, 0.9, "blue", torso_hist(blue=0.5)))
        track.blue = 6
        return track

    def test_clear_opponent_jersey_costs_the_team_penalty(self):
        track = self._blue_track()
        opponent = Det(1, 1, 104, 20, 30, 60, 0.9, "red", torso_hist(red=0.5))
        pred = (100.0, 20.0, 30.0, 60.0)

        plain = pair_cost(pred, opponent, track, 0.0)
        gated = pair_cost(pred, opponent, track, 0.0, team_penalty=1.0, team_margin=0.6)

        self.assertAlmostEqual(gated - plain, 1.0)

    def test_mixed_jersey_keeps_the_old_team_cost(self):
        track = self._blue_track()
        tangled = Det(1, 1, 104, 20, 30, 60, 0.9, "red", torso_hist(red=0.3, blue=0.25))
        pred = (100.0, 20.0, 30.0, 60.0)

        plain = pair_cost(pred, tangled, track, 0.0)
        gated = pair_cost(pred, tangled, track, 0.0, team_penalty=1.0, team_margin=0.6)

        self.assertAlmostEqual(gated, plain)

    def test_partial_body_with_red_torso_gets_no_team_penalty(self):
        # Legs plus the carried football: the torso band reads red.
        track = self._blue_track()
        partial = Det(1, 1, 104, 50, 30, 40, 0.9, "red", torso_hist(red=0.5))
        pred = (100.0, 20.0, 30.0, 60.0)

        plain = pair_cost(pred, partial, track, 0.0)
        exempt = pair_cost(pred, partial, track, 0.0, team_penalty=0.5, team_margin=0.6, team_partial=0.9)

        self.assertAlmostEqual(exempt, plain)

    def test_full_body_opponent_still_costs_the_team_penalty(self):
        track = self._blue_track()
        opponent = Det(1, 1, 104, 20, 30, 58, 0.9, "red", torso_hist(red=0.5))
        pred = (100.0, 20.0, 30.0, 60.0)

        plain = pair_cost(pred, opponent, track, 0.0)
        gated = pair_cost(pred, opponent, track, 0.0, team_penalty=0.5, team_margin=0.6, team_partial=0.9)

        self.assertAlmostEqual(gated - plain, 0.5)

    def test_same_team_detection_is_unaffected(self):
        track = self._blue_track()
        teammate = Det(1, 1, 104, 20, 30, 60, 0.9, "blue", torso_hist(blue=0.5))
        pred = (100.0, 20.0, 30.0, 60.0)

        self.assertAlmostEqual(
            pair_cost(pred, teammate, track, 0.0, team_penalty=1.0, team_margin=0.6),
            pair_cost(pred, teammate, track, 0.0),
        )


def chain(frames, conf, team="red", x=0):
    """One-tracklet chain with a detection on each listed frame."""
    obs = [detection(1000 * x + f, f, x, team=team, conf=conf) for f in frames]
    return [Tlet(obs, team, 0.0, 0.0, 0, set())]


class RosterRescueTests(unittest.TestCase):
    """Seven strong red chains plus one weaker 14-detection red chain ranked eighth."""

    def roster(self, strong_frames, extra_frames):
        strong = [chain(strong_frames, 0.9, x=i) for i in range(7)]
        extra = chain(extra_frames, 0.35, x=50)
        return strong, extra

    def test_rescue_off_keeps_the_old_whole_clip_ranking(self):
        strong, extra = self.roster(range(0, 6), range(30, 44))

        kept = limit_roster(strong + [extra])

        self.assertNotIn(extra, kept)
        self.assertEqual(len(kept), 7)

    def test_rescues_chain_when_its_frames_have_fewer_than_seven_teammates(self):
        strong, extra = self.roster(range(0, 6), range(30, 44))

        kept = limit_roster(strong + [extra], rescue_len=12, rescue_conf=0.30, interp_gap=10)

        self.assertIn(extra, kept)

    def test_keeps_dropping_when_seven_teammates_are_on_its_frames(self):
        strong, extra = self.roster(range(25, 31), range(30, 44))

        kept = limit_roster(strong + [extra], rescue_len=12, rescue_conf=0.30, interp_gap=10)

        self.assertNotIn(extra, kept)

    def test_counts_frames_present_not_overlapping_lifetimes(self):
        # Every strong chain lives 0-55 but has no box from 6 to 49, a gap too
        # long to interpolate, so frames 15-28 have no strong teammate.
        strong, extra = self.roster(list(range(0, 6)) + list(range(50, 56)), range(15, 29))

        kept = limit_roster(strong + [extra], rescue_len=12, rescue_conf=0.30, interp_gap=10)

        self.assertIn(extra, kept)

    def test_interpolated_frames_count_as_present(self):
        # Strong chains are seen on frames 0 and 21 only. With a 20-frame
        # interpolation gap they still publish boxes on 1-20, where the extra
        # chain is; without interpolation those frames are free.
        strong, extra = self.roster([0, 21, 100, 101, 102, 103], range(3, 17))

        filled = limit_roster(strong + [extra], rescue_len=12, rescue_conf=0.30, interp_gap=20)
        unfilled = limit_roster(strong + [extra], rescue_len=12, rescue_conf=0.30, interp_gap=0)

        self.assertNotIn(extra, filled)
        self.assertIn(extra, unfilled)

    def test_short_or_weak_chain_is_not_rescued(self):
        strong, _extra = self.roster(range(0, 6), range(30, 44))
        short = chain(range(30, 41), 0.35, x=60)
        weak = chain(range(30, 44), 0.25, x=70)

        kept = limit_roster(strong + [short, weak], rescue_len=12, rescue_conf=0.30, interp_gap=10)

        self.assertNotIn(short, kept)
        self.assertNotIn(weak, kept)


class ConeRunTests(unittest.TestCase):
    """A 1356x1016 frame and a still camera: small means at most 61 px tall and 54 px wide."""

    W, H = 1356, 1016

    def path(self, player_frames, pylon_frames, pylon_step=0.0):
        path = {}
        uid = 0
        for frame in player_frames:
            path[frame] = Det(uid, frame, 300 + 6 * frame, 400, 70, 140, 0.8, "red", np.zeros(48))
            uid += 1
        for k, frame in enumerate(pylon_frames):
            path[frame] = Det(uid, frame, 1060 + pylon_step * k, 440, 32, 55, 0.1, "red", np.zeros(48))
            uid += 1
        return path

    def run_filter(self, paths, warm_value=0.5, min_frames=12):
        warm = np.full(200, warm_value, dtype=np.float32)
        to_ref = [np.eye(3)] * 300
        return remove_cone_runs(paths, warm, to_ref, self.W, self.H, min_frames, 0.30)

    def test_removes_a_pylon_run_at_the_end_of_a_player_track(self):
        path = self.path(range(0, 40), range(50, 62))

        removed = self.run_filter([path])

        self.assertEqual(removed, 12)
        self.assertEqual(sorted(path), list(range(0, 40)))

    def test_keeps_a_run_shorter_than_the_minimum(self):
        path = self.path(range(0, 40), range(50, 61))

        self.assertEqual(self.run_filter([path]), 0)
        self.assertEqual(len(path), 51)

    def test_keeps_a_small_box_that_moves_on_the_field(self):
        # A distant player: small boxes, but the feet travel 8 px a frame.
        path = self.path(range(0, 40), range(50, 70), pylon_step=8.0)

        self.assertEqual(self.run_filter([path]), 0)

    def test_keeps_a_small_still_box_that_is_not_warm(self):
        path = self.path(range(0, 40), range(50, 70))

        self.assertEqual(self.run_filter([path], warm_value=0.1), 0)

    def test_leaves_pylon_sized_tracks_to_the_whole_track_filter(self):
        path = self.path(range(0, 5), range(10, 40))

        self.assertEqual(self.run_filter([path]), 0)
        self.assertEqual(len(path), 35)

    def test_off_without_warm_cache_or_with_zero_frames(self):
        path = self.path(range(0, 40), range(50, 62))

        self.assertEqual(remove_cone_runs([path], None, [np.eye(3)] * 300, self.W, self.H, 12, 0.30), 0)
        self.assertEqual(self.run_filter([path], min_frames=0), 0)
        self.assertEqual(len(path), 52)


def plaque_frame():
    """Green field with a dark '1' disc at (200, 300) and a blue arrow to its right."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[:] = (90, 140, 40)
    cv2.circle(frame, (232, 332), 30, (25, 25, 25), -1)
    cv2.line(frame, (236, 314), (236, 350), (255, 255, 255), 9)
    cv2.rectangle(frame, (268, 318), (300, 346), (255, 60, 0), -1)
    return frame


class IndicatorIconTests(unittest.TestCase):
    def setUp(self):
        self.hsv = cv2.cvtColor(plaque_frame(), cv2.COLOR_BGR2HSV)
        self.arrow = [(284.0, 332.0)]

    def test_flags_weak_short_plaque_beside_arrow(self):
        self.assertTrue(is_icon_box(self.hsv, 200, 300, 64, 64, 0.15, 260.0, self.arrow))

    def test_keeps_confident_box(self):
        self.assertFalse(is_icon_box(self.hsv, 200, 300, 64, 64, 0.45, 260.0, self.arrow))

    def test_keeps_box_without_an_arrow_beside_it(self):
        self.assertFalse(is_icon_box(self.hsv, 200, 300, 64, 64, 0.15, 260.0, []))

    def test_keeps_player_sized_box(self):
        self.assertFalse(is_icon_box(self.hsv, 200, 300, 64, 64, 0.15, 100.0, self.arrow))

    def test_build_frames_drops_only_before_tracking_icons(self):
        meta = {"width": 800, "height": 600, "nframes": 1}
        rows = [
            {"frame": "0", "x": "200", "y": "300", "w": "64", "h": "64", "conf": "0.15", "team": "blue"},
            {"frame": "0", "x": "500", "y": "200", "w": "60", "h": "160", "conf": "0.80", "team": "blue"},
            {"frame": "0", "x": "40", "y": "300", "w": "64", "h": "64", "conf": "0.15", "team": "red"},
        ]
        hist = np.zeros((3, 48), dtype=np.float32)
        hud = np.zeros((150, 200), dtype=np.uint8)
        icons = np.array([ICON_BEFORE, 0, ICON_AFTER], dtype=np.uint8)

        kept = build_frames(meta, rows, hist, hud, dict(DEFAULTS), icons=icons)
        unflagged = build_frames(meta, rows, hist, hud, dict(DEFAULTS))

        self.assertEqual(sorted(det.uid for det in kept[0]), [1, 2])
        self.assertEqual(sorted(det.uid for det in unflagged[0]), [0, 1, 2])

    def test_icon_level_separates_the_two_plaque_tests(self):
        # A taller frame so the 64 px plaque is under 10% of its height.
        tall = np.vstack([plaque_frame(), np.tile(plaque_frame()[:1], (400, 1, 1))])
        hsv = cv2.cvtColor(tall, cv2.COLOR_BGR2HSV)

        # Small next to the players: the original test, dropped before tracking.
        self.assertEqual(icon_level(hsv, 200, 300, 64, 64, 0.15, 260.0, self.arrow), DETECT_ICON_BEFORE)
        # Players on a wide shot are barely taller than the plaque.
        self.assertEqual(icon_level(hsv, 200, 300, 64, 64, 0.15, 100.0, self.arrow), DETECT_ICON_AFTER)
        # The same box is not the fixed-size plaque once it is over 10% of the frame.
        self.assertEqual(icon_level(self.hsv, 200, 300, 64, 64, 0.15, 100.0, self.arrow), ICON_NONE)
        self.assertEqual(icon_level(hsv, 200, 300, 64, 64, 0.15, 100.0, []), ICON_NONE)


class IconChainTests(unittest.TestCase):
    def chain_with_icons(self, real_frames, icon_frames, start_uid):
        obs = []
        for offset, frame in enumerate(sorted(real_frames + icon_frames)):
            obs.append(detection(start_uid + offset, frame, frame, team="blue"))
        icon_uids = {det.uid for det in obs if det.frame in icon_frames}
        return [Tlet(obs, "blue", 0.0, 0.0, 0, set())], icon_uids

    def test_keeps_real_detections_and_strips_icons_from_a_player_chain(self):
        player, icon_uids = self.chain_with_icons(list(range(0, 20)), [20, 21], 0)
        icons = np.zeros(100, dtype=np.uint8)
        icons[list(icon_uids)] = ICON_AFTER

        kept, dropped, stripped = strip_icon_chains([player], icons, 0.5)

        self.assertEqual(dropped, 0)
        self.assertEqual(stripped, 2)
        self.assertEqual([det.frame for det in kept[0][1]], list(range(0, 20)))

    def test_drops_a_chain_that_is_mostly_icon(self):
        icon_track, icon_uids = self.chain_with_icons([44, 45], list(range(10, 25)), 0)
        icons = np.zeros(100, dtype=np.uint8)
        icons[list(icon_uids)] = ICON_AFTER

        kept, dropped, _stripped = strip_icon_chains([icon_track], icons, 0.5)

        self.assertEqual(kept, [])
        self.assertEqual(dropped, 1)

    def test_half_icon_chain_is_dropped_and_less_is_kept(self):
        half, half_icons = self.chain_with_icons([0, 1, 2], [3, 4, 5], 0)
        less, less_icons = self.chain_with_icons([10, 11, 12, 13], [14, 15, 16], 50)
        icons = np.zeros(100, dtype=np.uint8)
        icons[list(half_icons | less_icons)] = ICON_AFTER

        kept, dropped, stripped = strip_icon_chains([half, less], icons, 0.5)

        self.assertEqual(dropped, 1)
        self.assertEqual([[det.frame for det in obs] for _chain, obs in kept], [[10, 11, 12, 13]])
        self.assertEqual(stripped, 3)

    def test_old_bool_icon_file_means_drop_before_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "clip_yolo11m"
            np.save(stem.with_suffix(".icons.npy"), np.array([True, False]))

            icons = load_icons(stem, 2)

        self.assertEqual(icons.tolist(), [ICON_BEFORE, 0])


if __name__ == "__main__":
    unittest.main()
