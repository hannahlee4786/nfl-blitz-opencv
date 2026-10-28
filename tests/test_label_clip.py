import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.label_clip import LabelState, dedupe, draft_frames, problems


def labels_with(frames):
    return {"video": "v.mp4", "crop_x0": 0, "frames": {str(f): entries for f, entries in frames.items()}}


def entry(person, team, box, main=False, hidden=False):
    return {"person": person, "team": team, "box": list(box), "main": main, "hidden": hidden}


class DraftTests(unittest.TestCase):
    def test_draft_uses_raw_detections_with_empty_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "clip_yolo11m"
            rows = [
                {"frame": 0, "id": 1, "team": "red", "x": 10, "y": 10, "w": 40, "h": 90, "conf": 0.9},
                {"frame": 0, "id": 2, "team": "red", "x": 12, "y": 12, "w": 38, "h": 85, "conf": 0.5},   # same body
                {"frame": 0, "id": 3, "team": "blue", "x": 200, "y": 10, "w": 40, "h": 90, "conf": 0.2},  # below conf
                {"frame": 0, "id": 4, "team": "blue", "x": 300, "y": 10, "w": 30, "h": 30, "conf": 0.6},  # icon
                {"frame": 5, "id": 5, "team": "blue", "x": 400, "y": 10, "w": 40, "h": 90, "conf": 0.8},  # not sampled
            ]
            with stem.with_suffix(".csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.save(stem.with_suffix(".icons.npy"), np.array([0, 0, 0, 2, 0], dtype=np.uint8))

            frames = draft_frames(stem, [0, 10], 0.30)

        self.assertEqual(list(frames), ["0", "10"])
        self.assertEqual(frames["10"], [])
        self.assertEqual(len(frames["0"]), 1)
        self.assertEqual(frames["0"][0]["person"], "")
        self.assertEqual(frames["0"][0]["box"], [10, 10, 40, 90])

    def test_dedupe_keeps_the_stronger_box(self):
        kept = dedupe([{"box": [0, 0, 40, 90], "conf": 0.4}, {"box": [2, 2, 38, 86], "conf": 0.8},
                       {"box": [200, 0, 40, 90], "conf": 0.3}])

        self.assertEqual([k["conf"] for k in kept], [0.8, 0.3])


class EditTests(unittest.TestCase):
    def setUp(self):
        self.state = LabelState(labels_with({
            0: [entry("red55", "red", (100, 100, 40, 90)), entry("blue8", "blue", (300, 100, 40, 90), main=True)],
            10: [entry("", "red", (110, 104, 40, 90)), entry("", "blue", (320, 100, 40, 90)),
                 entry("", "red", (700, 100, 40, 90))],
        }))
        self.state.move(1)

    def test_select_add_delete_and_undo(self):
        self.assertEqual(self.state.select_at(120, 150), 0)
        self.state.delete()
        self.assertEqual(len(self.state.entries), 2)
        self.state.add([500, 100, 40, 80], "blue")
        self.assertEqual(self.state.entries[-1]["person"], "")
        self.state.undo()
        self.state.undo()
        self.assertEqual(len(self.state.entries), 3)

    def test_naming_uses_the_box_team_or_an_explicit_prefix(self):
        self.state.select_at(120, 150)
        self.state.set_name("55")
        self.assertEqual(self.state.entries[0]["person"], "red55")
        self.state.set_name("blue_a")
        self.assertEqual(self.state.entries[0]["person"], "blue_a")
        self.assertEqual(self.state.entries[0]["team"], "blue")

    def test_toggling_team_renames_the_person(self):
        self.state.select_at(120, 150)
        self.state.set_name("55")
        self.state.toggle("team")
        self.assertEqual(self.state.entries[0]["person"], "blue55")

    def test_only_one_main_player_per_frame(self):
        self.state.select_at(120, 150)
        self.state.toggle("main")
        self.state.select_at(330, 150)
        self.state.toggle("main")
        self.assertEqual([e["main"] for e in self.state.entries], [False, True, False])

    def test_copy_previous_names_only_nearby_same_team_unnamed_boxes(self):
        filled = self.state.copy_previous()

        self.assertEqual(filled, 2)
        self.assertEqual([e["person"] for e in self.state.entries], ["red55", "blue8", ""])
        self.assertTrue(self.state.entries[1]["main"])


class CheckTests(unittest.TestCase):
    def test_reports_unnamed_duplicate_mismatched_and_overfull_frames(self):
        reds = [entry(f"red{n}", "red", (40 * n, 0, 30, 60)) for n in range(8)]
        found = problems(labels_with({
            0: [entry("", "red", (0, 0, 30, 60)), entry("blue8", "blue", (50, 0, 30, 60)),
                entry("blue8", "blue", (90, 0, 30, 60)), entry("red3", "blue", (130, 0, 30, 60))],
            10: reds,
        }))

        self.assertTrue(any("without a person name" in p for p in found))
        self.assertTrue(any("blue8 is used 2 times" in p for p in found))
        self.assertTrue(any("red3 is marked team blue" in p for p in found))
        self.assertTrue(any("8 visible red players" in p for p in found))

    def test_clean_labels_have_no_problems(self):
        self.assertEqual(problems(labels_with({0: [entry("red55", "red", (0, 0, 30, 60))]})), [])


if __name__ == "__main__":
    unittest.main()
