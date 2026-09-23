import unittest

from src.initialization import frame_offset_for_key, move_frame_index


class MoveFrameIndexTests(unittest.TestCase):
    def test_moves_within_video_bounds(self):
        self.assertEqual(move_frame_index(60, 30, 186), 90)
        self.assertEqual(move_frame_index(60, -30, 186), 30)

    def test_clamps_to_first_and_last_frame(self):
        self.assertEqual(move_frame_index(5, -30, 186), 0)
        self.assertEqual(move_frame_index(180, 30, 186), 185)

    def test_rejects_empty_video(self):
        with self.assertRaises(ValueError):
            move_frame_index(0, 1, 0)


class FrameOffsetForKeyTests(unittest.TestCase):
    def test_maps_navigation_keys_to_frame_offsets(self):
        self.assertEqual(frame_offset_for_key(ord("a"), 30.0), -1)
        self.assertEqual(frame_offset_for_key(ord("d"), 30.0), 1)
        self.assertEqual(frame_offset_for_key(ord("["), 30.0), -30)
        self.assertEqual(frame_offset_for_key(ord("]"), 30.0), 30)

    def test_ignores_non_navigation_keys(self):
        self.assertIsNone(frame_offset_for_key(ord("x"), 30.0))


if __name__ == "__main__":
    unittest.main()
