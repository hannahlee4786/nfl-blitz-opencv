import unittest
from pathlib import Path

from src.initialize_homography import build_argument_parser


class BuildArgumentParserTests(unittest.TestCase):
    def test_accepts_video_frame_and_visible_yard_lines(self):
        arguments = build_argument_parser().parse_args(
            [
                "--input",
                "clip.mp4",
                "--output-dir",
                "results",
                "--frame-index",
                "60",
                "--yard-lines",
                "30",
                "40",
                "50",
                "--field-half",
                "right",
            ]
        )

        self.assertEqual(arguments.input, Path("clip.mp4"))
        self.assertEqual(arguments.output_dir, Path("results"))
        self.assertEqual(arguments.frame_index, 60)
        self.assertEqual(arguments.yard_lines, [30, 40, 50])
        self.assertEqual(arguments.field_half, "right")


if __name__ == "__main__":
    unittest.main()
