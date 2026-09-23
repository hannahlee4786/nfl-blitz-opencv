import unittest
from pathlib import Path

from src.track_video import build_argument_parser


class BuildArgumentParserTests(unittest.TestCase):
    def test_accepts_custom_video_initialization_and_output_paths(self):
        arguments = build_argument_parser().parse_args(
            [
                "--input",
                "clip.mp4",
                "--initialization",
                "results/initial_homography.npz",
                "--output",
                "results/tracked.mp4",
            ]
        )

        self.assertEqual(arguments.input, Path("clip.mp4"))
        self.assertEqual(
            arguments.initialization,
            Path("results/initial_homography.npz"),
        )
        self.assertEqual(arguments.output, Path("results/tracked.mp4"))


if __name__ == "__main__":
    unittest.main()
