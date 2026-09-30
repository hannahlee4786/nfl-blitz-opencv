import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.track_video import build_argument_parser, write_tracking_diagnostics


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


class TrackingDiagnosticsTests(unittest.TestCase):
    def test_writes_per_frame_tracking_diagnostics_as_csv(self):
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "diagnostics.csv"
            write_tracking_diagnostics(
                output_path,
                [
                    {
                        "frame": 60,
                        "features": 120,
                        "inliers": 110,
                        "mean_alignment_error": 0.4,
                        "status": "HIGH",
                    }
                ],
            )

            self.assertEqual(
                output_path.read_text(),
                "frame,features,inliers,mean_alignment_error,status\n"
                "60,120,110,0.4,HIGH\n",
            )


if __name__ == "__main__":
    unittest.main()
