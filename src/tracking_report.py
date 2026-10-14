# Summarizes how well the tracked grid stays on the painted lines
import argparse

import numpy as np

from src.line_score import describe_line_errors
from src.video_paths import OUTPUT_DIR

# Frames per block in the timeline.
BLOCK_SIZE = 30


def describe_timeline(frame_indices, line_errors):
    """Return the median line error of each block of frames."""

    parts = []

    for start in range(0, len(line_errors), BLOCK_SIZE):
        block = line_errors[start:start + BLOCK_SIZE]
        measured = block[~np.isnan(block)]

        if len(measured) == 0:
            value = "--"
        else:
            value = f"{np.median(measured):.1f}"

        parts.append(f"{frame_indices[start]}: {value}")

    return "  ".join(parts)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Summarize the line error of tracked videos.",
    )

    parser.add_argument(
        "videos",
        nargs="*",
        help="The start of video file names (for example Clip_4). "
        "Leave empty to report every tracked video.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    paths = sorted(OUTPUT_DIR.glob("*_tracking.npz"))

    if arguments.videos:
        paths = [
            path for path in paths
            if any(path.name.startswith(name) for name in arguments.videos)
        ]

    if not paths:
        raise RuntimeError(
            "No tracked videos. Run python -m src.track_video first."
        )

    for path in paths:
        tracking = np.load(path)
        line_errors = tracking["line_error"]

        print(path.name.removesuffix("_tracking.npz"))
        print(
            f"  {len(line_errors)} frames. "
            + describe_line_errors(line_errors, tracking["status"])
        )

        # Files tracked before snapping existed have no snapped column.
        if "snapped" in tracking.files:
            snapped_percent = 100.0 * tracking["snapped"].mean()
            print(
                "  Snapped to the painted lines on "
                f"{snapped_percent:.0f}% of frames."
            )
        print(f"  Median line error (px) per {BLOCK_SIZE} frames:")
        print(
            "  "
            + describe_timeline(tracking["frame_index"], line_errors)
        )


if __name__ == "__main__":
    main()
