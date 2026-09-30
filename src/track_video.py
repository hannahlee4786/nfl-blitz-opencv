import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.field_model import create_field_grid
from src.preprocessing import preprocess_frame
from src.tracking import (
    build_feature_mask,
    estimate_descriptor_camera_motion,
)
from src.visualization import (
    draw_calibration_points,
    draw_field_grid,
    draw_tracking_status,
)


INPUT_PATH = Path("data/input/gameplay3.mp4")
INITIALIZATION_PATH = Path(
    "data/output/initial_homography.npz"
)
OUTPUT_PATH = Path(
    "data/output/tracked_field.mp4"
)
MINIMUM_FEATURE_INLIERS = 8


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Track a saved field homography through a gameplay video."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Gameplay video to track.",
    )
    parser.add_argument(
        "--initialization",
        type=Path,
        default=INITIALIZATION_PATH,
        help="Homography file created by initialize_homography.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Tracked overlay video to create.",
    )
    return parser


def get_confidence(tracked_count, inlier_count):
    if tracked_count == 0:
        return "LOW"

    inlier_ratio = inlier_count / tracked_count

    if tracked_count >= 40 and inlier_ratio >= 0.75:
        return "HIGH"

    if tracked_count >= 15 and inlier_ratio >= 0.50:
        return "MEDIUM"

    return "LOW"


def write_tracking_diagnostics(output_path, records):
    """Write per-frame tracking quality information for later review."""

    field_names = [
        "frame",
        "features",
        "inliers",
        "mean_alignment_error",
        "status",
    ]

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(records)


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)

    if not args.initialization.exists():
        raise RuntimeError(
            "Run python -m src.initialize_homography first."
        )

    initialization = np.load(args.initialization)

    previous_screen_points = initialization[
        "screen_points"
    ].astype(np.float32)

    current_homography = initialization[
        "homography"
    ].astype(np.float64)

    initial_frame_index = int(
        initialization["frame_index"]
    )

    capture = cv2.VideoCapture(str(args.input))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.input}"
        )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = capture.get(cv2.CAP_PROP_FPS)

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        initial_frame_index,
    )

    success, previous_frame = capture.read()

    if not success:
        raise RuntimeError(
            "Could not read the initialization frame."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output: {args.output}"
        )

    previous_gray, initial_field_mask = preprocess_frame(
        previous_frame,
    )
    previous_feature_mask = build_feature_mask(initial_field_mask)
    frame_index = initial_frame_index
    diagnostics = []

    # Draw the initialization frame.
    output_frame = draw_field_grid(
        previous_frame,
        create_field_grid(),
        current_homography,
    )

    output_frame = draw_calibration_points(
        output_frame,
        previous_screen_points,
    )

    output_frame = draw_tracking_status(
        output_frame,
        frame_index,
        len(previous_screen_points),
        "HIGH",
    )

    writer.write(output_frame)
    diagnostics.append(
        {
            "frame": frame_index,
            "features": len(previous_screen_points),
            "inliers": len(previous_screen_points),
            "mean_alignment_error": 0.0,
            "status": "HIGH",
        }
    )

    while True:
        success, current_frame = capture.read()

        if not success:
            break

        frame_index += 1

        current_gray, current_field_mask = preprocess_frame(
            current_frame,
        )
        current_feature_mask = build_feature_mask(current_field_mask)

        (
            camera_motion,
            feature_count,
            inlier_count,
            alignment_error,
        ) = estimate_descriptor_camera_motion(
            previous_gray,
            current_gray,
            previous_feature_mask,
            current_feature_mask,
        )

        if camera_motion is not None and inlier_count >= MINIMUM_FEATURE_INLIERS:
            current_homography = camera_motion @ current_homography
            current_homography /= current_homography[2, 2]
            previous_gray = current_gray
            previous_feature_mask = current_feature_mask
            status = get_confidence(feature_count, inlier_count)
        else:
            status = "LOW"

        output_frame = draw_field_grid(
            current_frame,
            create_field_grid(),
            current_homography,
        )

        output_frame = draw_tracking_status(
            output_frame,
            frame_index,
            feature_count,
            status,
        )

        writer.write(output_frame)
        diagnostics.append(
            {
                "frame": frame_index,
                "features": feature_count,
                "inliers": inlier_count,
                "mean_alignment_error": (
                    round(alignment_error, 3)
                    if alignment_error is not None
                    else ""
                ),
                "status": status,
            }
        )

        if camera_motion is None or inlier_count < MINIMUM_FEATURE_INLIERS:
            print(
                f"Tracking lost at frame {frame_index}: "
                f"{inlier_count} feature inliers remained."
            )

            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    diagnostics_path = args.output.with_suffix(".csv")
    write_tracking_diagnostics(diagnostics_path, diagnostics)
    print(f"Saved tracked video to: {args.output}")
    print(f"Saved tracking diagnostics to: {diagnostics_path}")


if __name__ == "__main__":
    main()
