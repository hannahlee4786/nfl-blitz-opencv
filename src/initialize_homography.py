import argparse
from pathlib import Path

import cv2
import numpy as np

from src.field_model import (
    build_calibration_points,
    create_field_grid,
    get_calibration_field_points,
)
from src.initialization import (
    read_video_frame,
    select_initialization_frame,
    select_screen_points,
)
from src.visualization import (
    draw_calibration_points,
    draw_field_grid,
)


INPUT_PATH = Path("data/input/gameplay3.mp4")

OUTPUT_DIRECTORY = Path("data/output")


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Choose a video frame and initialize its field homography."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Gameplay video to initialize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory for the homography and preview files.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        help="Use this frame directly instead of opening the frame browser.",
    )
    parser.add_argument(
        "--yard-lines",
        type=int,
        nargs=3,
        default=[20, 30, 40],
        metavar=("FIRST", "SECOND", "THIRD"),
        help="Three visible yard-line labels, in click order.",
    )
    parser.add_argument(
        "--field-half",
        choices=("left", "right"),
        default="left",
        help=(
            "Half of the fixed field model containing the selected lines. "
            "Use left before midfield and right after midfield."
        ),
    )
    return parser


def calculate_reprojection_error(
    field_points,
    screen_points,
    homography,
):
    projected_points = cv2.perspectiveTransform(
        field_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected_points - screen_points,
        axis=1,
    )

    return errors


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    homography_path = args.output_dir / "initial_homography.npz"
    preview_path = args.output_dir / "initial_homography_preview.png"

    capture = cv2.VideoCapture(str(args.input))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.input}"
        )

    try:
        if args.frame_index is None:
            frame_index, frame = select_initialization_frame(capture)
        else:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if not 0 <= args.frame_index < frame_count:
                raise ValueError(
                    f"Frame index must be between 0 and {frame_count - 1}."
                )

            frame_index = args.frame_index
            frame = read_video_frame(capture, frame_index)
    finally:
        capture.release()

    calibration_points = build_calibration_points(
        args.yard_lines,
        args.field_half,
    )
    field_points = get_calibration_field_points(calibration_points)

    print()
    print(f"Using frame {frame_index} from {args.input}")
    print("Select these points in the displayed order:")

    for index, point in enumerate(calibration_points):
        print(
            f"{index + 1}. {point['label']} "
            f"-> {point['coordinate']}"
        )

    screen_points = select_screen_points(
        frame,
        calibration_points,
    )

    homography, inlier_mask = cv2.findHomography(
        field_points,
        screen_points,
        cv2.RANSAC,
        3.0,
    )

    if homography is None:
        raise RuntimeError(
            "OpenCV could not calculate a homography. "
            "Check that the selected points are correct "
            "and are not all on one line."
        )

    errors = calculate_reprojection_error(
        field_points,
        screen_points,
        homography,
    )

    inlier_count = int(inlier_mask.sum())
    total_count = len(field_points)

    print()
    print("Homography calculated successfully.")
    print(f"RANSAC inliers: {inlier_count}/{total_count}")
    print(
        "Average reprojection error: "
        f"{errors.mean():.2f} pixels"
    )
    print(
        "Maximum reprojection error: "
        f"{errors.max():.2f} pixels"
    )

    np.savez(
        homography_path,
        homography=homography,
        field_points=field_points,
        screen_points=screen_points,
        inlier_mask=inlier_mask,
        frame_index=frame_index,
        input_path=str(args.input),
        yard_lines=np.asarray(args.yard_lines, dtype=np.int32),
        field_half=args.field_half,
    )

    preview = draw_field_grid(
        frame,
        create_field_grid(),
        homography,
    )

    preview = draw_calibration_points(
        preview,
        screen_points,
    )

    cv2.imwrite(
        str(preview_path),
        preview,
    )

    print(f"Saved homography to: {homography_path}")
    print(f"Saved preview to: {preview_path}")

    # Show the result for inspection.
    display_height, display_width = preview.shape[:2]

    scale = min(
        1.0,
        1100 / display_width,
        800 / display_height,
    )

    displayed_preview = cv2.resize(
        preview,
        (
            int(display_width * scale),
            int(display_height * scale),
        ),
    )

    cv2.imshow(
        "Initial homography preview",
        displayed_preview,
    )

    print("Press any key in the preview window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
