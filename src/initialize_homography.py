import argparse

import cv2
import numpy as np

from src.field_model import (
    create_calibration_points,
    create_field_grid,
    get_calibration_field_points,
)
from src.initialization import select_screen_points
from src.video_paths import (
    OUTPUT_DIR,
    calibration_path,
    resolve_video,
)
from src.visualization import (
    draw_calibration_points,
    draw_field_grid,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Click field landmarks to calibrate one video.",
    )

    parser.add_argument(
        "video",
        nargs="?",
        default="Clip_3",
        help="Video path, or the start of a file name in data/input "
        "(for example Clip_2).",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame to click on. Pick one where the hash marks are "
        "not covered by graphics or players.",
    )

    parser.add_argument(
        "--yard-lines",
        type=int,
        nargs=3,
        default=[20, 30, 40],
        metavar="YARDS",
        help="The three painted yard numbers to click.",
    )

    parser.add_argument(
        "--far-half",
        action="store_true",
        help="The yard lines are on the half away from the x = 0 "
        "(COWBOYS) end zone.",
    )

    parser.add_argument(
        "--reuse-clicks",
        action="store_true",
        help="Recalculate from the points clicked last time, on the same "
        "frame, instead of clicking again. Run this after the field "
        "model changes.",
    )

    return parser.parse_args()


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


def load_previous_clicks(homography_path, field_points):
    """
    Return the screen points and frame index saved by the last
    calibration of this video, so it can be redone without clicking.
    """

    if not homography_path.exists():
        raise RuntimeError(
            f"No earlier calibration to reuse: {homography_path}"
        )

    previous = np.load(homography_path)
    previous_field_points = previous["field_points"]

    # A yard line's x never changes, so different x values mean the
    # earlier calibration clicked different yard lines.
    if (
        previous_field_points.shape != field_points.shape
        or not np.allclose(
            previous_field_points[:, 0],
            field_points[:, 0],
        )
    ):
        raise RuntimeError(
            "The earlier calibration used different yard lines. "
            "Pass the same --yard-lines and --far-half as before."
        )

    return (
        previous["screen_points"].astype(np.float32),
        int(previous["frame_index"]),
    )


def main():
    arguments = parse_arguments()

    input_path = resolve_video(arguments.video)
    homography_path = calibration_path(input_path)
    preview_path = (
        OUTPUT_DIR / f"{input_path.stem}_homography_preview.png"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_points = create_calibration_points(
        arguments.yard_lines,
        arguments.far_half,
    )

    field_points = get_calibration_field_points(calibration_points)

    if arguments.reuse_clicks:
        screen_points, frame_index = load_previous_clicks(
            homography_path,
            field_points,
        )
    else:
        frame_index = arguments.frame

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_index}"
        )

    print()

    if arguments.reuse_clicks:
        print(
            f"Recalibrating {input_path.name} from the points "
            f"clicked on frame {frame_index}"
        )
    else:
        print(f"Calibrating {input_path.name}, frame {frame_index}")
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

    # The painted numbers are wide targets, so careful clicks can still be
    # a few pixels off center. At 3 px RANSAC drops some of them and fits
    # the rest exactly; 8 px keeps them and still rejects a wrong click.
    homography, inlier_mask = cv2.findHomography(
        field_points,
        screen_points,
        cv2.RANSAC,
        8.0,
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

    # Nothing was clicked, so check the saved preview instead of
    # waiting on a window.
    if arguments.reuse_clicks:
        return

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