from pathlib import Path

import cv2
import numpy as np

from src.field_model import create_field_grid
from src.tracking import track_screen_points
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


def get_confidence(tracked_count, inlier_count):
    if tracked_count == 0:
        return "LOW"

    inlier_ratio = inlier_count / tracked_count

    if tracked_count >= 6 and inlier_ratio >= 0.75:
        return "HIGH"

    if tracked_count >= 4 and inlier_ratio >= 0.50:
        return "MEDIUM"

    return "LOW"


def main():
    if not INITIALIZATION_PATH.exists():
        raise RuntimeError(
            "Run python -m src.initialize_homography first."
        )

    initialization = np.load(INITIALIZATION_PATH)

    fixed_field_points = initialization[
        "field_points"
    ].astype(np.float32)

    previous_screen_points = initialization[
        "screen_points"
    ].astype(np.float32)

    current_homography = initialization[
        "homography"
    ].astype(np.float64)

    initial_frame_index = int(
        initialization["frame_index"]
    )

    capture = cv2.VideoCapture(str(INPUT_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {INPUT_PATH}"
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

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output: {OUTPUT_PATH}"
        )

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY,
    )

    active_field_points = fixed_field_points.copy()
    frame_index = initial_frame_index

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

    while True:
        success, current_frame = capture.read()

        if not success:
            break

        frame_index += 1

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        (
            candidate_screen_points,
            valid_mask,
            tracking_errors,
        ) = track_screen_points(
            previous_gray,
            current_gray,
            previous_screen_points,
        )

        if candidate_screen_points is None:
            valid_count = 0
        else:
            valid_count = int(valid_mask.sum())

        inlier_count = 0

        if valid_count >= 4:
            valid_screen_points = (
                candidate_screen_points[valid_mask]
            ).astype(np.float32)

            valid_field_points = (
                active_field_points[valid_mask]
            ).astype(np.float32)

            new_homography, inlier_mask = (
                cv2.findHomography(
                    valid_field_points,
                    valid_screen_points,
                    cv2.RANSAC,
                    3.0,
                )
            )

            if (
                new_homography is not None
                and inlier_mask is not None
            ):
                current_homography = new_homography
                inlier_count = int(inlier_mask.sum())

                # Keep only the points that survived optical flow.
                previous_screen_points = (
                    valid_screen_points
                )

                active_field_points = (
                    valid_field_points
                )

                previous_gray = current_gray
            else:
                valid_count = 0

        confidence = get_confidence(
            valid_count,
            inlier_count,
        )

        output_frame = draw_field_grid(
            current_frame,
            create_field_grid(),
            current_homography,
        )

        if valid_count >= 4:
            output_frame = draw_calibration_points(
                output_frame,
                previous_screen_points,
            )

        output_frame = draw_tracking_status(
            output_frame,
            frame_index,
            valid_count,
            confidence,
        )

        writer.write(output_frame)

        # The first prototype cannot recover after fewer than
        # four calibration points remain.
        if valid_count < 4:
            print(
                f"Tracking lost at frame {frame_index}. "
                "The last valid homography will no longer "
                "be updated."
            )

            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved tracked video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()