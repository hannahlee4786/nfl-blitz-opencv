from pathlib import Path

import cv2
import numpy as np

from src.field_model import (
    CALIBRATION_POINTS,
    create_field_grid,
    get_calibration_field_points,
)
from src.initialization import select_screen_points
from src.visualization import (
    draw_calibration_points,
    draw_field_grid,
)


INPUT_PATH = Path("data/input/gameplay3.mp4")

OUTPUT_DIRECTORY = Path("data/output")
HOMOGRAPHY_PATH = OUTPUT_DIRECTORY / "initial_homography.npz"
PREVIEW_PATH = OUTPUT_DIRECTORY / "initial_homography_preview.png"

# Change this if frame zero is blocked by graphics or players.
INITIAL_FRAME_INDEX = 0


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


def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(str(INPUT_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {INPUT_PATH}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        INITIAL_FRAME_INDEX,
    )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {INITIAL_FRAME_INDEX}"
        )

    field_points = get_calibration_field_points()

    print()
    print("Select these points in the displayed order:")

    for index, point in enumerate(CALIBRATION_POINTS):
        print(
            f"{index + 1}. {point['label']} "
            f"-> {point['coordinate']}"
        )

    screen_points = select_screen_points(
        frame,
        CALIBRATION_POINTS,
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
        HOMOGRAPHY_PATH,
        homography=homography,
        field_points=field_points,
        screen_points=screen_points,
        inlier_mask=inlier_mask,
        frame_index=INITIAL_FRAME_INDEX,
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
        str(PREVIEW_PATH),
        preview,
    )

    print(f"Saved homography to: {HOMOGRAPHY_PATH}")
    print(f"Saved preview to: {PREVIEW_PATH}")

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