import argparse

import cv2
import numpy as np

from src.field_model import create_field_grid
from src.preprocessing import preprocess_frame
from src.tracking import track_screen_points
from src.video_paths import (
    OUTPUT_DIR,
    calibration_path,
    list_input_videos,
    resolve_video,
)
from src.visualization import (
    draw_field_grid,
    draw_tracked_points,
    draw_tracking_status,
)

# The clicked calibration points only set the starting homography.
# Tracking uses many automatic points on the field instead, so losing
# some to players or the screen edge does not matter.
MAX_FEATURES = 400

# Add new points when fewer than this many are still tracked.
MIN_FEATURES = 200

# Minimum distance, in pixels, between two tracked points.
FEATURE_SPACING = 12

# How far, in pixels, a point may be from where the homography places
# it before RANSAC treats it as an outlier.
RANSAC_THRESHOLD = 3.0


def get_confidence(tracked_count, inlier_count):
    if tracked_count == 0:
        return "LOW"

    inlier_ratio = inlier_count / tracked_count

    if inlier_count >= 100 and inlier_ratio >= 0.70:
        return "HIGH"

    if inlier_count >= 30 and inlier_ratio >= 0.50:
        return "MEDIUM"

    return "LOW"


def create_feature_mask(frame, existing_points):
    """Return the pixels where new feature points may be placed."""

    _, field_mask = preprocess_frame(frame)

    # Shrink the field so points are not placed on player outlines or
    # HUD edges, which do not move with the field.
    mask = cv2.erode(
        field_mask,
        np.ones((9, 9), np.uint8),
    )

    # Keep new points away from points that are already tracked.
    for x, y in existing_points:
        cv2.circle(
            mask,
            (int(round(x)), int(round(y))),
            FEATURE_SPACING,
            0,
            -1,
        )

    return mask


def detect_new_points(gray, mask, homography, count):
    """
    Find new feature points and record where each one is on the field.

    Returns (screen_points, field_points).
    """

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=count,
        qualityLevel=0.01,
        minDistance=FEATURE_SPACING,
        mask=mask,
    )

    if corners is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    screen_points = corners.reshape(-1, 2).astype(np.float32)

    # Each point stays tied to the same spot on the field. Look up that
    # spot with the current homography, which maps field to screen.
    field_points = cv2.perspectiveTransform(
        screen_points.reshape(-1, 1, 2),
        np.linalg.inv(homography),
    ).reshape(-1, 2)

    return screen_points, field_points.astype(np.float32)


def draw_output_frame(
    frame,
    homography,
    screen_points,
    frame_index,
    point_count,
    confidence,
):
    output_frame = draw_field_grid(
        frame,
        create_field_grid(),
        homography,
    )

    output_frame = draw_tracked_points(
        output_frame,
        screen_points,
    )

    return draw_tracking_status(
        output_frame,
        frame_index,
        point_count,
        confidence,
    )


def track_video(input_path, initialization_path, output_path):
    initialization = np.load(initialization_path)

    current_homography = initialization[
        "homography"
    ].astype(np.float64)

    initial_frame_index = int(
        initialization["frame_index"]
    )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
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
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output: {output_path}"
        )

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY,
    )

    screen_points, field_points = detect_new_points(
        previous_gray,
        create_feature_mask(previous_frame, []),
        current_homography,
        MAX_FEATURES,
    )

    frame_index = initial_frame_index
    low_confidence_frames = 0

    writer.write(
        draw_output_frame(
            previous_frame,
            current_homography,
            screen_points,
            frame_index,
            len(screen_points),
            "HIGH",
        )
    )

    while True:
        success, current_frame = capture.read()

        if not success:
            break

        frame_index += 1

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        valid_count = 0
        inlier_count = 0

        if len(screen_points) > 0:
            (
                candidate_screen_points,
                valid_mask,
                tracking_errors,
            ) = track_screen_points(
                previous_gray,
                current_gray,
                screen_points,
            )

            if candidate_screen_points is not None:
                valid_count = int(valid_mask.sum())

        if valid_count >= 4:
            valid_screen_points = (
                candidate_screen_points[valid_mask]
            ).astype(np.float32)

            valid_field_points = field_points[valid_mask]

            new_homography, inlier_mask = cv2.findHomography(
                valid_field_points,
                valid_screen_points,
                cv2.RANSAC,
                RANSAC_THRESHOLD,
            )

            if new_homography is not None:
                current_homography = new_homography
                inliers = inlier_mask.reshape(-1).astype(bool)
                inlier_count = int(inliers.sum())

                # Drop points that no longer agree with the field,
                # such as ones dragged along by a player.
                screen_points = valid_screen_points[inliers]
                field_points = valid_field_points[inliers]
            else:
                valid_count = 0

        if valid_count < 4:
            # Keep the last homography and start over with fresh points.
            screen_points = np.empty((0, 2), dtype=np.float32)
            field_points = np.empty((0, 2), dtype=np.float32)

        # Top up with new points as old ones are lost.
        if len(screen_points) < MIN_FEATURES:
            new_screen_points, new_field_points = detect_new_points(
                current_gray,
                create_feature_mask(current_frame, screen_points),
                current_homography,
                MAX_FEATURES - len(screen_points),
            )

            screen_points = np.vstack(
                [screen_points, new_screen_points]
            )

            field_points = np.vstack(
                [field_points, new_field_points]
            )

        confidence = get_confidence(
            valid_count,
            inlier_count,
        )

        if confidence == "LOW":
            low_confidence_frames += 1

        writer.write(
            draw_output_frame(
                current_frame,
                current_homography,
                screen_points,
                frame_index,
                inlier_count,
                confidence,
            )
        )

        previous_gray = current_gray

    capture.release()
    writer.release()

    print(f"Frames with LOW confidence: {low_confidence_frames}")
    print(f"Saved tracked video to: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Track the field grid through calibrated videos.",
    )

    parser.add_argument(
        "videos",
        nargs="*",
        help="Video paths, or the start of file names in data/input. "
        "Leave empty to track every video that has a calibration.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.videos:
        input_paths = [
            resolve_video(video) for video in arguments.videos
        ]
    else:
        input_paths = [
            video for video in list_input_videos()
            if calibration_path(video).exists()
        ]

    if not input_paths:
        raise RuntimeError(
            "No calibrated videos. "
            "Run python -m src.initialize_homography first."
        )

    for input_path in input_paths:
        initialization_path = calibration_path(input_path)

        if not initialization_path.exists():
            print(
                f"Skipping {input_path.name}: no calibration. Run "
                f'python -m src.initialize_homography "{input_path.stem}"'
            )
            continue

        print(f"Tracking {input_path.name}...")

        track_video(
            input_path,
            initialization_path,
            OUTPUT_DIR / f"{input_path.stem}_tracked.mp4",
        )


if __name__ == "__main__":
    main()
