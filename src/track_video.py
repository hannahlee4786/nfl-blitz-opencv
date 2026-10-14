import argparse

import cv2
import numpy as np

from src.field_model import create_field_grid
from src.line_score import (
    create_distance_map,
    describe_line_errors,
    find_painted_lines,
    get_tracking_status,
    measure_line_error,
)
from src.line_snapping import snap_to_lines
from src.preprocessing import find_active_area, preprocess_frame
from src.tracking import match_frame_motion, track_screen_points
from src.video_paths import (
    OUTPUT_DIR,
    calibration_path,
    tracking_path,
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

# Captures repeat some frames exactly. A frame whose mean difference
# from the previous one is below this is marked as a duplicate.
DUPLICATE_DIFFERENCE = 0.5


def create_feature_mask(field_mask, existing_points):
    """Return the pixels where new feature points may be placed."""

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
    return screen_points, screen_to_field(screen_points, homography)


def screen_to_field(screen_points, homography):
    """Return where each screen point is on the field."""

    if len(screen_points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    field_points = cv2.perspectiveTransform(
        screen_points.reshape(-1, 1, 2).astype(np.float32),
        np.linalg.inv(homography),
    )

    return field_points.reshape(-1, 2).astype(np.float32)


def draw_output_frame(
    frame,
    homography,
    screen_points,
    frame_index,
    point_count,
    line_error,
    status,
    snapped,
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
        line_error,
        status,
        snapped,
    )


def save_tracking(path, frames, active_area):
    """Save one entry per frame for the later camera and player steps."""

    np.savez(
        path,
        active_area=np.array(active_area),
        **{
            name: np.array([frame[name] for frame in frames])
            for name in frames[0]
        },
    )


def track_video(
    input_path,
    initialization_path,
    output_path,
    camera_output_path,
):
    initialization = np.load(initialization_path)

    current_homography = initialization[
        "homography"
    ].astype(np.float64)

    initial_frame_index = int(
        initialization["frame_index"]
    )

    active_area = find_active_area(input_path)

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

    _, field_mask = preprocess_frame(previous_frame, active_area)

    painted = find_painted_lines(previous_gray, field_mask)
    distance_map = create_distance_map(painted)

    # The clicks only pin down the grid near the clicked numbers, so snap
    # the calibration onto the painted lines as well.
    snapped_homography = snap_to_lines(
        current_homography,
        painted,
        distance_map,
        field_mask,
    )

    snapped = snapped_homography is not None

    if snapped:
        current_homography = snapped_homography

    screen_points, field_points = detect_new_points(
        previous_gray,
        create_feature_mask(field_mask, []),
        current_homography,
        MAX_FEATURES,
    )

    frame_index = initial_frame_index

    line_error = measure_line_error(
        distance_map,
        field_mask,
        current_homography,
    )

    status = get_tracking_status(line_error)

    frames = [
        {
            "frame_index": frame_index,
            "homography": current_homography,
            "line_error": line_error,
            "tracked_points": len(screen_points),
            "status": status,
            "snapped": snapped,
            "duplicate": False,
        }
    ]

    writer.write(
        draw_output_frame(
            previous_frame,
            current_homography,
            screen_points,
            frame_index,
            len(screen_points),
            line_error,
            status,
            snapped,
        )
    )

    while True:
        success, current_frame = capture.read()

        if not success:
            break

        frame_index += 1

        previous_homography = current_homography
        previous_field_mask = field_mask

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        _, field_mask = preprocess_frame(current_frame, active_area)

        duplicate = bool(
            cv2.absdiff(current_gray, previous_gray).mean()
            < DUPLICATE_DIFFERENCE
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

        painted = find_painted_lines(current_gray, field_mask)
        distance_map = create_distance_map(painted)

        # The tracked points only predict the grid. Correct it against
        # the painted yard lines so errors cannot build up.
        snapped_homography = snap_to_lines(
            current_homography,
            painted,
            distance_map,
            field_mask,
        )

        if snapped_homography is None:
            # Fast camera moves lose most tracked points, so that
            # prediction can be far off. Predict again from features
            # matched between the two frames, and snap that instead.
            motion = match_frame_motion(
                previous_gray,
                current_gray,
                create_feature_mask(previous_field_mask, []),
                create_feature_mask(field_mask, []),
            )

            if motion is not None:
                snapped_homography = snap_to_lines(
                    motion @ previous_homography,
                    painted,
                    distance_map,
                    field_mask,
                )

        snapped = snapped_homography is not None

        if snapped:
            current_homography = snapped_homography

            # Tie the tracked points to the corrected grid, so the next
            # prediction starts from it.
            field_points = screen_to_field(
                screen_points,
                current_homography,
            )

        # Top up with new points as old ones are lost.
        if len(screen_points) < MIN_FEATURES:
            new_screen_points, new_field_points = detect_new_points(
                current_gray,
                create_feature_mask(field_mask, screen_points),
                current_homography,
                MAX_FEATURES - len(screen_points),
            )

            screen_points = np.vstack(
                [screen_points, new_screen_points]
            )

            field_points = np.vstack(
                [field_points, new_field_points]
            )

        # Score the grid against the painted lines, not against the
        # tracked points, which drift along with the homography.
        line_error = measure_line_error(
            distance_map,
            field_mask,
            current_homography,
        )

        status = get_tracking_status(line_error)

        frames.append(
            {
                "frame_index": frame_index,
                "homography": current_homography,
                "line_error": line_error,
                "tracked_points": inlier_count,
                "status": status,
                "snapped": snapped,
                "duplicate": duplicate,
            }
        )

        writer.write(
            draw_output_frame(
                current_frame,
                current_homography,
                screen_points,
                frame_index,
                inlier_count,
                line_error,
                status,
                snapped,
            )
        )

        previous_gray = current_gray

    capture.release()
    writer.release()

    save_tracking(camera_output_path, frames, active_area)

    print(
        describe_line_errors(
            [frame["line_error"] for frame in frames],
            [frame["status"] for frame in frames],
        )
    )

    snapped_percent = 100.0 * np.mean(
        [frame["snapped"] for frame in frames]
    )

    print(
        "Snapped to the painted lines on "
        f"{snapped_percent:.0f}% of frames"
    )
    print(f"Saved tracked video to: {output_path}")
    print(f"Saved per-frame tracking to: {camera_output_path}")


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
            tracking_path(input_path),
        )


if __name__ == "__main__":
    main()
