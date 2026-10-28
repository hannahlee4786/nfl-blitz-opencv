"""Field tracker cherry-picked from the player-tracking branch.

Starts from a clicked calibration (fieldtrack.calibrate) and follows the
field with hundreds of automatic corner points, refitting the homography
with RANSAC on every frame. Yields one FieldState per frame, starting at
the calibration frame.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from fieldtrack.flow import track_screen_points
from fieldtrack.preprocessing import preprocess_frame

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


@dataclass
class FieldState:
    """Everything the field tracker knows about one frame."""

    frame_index: int
    frame: np.ndarray
    homography: np.ndarray  # field -> screen
    screen_points: np.ndarray
    point_count: int
    confidence: str
    fps: float
    size: tuple  # (width, height)


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


def track_field(input_path, initialization_path):
    """
    Generator that runs the field tracker and yields one FieldState per
    frame, starting at the calibration frame. Other stages (such as
    player tracking) consume this instead of re-running field tracking.
    """

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

    try:
        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = capture.get(cv2.CAP_PROP_FPS)
        size = (width, height)

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            initial_frame_index,
        )

        success, previous_frame = capture.read()

        if not success:
            raise RuntimeError(
                "Could not read the initialization frame."
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

        yield FieldState(
            frame_index,
            previous_frame,
            current_homography,
            screen_points,
            len(screen_points),
            "HIGH",
            fps,
            size,
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

            yield FieldState(
                frame_index,
                current_frame,
                current_homography,
                screen_points,
                inlier_count,
                confidence,
                fps,
                size,
            )

            previous_gray = current_gray
    finally:
        capture.release()
