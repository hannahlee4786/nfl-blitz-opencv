"""Follows the field grid through a clip, snapping it onto the painted lines.

Ported from the tracking loop in src/track_video.py on the camera-tracking
branch. Each frame, points tracked on the grass predict where the grid
moved, and the grid is then snapped onto the painted yard lines so errors
cannot build up. When the camera moves too fast for the tracked points,
features matched between the two frames give the prediction instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .line_score import (
    create_distance_map,
    find_painted_lines,
    get_tracking_status,
    measure_line_error,
)
from .line_snapping import snap_to_lines
from .point_tracking import match_frame_motion, track_screen_points
from .preprocessing import find_field_mask

# The clicked anchor points only set the starting homography. Tracking
# uses many automatic points on the field instead, so losing some to
# players or the screen edge does not matter.
MAX_FEATURES = 400

# Add new points when fewer than this many are still tracked.
MIN_FEATURES = 200

# Minimum distance, in pixels, between two tracked points.
FEATURE_SPACING = 12

# How far, in pixels, a point may be from where the homography places
# it before RANSAC treats it as an outlier.
RANSAC_THRESHOLD = 3.0


@dataclass(frozen=True)
class PaintedLines:
    """What one frame shows of the field: the grass and the paint on it."""

    field_mask: NDArray[np.uint8]
    painted: NDArray[np.uint8]
    distance_map: NDArray[np.float32]


@dataclass(frozen=True)
class GridTrack:
    """The tracked field-to-screen homography of every frame."""

    homographies: NDArray[np.float64]
    line_errors: NDArray[np.float64]
    statuses: NDArray[np.str_]
    snapped: NDArray[np.bool_]


def find_frame_lines(
    frame: NDArray[np.uint8],
    excluded: NDArray[np.bool_],
) -> PaintedLines:
    field_mask = find_field_mask(frame, excluded)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    painted = find_painted_lines(gray, field_mask)

    return PaintedLines(field_mask, painted, create_distance_map(painted))


def create_feature_mask(
    field_mask: NDArray[np.uint8],
    existing_points: NDArray[np.float32] | Sequence[tuple[float, float]],
) -> NDArray[np.uint8]:
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


def screen_to_field(
    screen_points: NDArray[np.float32],
    homography: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Return where each screen point is on the field."""

    if len(screen_points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    field_points = cv2.perspectiveTransform(
        screen_points.reshape(-1, 1, 2).astype(np.float32),
        np.linalg.inv(homography),
    )

    return field_points.reshape(-1, 2).astype(np.float32)


def detect_new_points(
    gray: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    homography: NDArray[np.float64],
    count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
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


def follow_grid(
    frames: Sequence[NDArray[np.uint8]],
    excluded: NDArray[np.bool_],
    homography: NDArray[np.float64],
) -> GridTrack:
    """
    Track the grid through `frames`, starting from `homography` on the
    first frame.
    """

    count = len(frames)
    homographies = np.zeros((count, 3, 3))
    line_errors = np.full(count, np.nan)
    statuses = np.full(count, "LOST", dtype="<U4")
    snapped = np.zeros(count, dtype=bool)

    previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    lines = find_frame_lines(frames[0], excluded)

    # The clicks only pin down the grid near the clicked points, so snap
    # the anchor onto the painted lines as well.
    current_homography = np.asarray(homography, dtype=np.float64)
    snapped_homography = snap_to_lines(
        current_homography,
        lines.painted,
        lines.distance_map,
        lines.field_mask,
    )

    snapped[0] = snapped_homography is not None

    if snapped[0]:
        current_homography = snapped_homography

    screen_points, field_points = detect_new_points(
        previous_gray,
        create_feature_mask(lines.field_mask, []),
        current_homography,
        MAX_FEATURES,
    )

    homographies[0] = current_homography
    line_errors[0] = measure_line_error(
        lines.distance_map,
        lines.field_mask,
        current_homography,
    )
    statuses[0] = get_tracking_status(line_errors[0])

    for index in range(1, count):
        previous_homography = current_homography
        previous_field_mask = lines.field_mask

        current_gray = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        lines = find_frame_lines(frames[index], excluded)

        valid_count = 0

        if len(screen_points) > 0:
            (
                candidate_screen_points,
                valid_mask,
                _,
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

        # The tracked points only predict the grid. Correct it against
        # the painted yard lines so errors cannot build up.
        snapped_homography = snap_to_lines(
            current_homography,
            lines.painted,
            lines.distance_map,
            lines.field_mask,
        )

        if snapped_homography is None:
            # Fast camera moves lose most tracked points, so that
            # prediction can be far off. Predict again from features
            # matched between the two frames, and snap that instead.
            motion = match_frame_motion(
                previous_gray,
                current_gray,
                create_feature_mask(previous_field_mask, []),
                create_feature_mask(lines.field_mask, []),
            )

            if motion is not None:
                snapped_homography = snap_to_lines(
                    motion @ previous_homography,
                    lines.painted,
                    lines.distance_map,
                    lines.field_mask,
                )

        snapped[index] = snapped_homography is not None

        if snapped[index]:
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
                create_feature_mask(lines.field_mask, screen_points),
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
        homographies[index] = current_homography
        line_errors[index] = measure_line_error(
            lines.distance_map,
            lines.field_mask,
            current_homography,
        )
        statuses[index] = get_tracking_status(line_errors[index])

        previous_gray = current_gray

    return GridTrack(homographies, line_errors, statuses, snapped)


def track_grid(
    frames: Sequence[NDArray[np.uint8]],
    excluded: NDArray[np.bool_],
    anchor_index: int,
    anchor_homography: NDArray[np.float64],
) -> GridTrack:
    """
    Track the grid from the anchor frame to both ends of the clip: forward
    through the frames after it, and backward through the ones before.
    """

    if not 0 <= anchor_index < len(frames):
        raise ValueError("Anchor index is outside the prepared frame sequence.")

    forward = follow_grid(frames[anchor_index:], excluded, anchor_homography)

    if anchor_index == 0:
        return forward

    backward = follow_grid(
        frames[anchor_index::-1], excluded, anchor_homography
    )

    # The backward walk starts on the anchor too; keep the forward copy.
    return GridTrack(
        *(
            np.concatenate([before[:0:-1], after])
            for before, after in (
                (backward.homographies, forward.homographies),
                (backward.line_errors, forward.line_errors),
                (backward.statuses, forward.statuses),
                (backward.snapped, forward.snapped),
            )
        )
    )


def measure_line_errors(
    frames: Sequence[NDArray[np.uint8]],
    excluded: NDArray[np.bool_],
    cameras: Sequence[Sequence[NDArray[np.float64] | None]],
) -> NDArray[np.float64]:
    """
    Return the line error of every frame for each list of homographies
    in `cameras`, one row per list. A None homography scores NaN.
    """

    errors = np.full((len(cameras), len(frames)), np.nan)

    for index, frame in enumerate(frames):
        lines = find_frame_lines(frame, excluded)

        for row, homographies in enumerate(cameras):
            if homographies[index] is not None:
                errors[row, index] = measure_line_error(
                    lines.distance_map,
                    lines.field_mask,
                    homographies[index],
                )

    return errors
