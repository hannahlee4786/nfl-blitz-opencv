"""Tracks field points between frames, and matches frames when they move fast.

Ported from src/tracking.py on the camera-tracking branch.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

OPTICAL_FLOW_PARAMETERS = {
    "winSize": (31, 31),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
}

# Features to match between frames when the camera moves too fast for
# optical flow.
MAX_MATCH_FEATURES = 3000

# A match must beat the second-best candidate by this ratio to count.
MATCH_RATIO = 0.75

# Too few agreeing matches to trust the camera motion.
MIN_MATCHES = 30


def track_screen_points(
    previous_gray: NDArray[np.uint8],
    current_gray: NDArray[np.uint8],
    previous_points: NDArray[np.float32],
) -> tuple[NDArray[np.float32] | None, NDArray[np.bool_] | None, NDArray[np.float32] | None]:
    """
    Track screen points into the next frame.

    Uses forward-backward checking to reject unreliable points.
    """

    previous_points = previous_points.reshape(-1, 1, 2).astype(
        np.float32
    )

    current_points, forward_status, forward_errors = (
        cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            **OPTICAL_FLOW_PARAMETERS,
        )
    )

    if current_points is None:
        return None, None, None

    # Track the newly found points backward into the previous frame.
    backward_points, backward_status, backward_errors = (
        cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            current_points,
            None,
            **OPTICAL_FLOW_PARAMETERS,
        )
    )

    if backward_points is None:
        return None, None, None

    forward_status = forward_status.reshape(-1).astype(bool)
    backward_status = backward_status.reshape(-1).astype(bool)

    backward_difference = np.linalg.norm(
        previous_points.reshape(-1, 2)
        - backward_points.reshape(-1, 2),
        axis=1,
    )

    # A point is reliable if:
    # 1. Forward tracking succeeded.
    # 2. Backward tracking succeeded.
    # 3. It returned close to its original position.
    valid = (
        forward_status
        & backward_status
        & (backward_difference < 2.0)
    )

    return (
        current_points.reshape(-1, 2),
        valid,
        backward_difference,
    )


def match_frame_motion(
    previous_gray: NDArray[np.uint8],
    current_gray: NDArray[np.uint8],
    previous_mask: NDArray[np.uint8],
    current_mask: NDArray[np.uint8],
) -> NDArray[np.float64] | None:
    """
    Return the homography that moves the previous frame onto the current
    one, from matched ORB features, or None.

    Optical flow loses most points in fast camera moves, such as a quick
    zoom. Matching features works however far the field moved.
    """

    orb = cv2.ORB_create(MAX_MATCH_FEATURES)

    previous_keypoints, previous_descriptors = orb.detectAndCompute(
        previous_gray,
        previous_mask,
    )

    current_keypoints, current_descriptors = orb.detectAndCompute(
        current_gray,
        current_mask,
    )

    if previous_descriptors is None or current_descriptors is None:
        return None

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        previous_descriptors,
        current_descriptors,
        k=2,
    )

    # Keep matches clearly better than the runner-up. The field repeats
    # itself, so ambiguous matches are common.
    matches = [
        pair[0] for pair in pairs
        if len(pair) == 2
        and pair[0].distance < MATCH_RATIO * pair[1].distance
    ]

    if len(matches) < MIN_MATCHES:
        return None

    previous_points = np.float32(
        [previous_keypoints[match.queryIdx].pt for match in matches]
    )

    current_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in matches]
    )

    motion, inlier_mask = cv2.findHomography(
        previous_points,
        current_points,
        cv2.USAC_MAGSAC,
        3.0,
    )

    if motion is None or inlier_mask.sum() < MIN_MATCHES:
        return None

    return motion
