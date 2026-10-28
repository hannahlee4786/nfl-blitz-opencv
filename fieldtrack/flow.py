# Tracks points and updates the homography
import cv2
import numpy as np


OPTICAL_FLOW_PARAMETERS = {
    "winSize": (31, 31),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
}


def track_screen_points(
    previous_gray,
    current_gray,
    previous_points,
):
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