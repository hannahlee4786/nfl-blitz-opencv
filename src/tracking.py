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
DESCRIPTOR_SCALE = 0.5


def build_feature_mask(field_mask, margin=12):
    """Remove field-mask boundaries where players and UI create false corners."""

    kernel_size = margin * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(field_mask, kernel)


def filter_features_by_mask(features, feature_mask):
    """Keep only feature centers that remain inside the safe field area."""

    if len(features) == 0:
        return np.empty((0, 2), dtype=np.float32)

    height, width = feature_mask.shape[:2]
    rounded_features = np.rint(features).astype(int)
    valid = (
        (rounded_features[:, 0] >= 0)
        & (rounded_features[:, 0] < width)
        & (rounded_features[:, 1] >= 0)
        & (rounded_features[:, 1] < height)
    )
    valid_indices = np.where(valid)[0]
    valid[valid_indices] = (
        feature_mask[
            rounded_features[valid_indices, 1],
            rounded_features[valid_indices, 0],
        ]
        > 0
    )

    return features[valid].astype(np.float32)


def detect_field_features(gray_frame, field_mask, max_features=160):
    """Find trackable corners inside the current field region."""

    features = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=10,
        mask=field_mask,
        blockSize=7,
        useHarrisDetector=False,
    )

    if features is None:
        return np.empty((0, 2), dtype=np.float32)

    return features.reshape(-1, 2).astype(np.float32)


def replenish_features(
    existing_features,
    candidate_features,
    max_features=160,
    minimum_distance=10.0,
):
    """Add well-separated candidates until the feature budget is full."""

    selected = [point.astype(np.float32) for point in existing_features]

    for candidate in candidate_features:
        if len(selected) >= max_features:
            break

        if all(
            np.linalg.norm(candidate - selected_point) >= minimum_distance
            for selected_point in selected
        ):
            selected.append(candidate.astype(np.float32))

    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def estimate_descriptor_camera_motion(
    previous_gray,
    current_gray,
    previous_feature_mask,
    current_feature_mask,
):
    """Match distinctive field patches to estimate larger camera movement."""

    # Match at half resolution: SIFT remains selective enough for repeated
    # field markings, while the tracker can still review a whole short clip.
    previous_gray = cv2.resize(
        previous_gray,
        None,
        fx=DESCRIPTOR_SCALE,
        fy=DESCRIPTOR_SCALE,
    )
    current_gray = cv2.resize(
        current_gray,
        None,
        fx=DESCRIPTOR_SCALE,
        fy=DESCRIPTOR_SCALE,
    )
    previous_feature_mask = cv2.resize(
        previous_feature_mask,
        None,
        fx=DESCRIPTOR_SCALE,
        fy=DESCRIPTOR_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )
    current_feature_mask = cv2.resize(
        current_feature_mask,
        None,
        fx=DESCRIPTOR_SCALE,
        fy=DESCRIPTOR_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    detector = cv2.SIFT_create(nfeatures=200)
    previous_keypoints, previous_descriptors = detector.detectAndCompute(
        previous_gray,
        previous_feature_mask,
    )
    current_keypoints, current_descriptors = detector.detectAndCompute(
        current_gray,
        current_feature_mask,
    )

    if previous_descriptors is None or current_descriptors is None:
        return None, 0, 0, None

    matches = cv2.BFMatcher().knnMatch(
        previous_descriptors,
        current_descriptors,
        k=2,
    )
    good_matches = [
        first
        for first, second in matches
        if first.distance < 0.7 * second.distance
    ]

    if len(good_matches) < 4:
        return None, len(good_matches), 0, None

    previous_points = np.float32(
        [previous_keypoints[match.queryIdx].pt for match in good_matches]
    )
    current_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in good_matches]
    )
    motion, inlier_mask = cv2.findHomography(
        previous_points,
        current_points,
        cv2.USAC_MAGSAC,
        3.0,
    )

    if motion is None or inlier_mask is None:
        return None, len(good_matches), 0, None

    inliers = inlier_mask.reshape(-1).astype(bool)
    projected_points = cv2.perspectiveTransform(
        previous_points.reshape(-1, 1, 2),
        motion,
    ).reshape(-1, 2)
    alignment_error = float(
        np.linalg.norm(
            projected_points[inliers] - current_points[inliers],
            axis=1,
        ).mean()
    )
    alignment_error /= DESCRIPTOR_SCALE

    scale_matrix = np.array(
        [
            [DESCRIPTOR_SCALE, 0.0, 0.0],
            [0.0, DESCRIPTOR_SCALE, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    full_resolution_motion = np.linalg.inv(scale_matrix) @ motion @ scale_matrix

    return (
        full_resolution_motion / full_resolution_motion[2, 2],
        len(good_matches),
        int(inliers.sum()),
        alignment_error,
    )


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


def estimate_camera_motion(
    previous_gray,
    current_gray,
    previous_features,
    previous_feature_mask=None,
    current_feature_mask=None,
):
    """Estimate a screen-to-screen homography from stable field features."""

    if previous_feature_mask is not None:
        previous_features = filter_features_by_mask(
            previous_features,
            previous_feature_mask,
        )

    current_points, valid_mask, tracking_errors = track_screen_points(
        previous_gray,
        current_gray,
        previous_features,
    )

    if current_points is None:
        return (
            None,
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.float32),
        )

    previous_valid = previous_features[valid_mask].astype(np.float32)
    current_valid = current_points[valid_mask].astype(np.float32)
    valid_errors = tracking_errors[valid_mask]

    if current_feature_mask is not None:
        current_in_mask = filter_features_by_mask(
            current_valid,
            current_feature_mask,
        )
        current_in_mask_set = {
            (float(point[0]), float(point[1])) for point in current_in_mask
        }
        current_mask = np.array(
            [
                (float(point[0]), float(point[1])) in current_in_mask_set
                for point in current_valid
            ],
            dtype=bool,
        )
        previous_valid = previous_valid[current_mask]
        current_valid = current_valid[current_mask]
        valid_errors = valid_errors[current_mask]

    if len(current_valid) < 4:
        return (
            None,
            current_valid,
            np.zeros(len(current_valid), dtype=bool),
            valid_errors,
        )

    motion, inlier_mask = cv2.findHomography(
        previous_valid,
        current_valid,
        cv2.USAC_MAGSAC,
        2.0,
    )

    if motion is None or inlier_mask is None:
        return (
            None,
            current_valid,
            np.zeros(len(current_valid), dtype=bool),
            valid_errors,
        )

    return (
        motion / motion[2, 2],
        current_valid,
        inlier_mask.reshape(-1).astype(bool),
        valid_errors,
    )
