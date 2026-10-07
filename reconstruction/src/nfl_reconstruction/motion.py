"""OpenCV-independent image motion estimation for the virtual camera."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage


def _phase_motion(
    previous: np.ndarray,
    current: np.ndarray,
    excluded: np.ndarray,
) -> tuple[np.ndarray, float]:
    def signal(frame):
        gray = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        edges = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
        edges[excluded] = 0
        window = np.outer(np.hanning(edges.shape[0]), np.hanning(edges.shape[1]))
        return (edges - edges.mean()) * window

    first, second = signal(previous), signal(current)
    cross = np.fft.fft2(first) * np.conj(np.fft.fft2(second))
    cross /= np.maximum(np.abs(cross), 1e-9)
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = np.asarray(peak, dtype=float)
    dimensions = np.asarray(correlation.shape)
    shift[shift > dimensions / 2] -= dimensions[shift > dimensions / 2]
    response = float(correlation[peak] / max(correlation.mean(), 1e-9))
    dx, dy = float(-shift[1]), float(-shift[0])
    return np.array(((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0))), response


def motion_is_sane(matrix: np.ndarray, image_size: tuple[int, int]) -> bool:
    """Reject projective jumps that cannot be adjacent-frame camera motion."""

    width, height = image_size
    corners = np.array(((0, 0), (width, 0), (width, height), (0, height)), float)
    homogeneous = np.column_stack((corners, np.ones(4))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        return False
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    x, y = projected[:, 0], projected[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area_ratio = area / (width * height)
    return bool(
        0.35 <= area_ratio <= 2.5
        and np.max(np.abs(projected[:, 0])) <= 3 * width
        and np.max(np.abs(projected[:, 1])) <= 3 * height
    )


def relative_to_anchor(
    pair_transforms: list[np.ndarray],
    anchor_index: int,
) -> list[np.ndarray]:
    """Compose adjacent transforms into transforms from the anchor frame."""

    cumulative = [np.eye(3)]
    for transform in pair_transforms:
        cumulative.append(transform @ cumulative[-1])
    anchor_inverse = np.linalg.inv(cumulative[anchor_index])
    return [matrix @ anchor_inverse for matrix in cumulative]


def estimate_frame_motion(
    frames: list[np.ndarray],
    excluded: np.ndarray,
    anchor_index: int,
    *,
    num_features: int = 300,
) -> tuple[list[np.ndarray], list[float]]:
    """Estimate full projective adjacent-frame motion with a translation fallback."""

    if not frames:
        return [], []
    if not 0 <= anchor_index < len(frames):
        raise ValueError("Anchor index is outside the prepared frame sequence.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import torch
        from kornia.feature import SIFTFeature, get_laf_center, match_smnn
        from kornia.geometry import RANSAC

    detector = SIFTFeature(num_features=num_features, upright=False)
    robust_fit = RANSAC(
        "homography", inl_th=3.0, max_iter=20, max_lo_iters=5, seed=7
    )
    feature_sets = []
    valid_mask = torch.from_numpy((~excluded)[None, None])
    with torch.inference_mode():
        for frame in frames:
            gray = frame.astype(np.float32) @ np.array(
                [0.299, 0.587, 0.114], dtype=np.float32
            )
            tensor = torch.from_numpy(gray)[None, None] / 255.0
            lafs, _, descriptors = detector(tensor, valid_mask)
            feature_sets.append((get_laf_center(lafs)[0], descriptors[0]))

        pair_transforms = []
        qualities = [1.0]
        image_size = (frames[0].shape[1], frames[0].shape[0])
        for index in range(1, len(frames)):
            previous_points, previous_descriptors = feature_sets[index - 1]
            current_points, current_descriptors = feature_sets[index]
            _, matches = match_smnn(
                previous_descriptors, current_descriptors, th=0.85
            )
            transform = None
            quality = 0.0
            if len(matches) >= 8:
                candidate, inliers = robust_fit(
                    previous_points[matches[:, 0]], current_points[matches[:, 1]]
                )
                candidate_array = candidate.detach().cpu().numpy().astype(float)
                candidate_array /= candidate_array[2, 2]
                inlier_count = int(inliers.sum())
                if inlier_count >= 8 and motion_is_sane(candidate_array, image_size):
                    transform = candidate_array
                    quality = min(1.0, inlier_count / max(20.0, len(matches) * 0.5))
            if transform is None:
                transform, response = _phase_motion(
                    frames[index - 1], frames[index], excluded
                )
                quality = min(0.24, response / 100.0)
            pair_transforms.append(transform)
            qualities.append(quality)

    relative = relative_to_anchor(pair_transforms, anchor_index)
    qualities[anchor_index] = 1.0
    return relative, qualities
