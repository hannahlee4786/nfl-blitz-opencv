import cv2
import numpy as np
from scipy import ndimage

from nfl_reconstruction.point_tracking import match_frame_motion, track_screen_points


def texture(size=(320, 400), blur=1.5):
    rng = np.random.default_rng(5)
    values = ndimage.gaussian_filter(rng.random(size), blur)
    return (255 * (values - values.min()) / np.ptp(values)).astype(np.uint8)


def test_points_follow_the_picture_into_the_next_frame():
    previous = texture()
    current = np.roll(previous, (-4, 6), axis=(0, 1))
    points = cv2.goodFeaturesToTrack(
        previous[40:-40, 40:-40], maxCorners=100, qualityLevel=0.01, minDistance=8
    ).reshape(-1, 2) + 40

    tracked, valid, _ = track_screen_points(previous, current, points)

    assert valid.mean() > 0.9
    assert np.allclose(tracked[valid] - points[valid], (6, -4), atol=0.1)


def test_frame_matching_recovers_a_large_camera_move():
    previous = texture()
    height, width = previous.shape
    # A quick zoom with a pan, too far for optical flow.
    move = np.array(((1.15, 0.02, -30.0), (-0.01, 1.15, 25.0), (0.0, 0.0002, 1.0)))
    current = cv2.warpPerspective(previous, move, (width, height))
    mask = np.full(previous.shape, 255, dtype=np.uint8)

    motion = match_frame_motion(previous, current, mask, mask)

    probe = np.array(((100.0, 100.0, 1.0), (250.0, 150.0, 1.0), (200.0, 250.0, 1.0)))
    expected = probe @ move.T
    found = probe @ motion.T
    assert np.allclose(found[:, :2] / found[:, 2:], expected[:, :2] / expected[:, 2:], atol=1.0)


def test_frame_matching_gives_up_without_features():
    blank = np.full((120, 160), 90, dtype=np.uint8)
    mask = np.full(blank.shape, 255, dtype=np.uint8)

    assert match_frame_motion(blank, blank, mask, mask) is None
