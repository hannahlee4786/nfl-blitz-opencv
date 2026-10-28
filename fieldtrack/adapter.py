"""Run the player-tracking field tracker and express it in pipeline coordinates.

The tracker works in player-tracking's field frame: x is yards from the back
of the x = 0 end zone (so the goal line is 10), y is yards from the upper
sideline, 0..53.33. The pipeline's world frame is yards from the goal line
(x_world = x_field - 10, so the 50-yard line is 50) with y the same sideline
distance. Both grow downfield-toward-camera and left-to-right on screen.
"""
import numpy as np

import cv2

from fieldtrack.refine import refine
from fieldtrack.smooth import smooth_homographies
from fieldtrack.tracker import track_field

GOAL_LINE_X = 10.0
# A frame needs this many painted-line points in view, and the lines must land
# this close, before its refined fit is trusted.
MIN_LINE_POINTS = 60
MAX_RESIDUAL_PX = 8.0
FIELD_TO_WORLD = np.array([[1.0, 0.0, -GOAL_LINE_X],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])


def screen_to_world(field_to_screen):
    """Image -> world matrix the pipeline projects feet with."""
    A = FIELD_TO_WORLD @ np.linalg.inv(field_to_screen)
    return A / A[2, 2]


def field_homographies(video, calibration, n_frames):
    """Per-frame image->world matrices and statuses for the first n_frames.

    Every fit is refined against the painted yard lines (fieldtrack.refine),
    then smoothed over time to remove frame-to-frame noise.
    Frames before the calibration frame reuse the calibration homography.
    """
    mats, status = [], []
    first = int(np.load(calibration)["frame_index"])
    for state in track_field(video, calibration):
        if state.frame_index >= n_frames:
            break
        # Pin the tracked fit to the painted yard lines so drift cannot build up.
        fit, residual, lines = refine(state.homography, state.frame)
        trusted = lines >= MIN_LINE_POINTS and residual <= MAX_RESIDUAL_PX
        if not trusted:
            fit = state.homography
        while len(mats) < state.frame_index:
            mats.append(screen_to_world(fit))
            status.append("stale")
        mats.append(screen_to_world(fit))
        if state.frame_index == first:
            status.append("fresh")
        elif state.confidence == "LOW" or not trusted:
            status.append("stale")
        else:
            status.append("chained")
    while len(mats) < n_frames:
        mats.append(mats[-1])
        status.append("stale")
    cap = cv2.VideoCapture(str(video))
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    return smooth_homographies(mats, shape), status
