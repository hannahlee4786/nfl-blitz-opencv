"""Clicking a manual field anchor on one video frame.

Ported from the camera-tracking branch's ``initialize_homography``: the
landmarks are the centers of the painted yard numbers, which sit about 22
yards apart across the field and pin the homography down better than the
hash rows.

Field coordinates follow the offense. NFL Blitz starts every play with the
camera behind the team with the ball, so x = 0 is the end zone behind the
camera on the anchor frame and x increases in the direction the offense
attacks. With z up, y = 0 is the sideline on the offense's right. Nothing
painted on the field, such as end-zone team names, is needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .field import NFLField

# NFL Blitz paints the yard numbers 15.8 yards in from each sideline
# (measured on Clip 3), not about 8 yards as on a real NFL field.
UPPER_NUMBER_Y = 15.8
LOWER_NUMBER_Y = round(NFLField.standard().width_yards - UPPER_NUMBER_Y, 3)

# The painted numbers are wide targets, so careful clicks can still be a few
# pixels off center. 8 px keeps them and still rejects a wrong click.
RANSAC_THRESHOLD_PX = 8.0

# Height of the instruction panel shown above the frame.
PANEL_HEIGHT = 80


def yard_number_labels(yard_numbers=(20, 30, 40)) -> list[str]:
    """Return the click prompts: each number's left then right copy."""

    return [
        f"{yard_number}-yard line, {side} number"
        for yard_number in yard_numbers
        for side in ("left", "right")
    ]


def anchor_field_points(
    yard_numbers,
    image_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Place clicked yard numbers on the field from where they appear.

    The camera looks down the field, so a farther number sits higher on the
    screen. If the smallest number is the nearest, the numbers are on the
    offense's own half (the 20 is x = 30); if it is the farthest, they are on
    the half the offense attacks (the 20 is x = 90). In each pair, the copy
    further left on screen is on the offense's left, y = LOWER_NUMBER_Y.
    """

    numbers = list(yard_numbers)
    pairs = np.asarray(image_points, dtype=float).reshape(len(numbers), 2, 2)
    line_heights = pairs[:, :, 1].mean(axis=1)
    smallest, largest = int(np.argmin(numbers)), int(np.argmax(numbers))
    own_half = line_heights[smallest] > line_heights[largest]
    field_points = []
    for number, pair in zip(numbers, pairs):
        x = 10.0 + number if own_half else 110.0 - number
        left_first = pair[0, 0] <= pair[1, 0]
        sides = (LOWER_NUMBER_Y, UPPER_NUMBER_Y) if left_first else (UPPER_NUMBER_Y, LOWER_NUMBER_Y)
        field_points.extend((x, y) for y in sides)
    return np.asarray(field_points)


def build_anchor(
    field_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
    frame: int,
    image_size: tuple[int, int],
) -> tuple[dict, float]:
    """Fit the clicks, drop rejected ones, and return the anchor JSON payload
    with the mean reprojection error of the kept clicks."""

    field_points = np.asarray(field_points, dtype=np.float32).reshape(-1, 2)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    homography, inliers = cv2.findHomography(
        field_points, image_points, cv2.RANSAC, RANSAC_THRESHOLD_PX
    )
    if homography is None:
        raise ValueError(
            "Could not fit the clicks. Check that they are on the requested "
            "numbers and not all on one line."
        )
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 4:
        raise ValueError(
            f"Only {int(keep.sum())} of {len(keep)} clicks agree; click again."
        )
    projected = cv2.perspectiveTransform(
        field_points[keep].reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    error = float(np.linalg.norm(projected - image_points[keep], axis=1).mean())
    payload = {
        "schema_version": 1,
        "frame": int(frame),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "field_points": np.round(field_points[keep].astype(float), 3).tolist(),
        "image_points": np.round(image_points[keep].astype(float), 1).tolist(),
    }
    return payload, error


def read_frame(video: Path, index: int) -> tuple[NDArray[np.uint8], int]:
    """Return one BGR frame and the video's frame count."""

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Could not read frame {index} of {video}")
    return frame, frame_count


def move_frame_index(current: int, offset: int, frame_count: int) -> int:
    """Step to another frame without leaving the video."""

    if frame_count <= 0:
        raise ValueError("The video must contain at least one frame.")
    return min(max(current + offset, 0), frame_count - 1)


def frame_offset_for_key(key: int, fps: float) -> int | None:
    """A/D step one frame and [ / ] step one second; other keys do nothing."""

    one_second = max(1, int(round(fps)))
    return {
        ord("a"): -1,
        ord("d"): 1,
        ord("["): -one_second,
        ord("]"): one_second,
    }.get(key)


def browse_frames(video: Path, start: int = 0) -> tuple[int, NDArray[np.uint8]]:
    """Let the user step through the video and pick the frame to click on.

    Ported from the frame browser on Joyce's game-state-reconstruction branch.
    A/D move one frame, [ / ] one second, Enter picks, Escape cancels.
    """

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    index = move_frame_index(start, 0, frame_count)
    window = "Choose anchor frame"
    cv2.namedWindow(window)
    try:
        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = capture.read()
            if not success:
                # The reported frame count can overshoot the decodable frames.
                if index == 0:
                    raise RuntimeError(f"Could not read frame 0 of {video}")
                frame_count = index
                index -= 1
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 1100 / width, (800 - PANEL_HEIGHT) / height)
            display = cv2.resize(frame, (int(width * scale), int(height * scale)))
            display = cv2.copyMakeBorder(display, PANEL_HEIGHT, 0, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(display, f"Frame {index}/{frame_count - 1}: pick one where the yard numbers show",
                        (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display, "A/D = 1 frame | [ / ] = 1 second | ENTER = select | ESC = cancel",
                        (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(0) & 0xFF
            if key in (10, 13):
                return index, frame
            if key == 27:
                raise RuntimeError("Frame selection cancelled.")
            offset = frame_offset_for_key(key, fps)
            if offset is not None:
                index = move_frame_index(index, offset, frame_count)
    finally:
        capture.release()
        cv2.destroyWindow(window)
        cv2.waitKey(1)


def click_image_points(
    frame: NDArray[np.uint8],
    labels: list[str],
) -> NDArray[np.float32]:
    """Show the frame and collect one click per label, in original pixels.

    Left click selects, U undoes, R resets, Enter accepts, Escape cancels.
    """

    height, width = frame.shape[:2]
    scale = min(1.0, 1100 / width, (800 - PANEL_HEIGHT) / height)
    shown = cv2.resize(frame, (int(width * scale), int(height * scale)))
    points: list[tuple[float, float]] = []
    window = "Select field points"

    def render():
        display = shown.copy()
        for index, (x, y) in enumerate(points):
            center = (int(x * scale), int(y * scale))
            cv2.circle(display, center, 7, (0, 0, 255), -1)
            cv2.putText(display, str(index + 1), (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) < len(labels):
            instruction = f"Click point {len(points) + 1}/{len(labels)}: {labels[len(points)]}"
        else:
            instruction = "All points selected. Press ENTER to continue."
        # The panel goes above the frame so landmarks near the top edge stay
        # visible and clickable.
        display = cv2.copyMakeBorder(display, PANEL_HEIGHT, 0, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(display, instruction, (15, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        cv2.putText(display, "U = undo | R = reset | ESC = cancel", (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return display

    def on_mouse(event, x, y, flags, parameter):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= len(labels):
            return
        y -= PANEL_HEIGHT
        if y >= 0:
            points.append((x / scale, y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            cv2.imshow(window, render())
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(points) == len(labels):
                break
            if key == ord("u") and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key == 27:
                raise RuntimeError("Point selection cancelled.")
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    return np.asarray(points, dtype=np.float32)
