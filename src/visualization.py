import cv2
import numpy as np


def draw_detected_lines(frame, lines):
    result = frame.copy()

    if lines is None:
        return result

    normalized_lines = np.asarray(lines).reshape(-1, 4)

    for x1, y1, x2, y2 in normalized_lines:
        cv2.line(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2,
        )

    return result


def draw_field_grid(frame, grid_lines, homography):
    """Project the permanent field grid into the gameplay frame."""

    result = frame.copy()
    height, width = result.shape[:2]

    # Field points in front of the camera share the sign of their
    # projective w with the point seen at the image center.
    image_center = np.array([width / 2, height / 2, 1.0])
    center_on_field = np.linalg.inv(homography) @ image_center
    center_on_field /= center_on_field[2]
    front_sign = np.sign((homography @ center_on_field)[2])

    for start, end in grid_lines:
        # Split each line into short pieces so the parts behind the
        # camera can be skipped. Projecting only the two endpoints
        # draws a wrong line when one of them is behind the camera.
        steps = np.linspace(0.0, 1.0, 61)[:, None]
        field_points = (
            np.array(start) + steps * (np.array(end) - np.array(start))
        )

        projected = np.hstack(
            [field_points, np.ones((len(field_points), 1))]
        ) @ homography.T

        w = projected[:, 2] * front_sign
        screen_points = projected[:, :2] / projected[:, 2:3]

        for index in range(len(screen_points) - 1):
            if w[index] <= 0 or w[index + 1] <= 0:
                continue

            first = screen_points[index]
            second = screen_points[index + 1]

            # Skip pieces near the horizon that project too far away.
            if np.abs(np.concatenate([first, second])).max() > 1e5:
                continue

            point1 = (
                int(round(first[0])),
                int(round(first[1])),
            )

            point2 = (
                int(round(second[0])),
                int(round(second[1])),
            )

            # Clip grid lines to the visible frame.
            visible, clipped_point1, clipped_point2 = cv2.clipLine(
                (0, 0, width, height),
                point1,
                point2,
            )

            if visible:
                cv2.line(
                    result,
                    clipped_point1,
                    clipped_point2,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    return result


def draw_tracked_points(frame, screen_points):
    """Draw the automatically tracked field points as small dots."""

    result = frame.copy()

    for x, y in screen_points:
        cv2.circle(
            result,
            (int(round(x)), int(round(y))),
            2,
            (0, 255, 0),
            -1,
        )

    return result


def draw_calibration_points(frame, screen_points):
    result = frame.copy()

    for index, point in enumerate(screen_points):
        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(
            result,
            (x, y),
            7,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            result,
            str(index + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    return result

def draw_tracking_status(
    frame,
    frame_index,
    tracked_count,
    confidence,
):
    result = frame.copy()

    text = (
        f"Frame: {frame_index} | "
        f"Points: {tracked_count} | "
        f"Confidence: {confidence}"
    )

    color = {
        "HIGH": (0, 255, 0),
        "MEDIUM": (0, 255, 255),
        "LOW": (0, 0, 255),
    }.get(confidence, (255, 255, 255))

    cv2.rectangle(
        result,
        (10, 10),
        (570, 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        result,
        text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return result

