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

    for start, end in grid_lines:
        field_line = np.array(
            [
                [start],
                [end],
            ],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            field_line,
            homography,
        )

        first = transformed[0, 0]
        second = transformed[1, 0]

        # Ignore invalid projected values.
        if not np.all(np.isfinite([first, second])):
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

