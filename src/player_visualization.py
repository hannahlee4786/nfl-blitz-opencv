# Drawing players on the gameplay frame, and composing the final frame
import cv2
import numpy as np


def id_color(track_id):
    """A distinct, bright BGR color for each ID."""

    hue = (track_id * 47) % 180
    hsv = np.array([[[hue, 230, 255]]], dtype=np.uint8)

    return tuple(
        int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    )


def draw_player_tracks(frame, tracks):
    """Boxes, foot points and IDs for players matched in this frame."""

    result = frame.copy()

    for track in tracks:
        if not track.matched:
            continue

        color = id_color(track.id)
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        cv2.circle(
            result,
            (int(round(track.last_foot[0])), int(round(track.last_foot[1]))),
            4,
            color,
            -1,
        )

        label = f"#{track.id}"

        cv2.putText(
            result,
            label,
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            result,
            label,
            (x1, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

    return result


def compose_side_by_side(gameplay, panel):
    """Gameplay on the left, top-down panel on the right, centered."""

    gameplay_height, gameplay_width = gameplay.shape[:2]
    panel_height, panel_width = panel.shape[:2]

    height = max(gameplay_height, panel_height)

    canvas = np.zeros(
        (height, gameplay_width + panel_width, 3),
        dtype=np.uint8,
    )

    top = (height - gameplay_height) // 2
    canvas[top:top + gameplay_height, :gameplay_width] = gameplay

    top = (height - panel_height) // 2
    canvas[top:top + panel_height, gameplay_width:] = panel

    return canvas