# Drawing players on the gameplay frame, and composing the final frame
import cv2
import numpy as np

# Debug colors (BGR) for each rejection reason.
REJECT_COLORS = {
    "FIELD": (255, 128, 0),    # feet off the field
    "SIZE": (0, 0, 255),       # wrong height / width in yards
    "ASPECT": (0, 140, 255),   # too wide and short
    "WHITE": (255, 255, 255),  # almost entirely white paint
    "GROUND": (255, 0, 255),   # nothing field-like under the feet
    "DUP": (140, 140, 140),    # overlaps a better box
    "JERSEY": (0, 255, 255),   # not enough red / blue (or flat HUD color)
    "QUALITY": (0, 180, 180),
}


def id_color(track_id):
    """A distinct, bright BGR color for each ID."""

    hue = (track_id * 47) % 180
    hsv = np.array([[[hue, 230, 255]]], dtype=np.uint8)

    return tuple(
        int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    )

# Box colors (BGR) by jersey team.
TEAM_COLORS = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
}


def team_color(track):
    """Red / blue by jersey, or the per-ID color if the team is unknown."""

    return TEAM_COLORS.get(track.team) or id_color(track.id)


def draw_player_tracks(frame, tracks):
    """Team-colored boxes, foot points and IDs for players matched now."""

    result = frame.copy()

    for track in tracks:
        if not track.matched:
            continue

        color = team_color(track)
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
        origin = (x1, max(12, y1 - 5))

        # White text with a dark outline stays readable on either color.
        cv2.putText(
            result, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )

        cv2.putText(
            result, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return result


def draw_rejected(frame, rejected):
    """
    Debug view: every candidate the filters threw out, with the reason
    and (for size / aspect rejections) its measured size in yards.
    """

    result = frame.copy()
    frame_height = frame.shape[0]

    for detection in rejected:
        color = REJECT_COLORS.get(detection.reason, (0, 0, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox]

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)

        label = str(detection.reason)

        if (
            detection.reason in ("SIZE", "ASPECT")
            and detection.height_yards is not None
        ):
            label += (
                f" h{detection.height_yards:.1f}"
                f" w{detection.width_yards:.1f}"
            )

        if detection.reason == "JERSEY":
            label += (
                f" r{detection.red_fraction:.2f}"
                f" b{detection.blue_fraction:.2f}"
            )

        cv2.putText(
            result,
            label,
            (x1, min(frame_height - 4, y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
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