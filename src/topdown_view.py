
"""Renders the top-down football field and tracked players.

Field coordinates:
    x: 0 to 120 yards (field length)
    y: 0 to 53.33 yards (field width)

Player markers and movement trails use jersey-team colors:
    Red team  -> Red circles
    Blue team -> Blue circles

Tracking IDs remain consistent with the gameplay view.
"""

import cv2
import numpy as np

from src.field_model import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    LOWER_HASH_Y,
    LOWER_NUMBER_Y,
    UPPER_HASH_Y,
    UPPER_NUMBER_Y,
)

from src.player_visualization import team_color

from src.field_mapping import (
    field_to_screen,
    invert_homography,
    screen_to_field,
)


MARGIN_YARDS = 1.0

GRASS = (46, 110, 46)
END_ZONE = (90, 70, 40)
LINE = (235, 235, 235)

ORIENTATIONS = ("camera", "cw", "ccw", "flip")


def auto_orientation(homography, frame_size):
    """
    Automatically choose the top-down field orientation
    based on the gameplay camera.
    """
    width, height = frame_size

    inverse = invert_homography(homography)

    if inverse is None:
        return "camera"

    center = screen_to_field(
        [[width / 2.0, height / 2.0]],
        inverse,
    )[0]

    probe = np.array(
        [
            center,
            center + [5.0, 0.0],
            center + [0.0, 5.0],
        ],
        dtype=np.float64,
    )

    screen = field_to_screen(probe, homography)

    x_goes_up = screen[1][1] < screen[0][1]
    y_goes_right = screen[2][0] > screen[0][0]

    if x_goes_up:
        return "ccw" if y_goes_right else "flip"

    return "camera" if y_goes_right else "cw"


class TopDownField:
    """Draws the football field and tracked player positions."""

    def __init__(
        self,
        height_pixels,
        orientation="camera",
        trail=True,
    ):
        if orientation not in ORIENTATIONS:
            raise ValueError(
                f"Unknown orientation '{orientation}'. "
                f"Expected one of {ORIENTATIONS}."
            )

        self.orientation = orientation

        self.pixels_per_yard = height_pixels / (
            FIELD_LENGTH + 2 * MARGIN_YARDS
        )

        self.margin = (
            MARGIN_YARDS * self.pixels_per_yard
        )

        self.trail = trail

        self.marker_radius = max(
            5,
            int(round(self.pixels_per_yard * 1.0)),
        )

        self.base = self._draw_base()

    @property
    def size(self):
        height, width = self.base.shape[:2]
        return width, height

    def to_pixel(self, x, y):
        """
        Convert field coordinates to top-down pixels.

        camera: +x down, +y right
        cw:     +x down, +y left
        ccw:    +x up,   +y right
        flip:   +x up,   +y left
        """
        if self.orientation == "camera":
            screen_x = (
                self.margin
                + y * self.pixels_per_yard
            )
            screen_y = (
                self.margin
                + x * self.pixels_per_yard
            )

        elif self.orientation == "cw":
            screen_x = (
                self.margin
                + (FIELD_WIDTH - y)
                * self.pixels_per_yard
            )
            screen_y = (
                self.margin
                + x * self.pixels_per_yard
            )

        elif self.orientation == "ccw":
            screen_x = (
                self.margin
                + y * self.pixels_per_yard
            )
            screen_y = (
                self.margin
                + (FIELD_LENGTH - x)
                * self.pixels_per_yard
            )

        else:  # flip
            screen_x = (
                self.margin
                + (FIELD_WIDTH - y)
                * self.pixels_per_yard
            )
            screen_y = (
                self.margin
                + (FIELD_LENGTH - x)
                * self.pixels_per_yard
            )

        return (
            int(round(screen_x)),
            int(round(screen_y)),
        )

    def _draw_base(self):
        """Draw the football field background."""
        width = int(
            round(
                FIELD_WIDTH * self.pixels_per_yard
                + 2 * self.margin
            )
        )

        height = int(
            round(
                FIELD_LENGTH * self.pixels_per_yard
                + 2 * self.margin
            )
        )

        image = np.full(
            (height, width, 3),
            GRASS,
            dtype=np.uint8,
        )

        # End zones
        for start, end in (
            (0.0, 10.0),
            (110.0, 120.0),
        ):
            cv2.rectangle(
                image,
                self.to_pixel(start, 0.0),
                self.to_pixel(end, FIELD_WIDTH),
                END_ZONE,
                -1,
            )

        # Yard lines every 5 yards
        x = 10.0

        while x <= 110.0:
            thickness = (
                2 if x % 10 == 0 else 1
            )

            cv2.line(
                image,
                self.to_pixel(x, 0.0),
                self.to_pixel(x, FIELD_WIDTH),
                LINE,
                thickness,
                cv2.LINE_AA,
            )

            x += 5.0

        # Hash marks
        for hash_y in (
            UPPER_HASH_Y,
            LOWER_HASH_Y,
        ):
            for x in range(11, 110):
                cv2.line(
                    image,
                    self.to_pixel(x, hash_y - 0.3),
                    self.to_pixel(x, hash_y + 0.3),
                    LINE,
                    1,
                )

        # Yard numbers
        scale = max(
            0.35,
            self.pixels_per_yard / 14.0,
        )

        for x in range(20, 101, 10):
            label = str(
                min(x - 10, 110 - x)
            )

            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                1,
            )

            for number_y in (
                UPPER_NUMBER_Y,
                LOWER_NUMBER_Y,
            ):
                center = self.to_pixel(
                    x,
                    number_y,
                )

                cv2.putText(
                    image,
                    label,
                    (
                        center[0] - text_size[0] // 2,
                        center[1] + text_size[1] // 2,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    LINE,
                    1,
                    cv2.LINE_AA,
                )

        # Field boundary
        cv2.rectangle(
            image,
            self.to_pixel(0.0, 0.0),
            self.to_pixel(
                FIELD_LENGTH,
                FIELD_WIDTH,
            ),
            LINE,
            2,
        )

        return image

    def render(self, tracks):
        """
        Render tracked players on the top-down field.

        Red jerseys -> red circles
        Blue jerseys -> blue circles

        Matched tracks are filled.
        Temporarily missing tracks are hollow.

        Player IDs match the gameplay bounding boxes.
        """
        image = self.base.copy()

        # Draw movement trails
        if self.trail:
            for track in tracks:
                if len(track.history) < 2:
                    continue

                points = np.array(
                    [
                        self.to_pixel(x, y)
                        for x, y in track.history
                    ],
                    dtype=np.int32,
                )

                # Use the same color as the gameplay box.
                color = team_color(track)

                cv2.polylines(
                    image,
                    [points],
                    False,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Draw player markers
        for track in tracks:
            # Same red/blue color as gameplay bounding box
            color = team_color(track)

            center = self.to_pixel(
                *track.position
            )

            if track.matched:
                # Filled circle for currently detected players
                cv2.circle(
                    image,
                    center,
                    self.marker_radius,
                    color,
                    -1,
                )

                cv2.circle(
                    image,
                    center,
                    self.marker_radius,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            else:
                # Hollow circle for temporarily missing players
                cv2.circle(
                    image,
                    center,
                    self.marker_radius,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Same persistent tracking ID as gameplay
            label = str(track.id)

            font_scale = max(
                0.35,
                self.marker_radius / 14.0,
            )

            text_position = (
                center[0] + self.marker_radius + 1,
                center[1] - 2,
            )

            # Black outline for readability
            cv2.putText(
                image,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # White ID label
            cv2.putText(
                image,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Display orientation
        cv2.putText(
            image,
            f"view: {self.orientation}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return image
