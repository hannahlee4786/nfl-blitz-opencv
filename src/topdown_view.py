# Renders the canonical top-down football field and the player markers
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
from src.player_visualization import id_color

MARGIN_YARDS = 1.0

GRASS = (46, 110, 46)
END_ZONE = (90, 70, 40)
LINE = (235, 235, 235)


class TopDownField:
    def __init__(self, width_pixels, trail=True):
        """
        width_pixels: width of the whole panel. The scale (pixels per
        yard) follows from it, so the panel is rendered at its final
        size and never resized.
        """

        self.pixels_per_yard = width_pixels / (
            FIELD_LENGTH + 2 * MARGIN_YARDS
        )

        self.margin = MARGIN_YARDS * self.pixels_per_yard
        self.trail = trail
        self.marker_radius = max(5, int(round(self.pixels_per_yard * 1.0)))
        self.base = self._draw_base()

    @property
    def size(self):
        height, width = self.base.shape[:2]
        return width, height

    def to_pixel(self, x, y):
        return (
            int(round(self.margin + x * self.pixels_per_yard)),
            int(round(self.margin + y * self.pixels_per_yard)),
        )

    def _draw_base(self):
        width = int(round(FIELD_LENGTH * self.pixels_per_yard + 2 * self.margin))
        height = int(round(FIELD_WIDTH * self.pixels_per_yard + 2 * self.margin))

        image = np.full((height, width, 3), GRASS, np.uint8)

        for start, end in ((0.0, 10.0), (110.0, 120.0)):
            cv2.rectangle(
                image,
                self.to_pixel(start, 0.0),
                self.to_pixel(end, FIELD_WIDTH),
                END_ZONE,
                -1,
            )

        # Yard lines every 5 yards, heavier every 10.
        x = 10.0

        while x <= 110.0:
            thickness = 2 if x % 10 == 0 else 1

            cv2.line(
                image,
                self.to_pixel(x, 0.0),
                self.to_pixel(x, FIELD_WIDTH),
                LINE,
                thickness,
                cv2.LINE_AA,
            )

            x += 5.0

        # One-yard hash ticks.
        for hash_y in (UPPER_HASH_Y, LOWER_HASH_Y):
            for x in range(11, 110):
                cv2.line(
                    image,
                    self.to_pixel(x, hash_y - 0.3),
                    self.to_pixel(x, hash_y + 0.3),
                    LINE,
                    1,
                )

        # Yard numbers, in the positions measured for Blitz.
        scale = max(0.35, self.pixels_per_yard / 14.0)

        for x in range(20, 101, 10):
            label = str(min(x - 10, 110 - x))
            size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                1,
            )

            for number_y in (UPPER_NUMBER_Y, LOWER_NUMBER_Y):
                center = self.to_pixel(x, number_y)

                cv2.putText(
                    image,
                    label,
                    (center[0] - size[0] // 2, center[1] + size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    LINE,
                    1,
                    cv2.LINE_AA,
                )

        cv2.rectangle(
            image,
            self.to_pixel(0.0, 0.0),
            self.to_pixel(FIELD_LENGTH, FIELD_WIDTH),
            LINE,
            2,
        )

        return image

    def render(self, tracks):
        """Draw every confirmed track. Coasting tracks are drawn hollow."""

        image = self.base.copy()

        if self.trail:
            for track in tracks:
                if len(track.history) < 2:
                    continue

                points = np.array(
                    [self.to_pixel(x, y) for x, y in track.history],
                    dtype=np.int32,
                )

                cv2.polylines(
                    image,
                    [points],
                    False,
                    id_color(track.id),
                    1,
                    cv2.LINE_AA,
                )

        for track in tracks:
            color = id_color(track.id)
            center = self.to_pixel(*track.position)

            if track.matched:
                cv2.circle(image, center, self.marker_radius, color, -1)
                cv2.circle(
                    image, center, self.marker_radius, (0, 0, 0), 1,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    image, center, self.marker_radius, color, 1,
                    cv2.LINE_AA,
                )

            label = str(track.id)
            font_scale = max(0.35, self.marker_radius / 14.0)

            cv2.putText(
                image,
                label,
                (center[0] + self.marker_radius + 1, center[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                label,
                (center[0] + self.marker_radius + 1, center[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return image