# Per-frame model of what the empty field looks like, built from the
# field homography:
#   region      where players may be (the field and end zones)
#   background  pixels that are just grass / end-zone paint / painted markings
#   support     pixels a player could be standing on
#   white, hud  helper masks
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.field_mapping import (
    field_coordinate_grid,
    front_sign,
    step_vectors,
    upright_scale,
)
from src.field_model import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    LOWER_HASH_Y,
    LOWER_NUMBER_Y,
    UPPER_HASH_Y,
    UPPER_NUMBER_Y,
)
from src.preprocessing import create_player_hud_mask

# ---- Tuning ----
# A pixel is "field" if its hue is within this many OpenCV hue units
# (0-179) of the zone's paint hue. Brightness is mostly ignored, so
# shadows and mowing stripes still count as field.
HUE_TOLERANCE = 14

# ...and its saturation is at least this fraction of the paint's.
SATURATION_FRACTION = 0.45
MIN_SATURATION = 25
MIN_VALUE = 20

# A zone needs at least this fraction of the frame to refit its color.
MIN_ZONE_FRACTION = 0.01

# Players may stand this far outside the sidelines / end lines.
REGION_MARGIN_YARDS = 0.5

# Extra room above the far edge so heads of far players are not cut off.
# Kept small so crowd pixels there stay too short to pass as a player.
HEADROOM_YARDS = 0.8

# Half-width of the painted-line mask, in yards.
LINE_HALF_WIDTH_YARDS = 0.4
HASH_HALF_WIDTH_YARDS = 0.6

# Half-size of the box around each painted yard number, in yards.
NUMBER_HALF_EXTENT_YARDS = 2.0

WHITE_LOWER = np.array([0, 0, 150])
WHITE_UPPER = np.array([180, 70, 255])

ZONES = (
    ("left_end_zone", -np.inf, 10.0),
    ("field", 10.0, 110.0),
    ("right_end_zone", 110.0, np.inf),
)


@dataclass
class BackgroundInfo:
    region: np.ndarray
    background: np.ndarray
    support: np.ndarray
    white: np.ndarray
    hud: np.ndarray
    models: dict = field(default_factory=dict)  # zone -> (hue, saturation)


def _project(points, homography):
    homogeneous = np.hstack(
        [points, np.ones((len(points), 1))]
    ) @ homography.T

    w = homogeneous[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        screen = homogeneous[:, :2] / w[:, None]

    return screen, w


def _marking_lines():
    """(start, end, perpendicular axis, half width). Axis 0 = field x."""

    lines = []

    x = 10.0

    while x <= 110.0:
        lines.append(
            ((x, 0.0), (x, FIELD_WIDTH), 0, LINE_HALF_WIDTH_YARDS)
        )
        x += 5.0

    for x in (0.0, FIELD_LENGTH):
        lines.append(
            ((x, 0.0), (x, FIELD_WIDTH), 0, LINE_HALF_WIDTH_YARDS)
        )

    for y in (0.0, FIELD_WIDTH):
        lines.append(
            ((0.0, y), (FIELD_LENGTH, y), 1, LINE_HALF_WIDTH_YARDS)
        )

    for y in (UPPER_HASH_Y, LOWER_HASH_Y):
        lines.append(
            ((10.0, y), (110.0, y), 1, HASH_HALF_WIDTH_YARDS)
        )

    return lines


def create_marking_mask(shape, homography):
    """
    Project the known painted markings (yard lines, sidelines, hash rows,
    number areas) into the frame. Line thickness follows the local scale.
    """

    height, width = shape[:2]
    mask = np.zeros((height, width), np.uint8)

    sign = front_sign(homography, width, height)
    samples = np.linspace(0.0, 1.0, 41)[:, None]

    for start, end, axis, half_width in _marking_lines():
        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)

        points = start + samples * (end - start)
        screen, w = _project(points, homography)

        step = step_vectors(homography, points)[axis]
        thickness = np.linalg.norm(step, axis=1) * 2.0 * half_width
        thickness = np.clip(
            np.nan_to_num(thickness, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            200.0,
        )

        front = (w * sign) > 0

        for index in range(len(points) - 1):
            if not (front[index] and front[index + 1]):
                continue

            first = screen[index]
            second = screen[index + 1]
            pair = np.concatenate([first, second])

            if not np.all(np.isfinite(pair)) or np.abs(pair).max() > 1e5:
                continue

            line_width = max(
                3,
                int(round(max(thickness[index], thickness[index + 1]))) + 2,
            )

            visible, point1, point2 = cv2.clipLine(
                (0, 0, width, height),
                (int(round(first[0])), int(round(first[1]))),
                (int(round(second[0])), int(round(second[1]))),
            )

            if visible:
                cv2.line(mask, point1, point2, 255, line_width)

    extent = NUMBER_HALF_EXTENT_YARDS

    for x in range(20, 101, 10):
        for y in (UPPER_NUMBER_Y, LOWER_NUMBER_Y):
            corners = np.array(
                [
                    [x - extent, y - extent],
                    [x + extent, y - extent],
                    [x + extent, y + extent],
                    [x - extent, y + extent],
                ],
                dtype=np.float64,
            )

            screen, w = _project(corners, homography)

            if np.any(w * sign <= 0):
                continue

            if not np.all(np.isfinite(screen)) or np.abs(screen).max() > 1e5:
                continue

            cv2.fillConvexPoly(
                mask,
                np.round(screen).astype(np.int32),
                255,
            )

    return mask


class FieldBackground:
    def __init__(self):
        # Last good paint color of each zone, reused when a zone is
        # barely visible in a frame.
        self.models = {}

    def reset(self):
        self.models.clear()

    @staticmethod
    def _to_full(small_mask, width, height):
        return cv2.resize(
            small_mask.astype(np.uint8) * 255,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    def _fit(self, name, hsv, fit_mask):
        enough = cv2.countNonZero(fit_mask) >= (
            MIN_ZONE_FRACTION * fit_mask.size
        )

        if enough:
            bright = cv2.inRange(hsv[:, :, 2], 40, 255)
            mask = cv2.bitwise_and(fit_mask, bright)

            histogram = cv2.calcHist(
                [hsv],
                [0, 1],
                mask,
                [90, 32],
                [0, 180, 0, 256],
            )

            # Ignore gray / white bins (painted lines, glare).
            histogram[:, :3] = 0

            if histogram.max() > 0:
                hue_bin, sat_bin = np.unravel_index(
                    np.argmax(histogram),
                    histogram.shape,
                )

                self.models[name] = (
                    hue_bin * 2.0 + 1.0,
                    sat_bin * 8.0 + 4.0,
                )

        return self.models.get(name)

    @staticmethod
    def _like(hsv, model):
        hue, saturation = model

        hue_difference = np.abs(
            hsv[:, :, 0].astype(np.int16) - int(round(hue))
        )

        hue_difference = np.minimum(hue_difference, 180 - hue_difference)

        min_saturation = max(
            MIN_SATURATION,
            SATURATION_FRACTION * saturation,
        )

        like = (
            (hue_difference <= HUE_TOLERANCE)
            & (hsv[:, :, 1] >= min_saturation)
            & (hsv[:, :, 2] >= MIN_VALUE)
        )

        return like.astype(np.uint8) * 255

    def analyze(self, frame, homography):
        """Return a BackgroundInfo, or None if the field is not usable."""

        height, width = frame.shape[:2]

        grid = field_coordinate_grid(homography, width, height)

        if grid is None:
            return None

        field_x, field_y, valid = grid

        margin = REGION_MARGIN_YARDS

        in_region = (
            valid
            & (field_x >= -margin)
            & (field_x <= FIELD_LENGTH + margin)
            & (field_y >= -margin)
            & (field_y <= FIELD_WIDTH + margin)
        )

        if not in_region.any():
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hud = create_player_hud_mask(frame.shape)

        hud_small = cv2.resize(
            hud,
            (field_x.shape[1], field_x.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

        region = np.zeros((height, width), np.uint8)
        background = np.zeros((height, width), np.uint8)
        used = {}

        for name, x_low, x_high in ZONES:
            zone_small = (
                in_region & (field_x >= x_low) & (field_x < x_high)
            )

            if not zone_small.any():
                continue

            # Fit the color away from zone borders, sidelines and the HUD,
            # where painted lines and overlays would pollute it.
            fit_small = (
                zone_small
                & ~hud_small
                & (field_x >= x_low + 1.0)
                & (field_x < x_high - 1.0)
                & (field_y >= 1.0)
                & (field_y <= FIELD_WIDTH - 1.0)
            )

            model = self._fit(
                name,
                hsv,
                self._to_full(fit_small, width, height),
            )

            if model is None:
                continue

            zone_full = self._to_full(zone_small, width, height) > 0
            like = self._like(hsv, model)

            region[zone_full] = 255
            background[zone_full] = like[zone_full]
            used[name] = model

        if not used:
            return None

        # Headroom: let the region reach a little above the far edge.
        rows, columns = np.nonzero(in_region)
        stride = max(1, len(rows) // 200)

        sample = np.stack(
            [
                field_x[rows[::stride], columns[::stride]],
                field_y[rows[::stride], columns[::stride]],
            ],
            axis=1,
        )

        scales = upright_scale(homography, sample)
        scales = scales[np.isfinite(scales) & (scales > 0)]

        if len(scales) > 0:
            pad = int(round(HEADROOM_YARDS * np.percentile(scales, 10)))
            pad = min(pad, int(0.1 * height))

            if pad >= 1:
                region = cv2.dilate(
                    region,
                    np.ones((pad + 1, 1), np.uint8),
                    anchor=(0, 0),  # grows the region upward only
                )

        # Painted markings: white pixels on the projected markings are
        # background. Dilate a little to take the anti-aliased edges too.
        markings = create_marking_mask(frame.shape, homography)
        white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

        marking_paint = cv2.dilate(
            cv2.bitwise_and(white, markings),
            np.ones((5, 5), np.uint8),
        )

        background = cv2.bitwise_or(background, marking_paint)

        support = cv2.bitwise_or(
            cv2.bitwise_or(background, markings),
            hud,
        )

        return BackgroundInfo(
            region=region,
            background=background,
            support=support,
            white=white,
            hud=hud,
            models=used,
        )