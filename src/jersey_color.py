# Classifies a detection's jersey as red or blue from its upper body.
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class JerseyConfig:
    # ---- Color definitions (OpenCV HSV: hue 0-179, sat / value 0-255) ----
    # Red wraps around hue 0: hue <= red_hue_max or hue >= red_hue_min.
    red_hue_max: int = 10
    red_hue_min: int = 168
    red_min_saturation: int = 90

    # Blue sits just above the field's teal-green (field hue is ~80-100).
    blue_hue_min: int = 104
    blue_hue_max: int = 132
    blue_min_saturation: int = 70

    # Pixels darker than this (outlines, deep shadow) are ignored.
    min_value: int = 40

    # ---- Upper-body region, as fractions of the box ----
    torso_top: float = 0.10
    torso_bottom: float = 0.70
    torso_left: float = 0.10
    torso_right: float = 0.90

    # ---- Decision thresholds ----
    # Fraction of the non-field pixels in the region that must be the
    # jersey color. "strong" may start a new track; "weak" (partly
    # hidden jersey) may only keep an existing track alive.
    strong_fraction: float = 0.20
    weak_fraction: float = 0.08

    # Jersey-colored area, in square yards (so it holds across zoom).
    min_area_yards2: float = 0.04

    # Reject blobs that are almost purely one flat color: HUD icons and
    # field overlays, not sprites with pants, helmet and skin.
    max_fraction: float = 0.98
    max_fraction_min_pixels: int = 40

    # Fewer valid pixels than this: not enough to judge.
    min_pixels: int = 6


@dataclass
class JerseyResult:
    team: Optional[str]  # "red", "blue" or None
    red_fraction: float
    blue_fraction: float
    level: Optional[str]  # "strong", "weak" or None (not a jersey)
    factor: float  # 0.7 - 1.0 multiplier for the detection quality


def _clean(mask):
    """Remove isolated single-pixel speckle."""

    if min(mask.shape) < 4:
        return mask

    opened = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    return opened > 0


class JerseyClassifier:
    def __init__(self, config=None):
        self.config = config or JerseyConfig()

    def analyze(self, hsv, bbox, scale, info=None):
        """
        hsv:   the whole frame in HSV
        bbox:  (x1, y1, x2, y2)
        scale: pixels per yard for an upright player at this spot
        info:  optional BackgroundInfo; its field / HUD pixels are
               excluded so grass and end-zone paint never count.
        """

        config = self.config
        height, width = hsv.shape[:2]

        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        left = int(np.floor(x1 + config.torso_left * box_width))
        right = int(np.ceil(x1 + config.torso_right * box_width))
        top = int(np.floor(y1 + config.torso_top * box_height))
        bottom = int(np.ceil(y1 + config.torso_bottom * box_height))

        left, right = np.clip([left, right], 0, width)
        top, bottom = np.clip([top, bottom], 0, height)

        nothing = JerseyResult(None, 0.0, 0.0, None, 0.0)

        if right - left < 1 or bottom - top < 1:
            return nothing

        region = hsv[top:bottom, left:right]
        hue = region[:, :, 0]
        saturation = region[:, :, 1]
        value = region[:, :, 2]

        valid = value >= config.min_value

        if info is not None:
            valid = (
                valid
                & (info.background[top:bottom, left:right] == 0)
                & (info.hud[top:bottom, left:right] == 0)
            )

        valid_count = int(valid.sum())

        if valid_count < config.min_pixels:
            return nothing

        red = (
            ((hue <= config.red_hue_max) | (hue >= config.red_hue_min))
            & (saturation >= config.red_min_saturation)
            & valid
        )

        blue = (
            (hue >= config.blue_hue_min)
            & (hue <= config.blue_hue_max)
            & (saturation >= config.blue_min_saturation)
            & valid
        )

        red = _clean(red)
        blue = _clean(blue)

        red_count = int(red.sum())
        blue_count = int(blue.sum())

        red_fraction = red_count / valid_count
        blue_fraction = blue_count / valid_count

        if red_fraction >= blue_fraction:
            team, fraction, count = "red", red_fraction, red_count
        else:
            team, fraction, count = "blue", blue_fraction, blue_count

        area = count / max(scale, 1e-6) ** 2

        if (
            valid_count >= config.max_fraction_min_pixels
            and fraction > config.max_fraction
        ):
            level = None
        elif (
            fraction >= config.strong_fraction
            and area >= config.min_area_yards2
        ):
            level = "strong"
        elif (
            fraction >= config.weak_fraction
            and area >= 0.5 * config.min_area_yards2
        ):
            level = "weak"
        else:
            level = None

        factor = 0.7 + 0.3 * min(1.0, fraction / config.strong_fraction)

        return JerseyResult(team, red_fraction, blue_fraction, level, factor)