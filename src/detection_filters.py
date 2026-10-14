
"""Validate proposed player detections."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.field_mapping import screen_to_field, upright_scale
from src.field_model import FIELD_LENGTH, FIELD_WIDTH
from src.jersey_color import JerseyClassifier, JerseyConfig


@dataclass
class FilterConfig:
    min_height: float = 1.6
    max_height: float = 4.5

    min_width: float = 0.6
    max_width: float = 3.0

    min_aspect: float = 0.65

    field_margin: float = 1.0
    max_white_fraction: float = 0.90
    min_ground_support: float = 0.25
    ground_band_yards: float = 0.3

    duplicate_overlap: float = 0.55
    min_detection_quality: float = 0.10
    min_jersey_difference: float = 0.03

    use_jersey_filter: bool = True
    jersey: JerseyConfig = field(
        default_factory=JerseyConfig
    )

class DetectionFilter:
    def __init__(self, config=None):
        self.config = config or FilterConfig()
        self.jersey = JerseyClassifier(self.config.jersey)

    def apply(
        self,
        detections,
        frame,
        homography,
        inverse,
        info,
    ):
        if not detections:
            return [], []

        hsv = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.config.use_jersey_filter
            else None
        )

        feet = np.asarray(
            [d.foot for d in detections],
            dtype=np.float32,
        )

        field_points = screen_to_field(feet, inverse)
        scales = upright_scale(homography, field_points)

        survivors = []
        rejected = []

        for detection, point, scale in zip(
            detections, field_points, scales
        ):
            detection.field_xy = point.astype(np.float64)

            reason = self._judge(
                detection, point, scale, info, hsv
            )

            if reason:
                detection.reason = reason
                rejected.append(detection)
            else:
                survivors.append(detection)

        kept, duplicates = self._suppress_duplicates(
            survivors
        )

        return kept, rejected + duplicates

    def _judge(self, detection, point, scale, info, hsv):
        cfg = self.config
        margin = cfg.field_margin

        if (
            not np.all(np.isfinite(point))
            or not np.isfinite(scale)
            or scale <= 1e-3
        ):
            return "FIELD"

        if not (
            -margin <= point[0] <= FIELD_LENGTH + margin
            and -margin <= point[1] <= FIELD_WIDTH + margin
        ):
            return "FIELD"

        x1, y1, x2, y2 = detection.bbox

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return "SIZE"

        detection.height_yards = height / scale
        detection.width_yards = width / scale

        if not (
            cfg.min_height
            <= detection.height_yards
            <= cfg.max_height
        ):
            return "SIZE"

        if not (
            cfg.min_width
            <= detection.width_yards
            <= cfg.max_width
        ):
            return "SIZE"

        if height / width < cfg.min_aspect:
            return "ASPECT"

        white_fraction = 0.0
        support_value = 1.0

        if info is not None:
            white_fraction = self._white_fraction(
                detection.bbox, info
            )

            if white_fraction > cfg.max_white_fraction:
                return "WHITE"

            support = self._ground_support(
                detection.bbox, scale, info
            )

            if (
                support is not None
                and support < cfg.min_ground_support
            ):
                return "GROUND"

            if support is not None:
                support_value = support

        jersey_factor = 1.0

        if hsv is not None:
            jersey = self.jersey.analyze(
                hsv,
                detection.bbox,
                scale,
                info,
            )

            detection.team = jersey.team
            detection.red_fraction = jersey.red_fraction
            detection.blue_fraction = jersey.blue_fraction

            if jersey.level is None:
                return "JERSEY"

            difference = abs(
                jersey.red_fraction
                - jersey.blue_fraction
            )

            if difference < cfg.min_jersey_difference:
                return "JERSEY"

            detection.jersey_strong = (
                jersey.level == "strong"
            )

            jersey_factor = jersey.factor

        quality = (
            detection.score
            * (0.5 + 0.5 * support_value)
            * (1.0 - 0.5 * white_fraction)
            * jersey_factor
        )

        detection.quality = float(
            np.clip(quality, 0.0, 1.0)
        )

        if detection.quality < cfg.min_detection_quality:
            return "QUALITY"

        return None

    @staticmethod
    def _clip_box(bbox, shape):
        height, width = shape[:2]

        x1 = max(0, int(np.floor(bbox[0])))
        y1 = max(0, int(np.floor(bbox[1])))
        x2 = min(width, int(np.ceil(bbox[2])))
        y2 = min(height, int(np.ceil(bbox[3])))

        return x1, y1, x2, y2

    def _white_fraction(self, bbox, info):
        x1, y1, x2, y2 = self._clip_box(
            bbox, info.background.shape
        )

        if x2 <= x1 or y2 <= y1:
            return 0.0

        foreground = (
            (info.background[y1:y2, x1:x2] == 0)
            & (info.hud[y1:y2, x1:x2] == 0)
        )

        count = int(foreground.sum())

        if count == 0:
            return 0.0

        white = (
            (info.white[y1:y2, x1:x2] > 0)
            & foreground
        )

        return float(white.sum()) / count

    def _ground_support(self, bbox, scale, info):
        height, width = info.support.shape

        x1 = max(0, int(np.floor(bbox[0])) - 2)
        x2 = min(width, int(np.ceil(bbox[2])) + 2)

        top = int(np.ceil(bbox[3])) + 1
        band = max(
            2,
            int(round(
                self.config.ground_band_yards * scale
            )),
        )
        bottom = min(height, top + band)

        if (
            x2 <= x1
            or top < 0
            or top >= height
            or bottom - top < 2
        ):
            return None

        return float(
            (info.support[top:bottom, x1:x2] > 0).mean()
        )

    @staticmethod
    def _overlap(a, b):
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[2], b[2])
        bottom = min(a[3], b[3])

        intersection = (
            max(0.0, right - left)
            * max(0.0, bottom - top)
        )

        area_a = (
            max(0.0, a[2] - a[0])
            * max(0.0, a[3] - a[1])
        )

        area_b = (
            max(0.0, b[2] - b[0])
            * max(0.0, b[3] - b[1])
        )

        smaller = min(area_a, area_b)

        return (
            intersection / smaller
            if smaller > 0
            else 0.0
        )

    def _suppress_duplicates(self, detections):
        """
        Remove duplicate boxes representing the same player.

        Uses overlap, center proximity, and team consistency.
        """
        ordered = sorted(
            detections,
            key=lambda d: d.quality,
            reverse=True,
        )

        kept = []
        rejected = []

        for detection in ordered:
            x1, y1, x2, y2 = detection.bbox

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)

            is_duplicate = False

            for other in kept:
                if detection.team != other.team:
                    continue

                ox1, oy1, ox2, oy2 = other.bbox

                other_cx = (ox1 + ox2) / 2.0
                other_cy = (oy1 + oy2) / 2.0

                other_width = max(1.0, ox2 - ox1)
                other_height = max(1.0, oy2 - oy1)

                overlap = self._overlap(
                    detection.bbox,
                    other.bbox,
                )

                dx = abs(center_x - other_cx)
                dy = abs(center_y - other_cy)

                close_centers = (
                    dx < 0.35 * max(width, other_width)
                    and dy < 0.35 * max(height, other_height)
                )

                if (
                    overlap > self.config.duplicate_overlap
                    or close_centers
                ):
                    is_duplicate = True
                    break

            if is_duplicate:
                detection.reason = "DUP"
                rejected.append(detection)
            else:
                kept.append(detection)

        return kept, rejected
