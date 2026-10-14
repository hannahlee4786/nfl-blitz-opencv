
"""Jersey-guided player detection for NFL Blitz gameplay."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.detection_filters import DetectionFilter, FilterConfig
from src.field_background import FieldBackground
from src.field_mapping import (
    invert_homography,
    screen_to_field,
    upright_scale,
)


@dataclass
class Detection:
    bbox: tuple
    score: float
    color: tuple
    field_xy: Optional[np.ndarray] = None
    quality: float = 1.0
    reason: Optional[str] = None
    height_yards: Optional[float] = None
    width_yards: Optional[float] = None
    team: Optional[str] = None
    red_fraction: float = 0.0
    blue_fraction: float = 0.0
    jersey_strong: bool = False

    @property
    def foot(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


def estimate_torso_color(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    left = max(0, int(x1 + (x2 - x1) * 0.25))
    right = min(w, int(x1 + (x2 - x1) * 0.75))
    top = max(0, int(y1 + (y2 - y1) * 0.15))
    bottom = min(h, int(y1 + (y2 - y1) * 0.65))

    crop = frame[top:bottom, left:right]

    if crop.size == 0:
        return (128.0, 128.0, 128.0)

    return tuple(
        float(v) for v in crop.reshape(-1, 3).mean(axis=0)
    )


class ColorBlobDetector:
    """
    Find red/blue jersey seeds and expand them into player boxes.

    Uses the existing field-background model to estimate
    player extent. Jersey pixels identify candidate centers.
    """

    name = "color"

    def __init__(
        self,
        min_seed_area=5,
        max_candidates=80,
        torso_width_yards=0.9,
        player_height_yards=2.5,
        min_foreground_fraction=0.08,
    ):
        self.min_seed_area = min_seed_area
        self.max_candidates = max_candidates
        self.torso_width_yards = torso_width_yards
        self.player_height_yards = player_height_yards
        self.min_foreground_fraction = min_foreground_fraction
        self.last_mask = None

    @staticmethod
    def _color_masks(hsv):
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        red = (
            ((hue <= 10) | (hue >= 168))
            & (sat >= 90)
            & (val >= 40)
        )

        blue = (
            (hue >= 104)
            & (hue <= 132)
            & (sat >= 70)
            & (val >= 40)
        )

        return (
            red.astype(np.uint8) * 255,
            blue.astype(np.uint8) * 255,
        )

    def propose(self, frame, homography, info):
        if info is None:
            self.last_mask = None
            return []

        inverse = invert_homography(homography)

        if inverse is None:
            return []

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red, blue = self._color_masks(hsv)

        # Merge nearby jersey-colored regions into larger candidates.
        # Keep the kernel small to avoid combining adjacent players.
        kernel_size = max(
            3,
            int(round(frame.shape[0] / 480.0 * 5)),
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        red = cv2.morphologyEx(
            red,
            cv2.MORPH_CLOSE,
            kernel,
        )

        blue = cv2.morphologyEx(
            blue,
            cv2.MORPH_CLOSE,
            kernel,
        )


        # Field geometry is used as a permissive spatial prior.
        # Do not mask the entire lower screen: real players
        # can extend into the lower HUD area.
        region = info.region > 0

        hud = info.hud > 0

        # Retain color pixels within the field region.
        valid = region & ~hud

        red = np.where(valid, red, 0).astype(np.uint8)
        blue = np.where(valid, blue, 0).astype(np.uint8)

        combined = cv2.bitwise_or(red, blue)
        self.last_mask = combined.copy()

        foreground = (
            (info.background == 0)
            & (info.hud == 0)
            & region
        )

        candidates = []

        for team, mask in (("red", red), ("blue", blue)):
            count, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    mask,
                    connectivity=8,
                )
            )

            components = sorted(
                range(1, count),
                key=lambda i: stats[i, cv2.CC_STAT_AREA],
                reverse=True,
            )

            for index in components[:self.max_candidates]:
                x, y, w, h, area = stats[index]

                if area < self.min_seed_area:
                    continue

                cx, cy = centroids[index]

                point = screen_to_field(
                    np.array([[cx, cy]], dtype=np.float32),
                    inverse,
                )

                if not np.all(np.isfinite(point)):
                    continue

                scale = float(
                    upright_scale(homography, point)[0]
                )

                if not np.isfinite(scale) or scale <= 0:
                    continue

                # Estimate a player box around the jersey.
                estimated_width = max(
                    8,
                    int(self.torso_width_yards * scale),
                    w,
                )

                estimated_height = max(
                    14,
                    int(self.player_height_yards * scale),
                    h,
                )

                # Jersey is normally in the upper body.
                left = int(cx - estimated_width / 2)
                right = int(cx + estimated_width / 2)
                top = int(cy - estimated_height * 0.35)
                bottom = top + estimated_height

                left = max(0, left)
                right = min(width, right)
                top = max(0, top)
                bottom = min(height, bottom)

                if right <= left or bottom <= top:
                    continue

                roi = foreground[top:bottom, left:right]

                if roi.size == 0:
                    continue

                foreground_fraction = float(roi.mean())

                if (
                    foreground_fraction
                    < self.min_foreground_fraction
                ):
                    continue

                # Refine using the local foreground extent,
                # but avoid expanding to unrelated distant blobs.
                rows, cols = np.where(roi)

                if len(rows) >= 5:
                    x_low = int(np.percentile(cols, 5))
                    x_high = int(np.percentile(cols, 95))
                    y_low = int(np.percentile(rows, 5))
                    y_high = int(np.percentile(rows, 95))

                    refined_left = left + x_low
                    refined_right = left + x_high + 1
                    refined_top = top + y_low
                    refined_bottom = top + y_high + 1

                    # Preserve the jersey seed within the box.
                    left = min(refined_left, x)
                    right = max(refined_right, x + w)
                    top = min(refined_top, y)
                    bottom = max(refined_bottom, y + h)

                left = max(0, left)
                top = max(0, top)
                right = min(width, right)
                bottom = min(height, bottom)

                bbox = (
                    float(left),
                    float(top),
                    float(right),
                    float(bottom),
                )

                color = estimate_torso_color(frame, bbox)

                # This is a proposal score, not calibrated
                # object-detection confidence.
                score = float(
                    np.clip(
                        0.4 + 0.6 * foreground_fraction,
                        0.0,
                        1.0,
                    )
                )

                candidates.append(
                    Detection(
                        bbox=bbox,
                        score=score,
                        color=color,
                        team=team,
                    )
                )

        return candidates


class YoloDetector:
    name = "yolo"

    def __init__(
        self,
        weights="yolov8n.pt",
        conf=0.25,
        person_only=True,
        image_size=960,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Install ultralytics to use YOLO."
            ) from error

        self.model = YOLO(weights)
        self.conf = conf
        self.person_only = person_only
        self.image_size = image_size
        self.last_mask = None

    def propose(self, frame, homography, info):
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.image_size,
            classes=[0] if self.person_only else None,
            verbose=False,
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        return [
            Detection(
                bbox=tuple(float(v) for v in box),
                score=float(score),
                color=estimate_torso_color(frame, box),
            )
            for box, score in zip(boxes, scores)
        ]


class PlayerDetector:
    def __init__(
        self,
        backend,
        background=None,
        detection_filter=None,
    ):
        self.backend = backend
        self.background = background or FieldBackground()
        self.filter = detection_filter or DetectionFilter()
        self.name = backend.name
        self.last_mask = None
        self.rejected = []

    def reset(self):
        self.background.reset()
        self.clear()

    def clear(self):
        self.last_mask = None
        self.rejected = []

    def detect(self, frame, homography):
        self.clear()

        inverse = invert_homography(homography)

        if inverse is None:
            return []

        info = self.background.analyze(frame, homography)

        proposals = self.backend.propose(
            frame, homography, info
        )

        self.last_mask = getattr(
            self.backend, "last_mask", None
        )

        kept, self.rejected = self.filter.apply(
            proposals,
            frame,
            homography,
            inverse,
            info,
        )

        return kept


def create_detector(
    name,
    weights="yolov8n.pt",
    conf=0.25,
    person_only=True,
    filter_config=None,
):
    if name == "color":
        backend = ColorBlobDetector()

    elif name == "yolo":
        backend = YoloDetector(
            weights=weights,
            conf=conf,
            person_only=person_only,
        )

    else:
        raise ValueError(f"Unknown detector: {name}")

    return PlayerDetector(
        backend,
        detection_filter=DetectionFilter(
            filter_config or FilterConfig()
        ),
    )
