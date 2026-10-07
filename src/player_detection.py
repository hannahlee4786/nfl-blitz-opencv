# Finds players in a gameplay frame. Two interchangeable backends:
#   "color": classical blob detector for the green-field game footage
#   "yolo":  optional ultralytics model (pretrained COCO or fine-tuned)
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.field_mapping import (
    invert_homography,
    pixels_per_yard,
    screen_to_field,
)
from src.preprocessing import create_green_mask, create_hud_mask

# ---- Color-blob detector tuning (all sizes in yards, so they carry
# ---- across clips and zoom levels). Adjust after looking at results.
MIN_HEIGHT_YARDS = 0.8
MAX_HEIGHT_YARDS = 4.0
MIN_WIDTH_YARDS = 0.3
MAX_WIDTH_YARDS = 2.5
TYPICAL_WIDTH_YARDS = 1.2

# Blobs wider than SPLIT_RATIO * typical width are cut into several players.
SPLIT_RATIO = 1.6
MAX_SPLIT = 4

# Height / width of the box. Painted yard numbers are flat on the
# ground, so they come out very wide and short and are rejected here.
MIN_ASPECT = 0.7

# Fraction of the box that must be foreground.
MIN_FILL = 0.25

# Area in pixels at 480 p; scaled with the frame height.
MIN_AREA_PIXELS = 60


@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2) in screen pixels
    score: float
    color: tuple  # mean BGR color of the player
    field_xy: Optional[np.ndarray] = None  # filled by attach_field_positions

    @property
    def foot(self):
        """Ground contact estimate: bottom-center of the box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


def estimate_torso_color(frame, bbox):
    """Mean BGR of the middle of the box, where the jersey usually is."""

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    height = y2 - y1
    width = x2 - x1

    crop = frame[
        max(0, y1 + int(height * 0.2)):max(1, y1 + int(height * 0.6)),
        max(0, x1 + int(width * 0.25)):max(1, x1 + int(width * 0.75)),
    ]

    if crop.size == 0:
        return (128.0, 128.0, 128.0)

    return tuple(float(v) for v in crop.reshape(-1, 3).mean(axis=0))


class ColorBlobDetector:
    """
    Players are the non-green blobs standing on the green field.

    1. Take the hull of the largest green region as the field area.
    2. Keep non-green pixels inside it, minus the HUD.
    3. Open the mask to erase thin painted lines, close it vertically
       so head / jersey / legs join into one blob.
    4. Validate each blob by its size in yards (via the homography),
       split wide blobs, and keep the box of each piece.
    """

    name = "color"

    def __init__(self):
        self.last_mask = None  # candidate mask, for debugging

    def _field_region(self, green_mask):
        # Work at 1/4 resolution; only the rough outline is needed.
        height, width = green_mask.shape
        small = cv2.resize(
            green_mask,
            (max(1, width // 4), max(1, height // 4)),
            interpolation=cv2.INTER_NEAREST,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15),
        )

        closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest) * 4

        region = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(region, hull, 255)

        return region

    def _column_segments(self, roi, typical_width_pixels):
        width = roi.shape[1]

        if width <= SPLIT_RATIO * typical_width_pixels:
            return [(0, width)]

        parts = min(
            MAX_SPLIT,
            max(2, int(round(width / typical_width_pixels))),
        )

        profile = np.convolve(
            roi.sum(axis=0).astype(np.float32),
            np.ones(5, np.float32) / 5.0,
            mode="same",
        )

        step = width / parts
        cuts = [0]

        for index in range(1, parts):
            center = int(round(index * step))
            half = max(1, int(step * 0.3))

            low = max(cuts[-1] + 1, center - half)
            high = min(width - 1, center + half)

            if low >= high:
                continue

            cuts.append(low + int(np.argmin(profile[low:high + 1])))

        cuts.append(width)

        return [
            (start, end)
            for start, end in zip(cuts[:-1], cuts[1:])
            if end > start
        ]

    def detect(self, frame, homography):
        height, width = frame.shape[:2]
        scale = height / 480.0

        inverse = invert_homography(homography)

        if inverse is None:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = create_green_mask(hsv)

        region = self._field_region(green)

        if region is None:
            return []

        candidates = cv2.bitwise_and(region, cv2.bitwise_not(green))
        candidates = cv2.bitwise_and(
            candidates,
            cv2.bitwise_not(create_hud_mask(frame.shape)),
        )

        open_size = max(3, int(round(5 * scale)) | 1)

        candidates = cv2.morphologyEx(
            candidates,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (open_size, open_size),
            ),
        )

        candidates = cv2.morphologyEx(
            candidates,
            cv2.MORPH_CLOSE,
            np.ones(
                (max(3, int(round(9 * scale))), max(3, int(round(3 * scale)))),
                np.uint8,
            ),
        )

        self.last_mask = candidates

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidates,
            connectivity=8,
        )

        min_area = MIN_AREA_PIXELS * scale * scale
        detections = []

        for label in range(1, count):
            x, y, w, h, area = stats[label]

            if area < min_area or w < 2 or h < 2:
                continue

            foot = np.array([[x + w / 2.0, y + h]], dtype=np.float32)
            field_point = screen_to_field(foot, inverse)

            if not np.all(np.isfinite(field_point)):
                continue

            ppy = float(pixels_per_yard(homography, field_point)[0])

            if not np.isfinite(ppy) or ppy < 1e-3:
                continue

            roi = labels[y:y + h, x:x + w] == label

            for start, end in self._column_segments(
                roi,
                TYPICAL_WIDTH_YARDS * ppy,
            ):
                sub = roi[:, start:end]
                rows = np.flatnonzero(sub.any(axis=1))

                if rows.size == 0:
                    continue

                top, bottom = rows[0], rows[-1] + 1
                box_width = end - start
                box_height = bottom - top

                height_yards = box_height / ppy
                width_yards = box_width / ppy

                if not (MIN_HEIGHT_YARDS <= height_yards <= MAX_HEIGHT_YARDS):
                    continue

                if not (MIN_WIDTH_YARDS <= width_yards <= MAX_WIDTH_YARDS):
                    continue

                if box_height / box_width < MIN_ASPECT:
                    continue

                fill = float(sub[top:bottom].mean())

                if fill < MIN_FILL:
                    continue

                pixels = frame[
                    y + top:y + bottom,
                    x + start:x + end,
                ][sub[top:bottom]]

                detections.append(
                    Detection(
                        bbox=(
                            float(x + start),
                            float(y + top),
                            float(x + end),
                            float(y + bottom),
                        ),
                        score=fill,
                        color=tuple(float(v) for v in pixels.mean(axis=0)),
                    )
                )

        return detections


class YoloDetector:
    """
    Optional ultralytics backend. With the default COCO weights it looks
    for 'person'. With a model fine-tuned on Blitz frames, pass
    person_only=False so every class is kept.
    """

    name = "yolo"

    def __init__(self, weights="yolov8n.pt", conf=0.25, person_only=True,
                 image_size=960):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "The yolo detector needs: pip install ultralytics"
            ) from error

        self.model = YOLO(weights)
        self.conf = conf
        self.person_only = person_only
        self.image_size = image_size
        self.last_mask = None

    def detect(self, frame, homography):
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


def create_detector(name, weights="yolov8n.pt", conf=0.25,
                    person_only=True):
    if name == "color":
        return ColorBlobDetector()

    if name == "yolo":
        return YoloDetector(
            weights=weights,
            conf=conf,
            person_only=person_only,
        )

    raise ValueError(f"Unknown detector: {name!r}")