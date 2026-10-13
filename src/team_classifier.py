"""HSV-based jersey color calibration and team classification."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorPrototype:
    """A representative jersey color measured from a sample region."""

    label: str
    hsv: tuple[float, float, float]
    bgr: tuple[int, int, int]
    sample_pixel_count: int


@dataclass(frozen=True)
class ClassificationResult:
    """The team prediction and supporting color-vote statistics."""

    label: str
    confidence: float
    team_a_score: float
    team_b_score: float
    valid_pixel_count: int
    matched_pixel_count: int
    matched_fraction: float
    vote_agreement: float


def load_calibration_report(report_path):
    """Load the two team-color prototypes saved by ``classify_teams``."""

    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(
            f"Calibration report does not exist: {report_path}"
        )

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        prototypes = []

        for key in ("team_a", "team_b"):
            saved = report[key]
            prototypes.append(
                ColorPrototype(
                    label=str(saved["label"]),
                    hsv=tuple(float(value) for value in saved["hsv"]),
                    bgr=tuple(int(value) for value in saved["bgr"]),
                    sample_pixel_count=int(saved["sample_pixel_count"]),
                )
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid calibration report: {report_path}"
        ) from error

    return tuple(prototypes)


def crop_region(frame, box):
    """Return a clipped image region for an (x, y, width, height) box."""

    x, y, width, height = (int(value) for value in box)
    frame_height, frame_width = frame.shape[:2]

    left = max(0, x)
    top = max(0, y)
    right = min(frame_width, x + width)
    bottom = min(frame_height, y + height)

    if right <= left or bottom <= top:
        raise ValueError(f"Selection is outside the frame: {box}")

    return frame[top:bottom, left:right]


def extract_color_pixels(
    bgr_region,
    min_saturation=55,
    min_value=35,
):
    """Return colorful, visible HSV pixels from a selected jersey region."""

    if bgr_region is None or bgr_region.size == 0:
        raise ValueError("The selected jersey region is empty.")

    hsv_region = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)

    # This removes white numbers, gray pants, deep shadows, and most black
    # pixels so they do not overwhelm the jersey color.
    color_mask = cv2.inRange(
        hsv_region,
        np.array([0, min_saturation, min_value], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    return hsv_region[color_mask > 0].astype(np.float32)


def _circular_mean_hue(hues):
    """Average OpenCV hues while respecting the red 179-to-0 wraparound."""

    angles = hues * (2.0 * np.pi / 180.0)
    mean_angle = np.arctan2(
        np.sin(angles).mean(),
        np.cos(angles).mean(),
    )

    return float((mean_angle % (2.0 * np.pi)) * 180.0 / (2.0 * np.pi))


def _hsv_to_bgr(hsv):
    hsv_pixel = np.array(
        [[[round(hsv[0]) % 180, round(hsv[1]), round(hsv[2])]]],
        dtype=np.uint8,
    )
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr_pixel)


def build_color_prototype(
    bgr_sample,
    label,
    min_saturation=55,
    min_value=35,
    min_pixels=20,
):
    """Measure a robust representative HSV color from a jersey sample."""

    pixels = extract_color_pixels(
        bgr_sample,
        min_saturation=min_saturation,
        min_value=min_value,
    )

    if len(pixels) < min_pixels:
        raise ValueError(
            f"{label} sample has only {len(pixels)} usable color pixels. "
            "Select a tighter, brighter, more colorful jersey patch."
        )

    hsv = (
        _circular_mean_hue(pixels[:, 0]),
        float(np.median(pixels[:, 1])),
        float(np.median(pixels[:, 2])),
    )

    return ColorPrototype(
        label=label,
        hsv=hsv,
        bgr=_hsv_to_bgr(hsv),
        sample_pixel_count=len(pixels),
    )


def _color_distances(hsv_pixels, prototype):
    """Calculate normalized perceptual-ish HSV distances to a prototype."""

    hue_difference = np.abs(hsv_pixels[:, 0] - prototype.hsv[0])
    hue_difference = np.minimum(hue_difference, 180.0 - hue_difference) / 90.0
    saturation_difference = (
        np.abs(hsv_pixels[:, 1] - prototype.hsv[1]) / 255.0
    )
    value_difference = np.abs(hsv_pixels[:, 2] - prototype.hsv[2]) / 255.0

    return np.sqrt(
        0.75 * np.square(hue_difference)
        + 0.15 * np.square(saturation_difference)
        + 0.10 * np.square(value_difference)
    )


def prototype_distance(first, second):
    """Return the HSV distance between two calibrated team colors."""

    pixels = np.array([first.hsv], dtype=np.float32)
    return float(_color_distances(pixels, second)[0])


def classify_jersey_region(
    bgr_region,
    team_a,
    team_b,
    min_saturation=55,
    min_value=35,
    max_color_distance=0.28,
    min_vote_agreement=0.60,
    min_matched_pixels=20,
    min_matched_fraction=0.10,
):
    """Classify a selected or automatically detected jersey region."""

    pixels = extract_color_pixels(
        bgr_region,
        min_saturation=min_saturation,
        min_value=min_value,
    )

    if len(pixels) == 0:
        return ClassificationResult(
            label="UNKNOWN",
            confidence=0.0,
            team_a_score=0.0,
            team_b_score=0.0,
            valid_pixel_count=0,
            matched_pixel_count=0,
            matched_fraction=0.0,
            vote_agreement=0.0,
        )

    team_a_distances = _color_distances(pixels, team_a)
    team_b_distances = _color_distances(pixels, team_b)
    nearest_distances = np.minimum(team_a_distances, team_b_distances)
    matched = nearest_distances <= max_color_distance
    matched_count = int(matched.sum())
    matched_fraction = matched_count / len(pixels)

    if matched_count == 0:
        team_a_votes = 0
        team_b_votes = 0
        team_a_score = 0.0
        team_b_score = 0.0
        vote_agreement = 0.0
    else:
        team_a_votes = int(
            (matched & (team_a_distances < team_b_distances)).sum()
        )
        team_b_votes = matched_count - team_a_votes
        # Scores use every usable color pixel as the denominator. Unmatched
        # pixels therefore reduce support instead of being silently ignored.
        team_a_score = team_a_votes / len(pixels)
        team_b_score = team_b_votes / len(pixels)
        vote_agreement = max(team_a_votes, team_b_votes) / matched_count

    if (
        matched_count < min_matched_pixels
        or matched_fraction < min_matched_fraction
        or vote_agreement < min_vote_agreement
    ):
        label = "UNKNOWN"
        confidence = 0.0
    elif team_a_score > team_b_score:
        label = team_a.label
        confidence = vote_agreement
    else:
        label = team_b.label
        confidence = vote_agreement

    return ClassificationResult(
        label=label,
        confidence=confidence,
        team_a_score=team_a_score,
        team_b_score=team_b_score,
        valid_pixel_count=len(pixels),
        matched_pixel_count=matched_count,
        matched_fraction=matched_fraction,
        vote_agreement=vote_agreement,
    )
