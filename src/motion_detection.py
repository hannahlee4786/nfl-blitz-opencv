
"""
Homography-compensated multi-frame motion detection.

This module is a diagnostic stage for NFL Blitz.

It detects moving objects relative to the football field,
but does not assign player tracking IDs.

Run:
    python3 -m src.motion_detection gameplay3
"""

import argparse
from collections import deque

import cv2
import numpy as np

from src.field_background import FieldBackground
from src.field_mapping import invert_homography
from src.track_video import track_field, create_writer
from src.video_paths import (
    OUTPUT_DIR,
    calibration_path,
    resolve_video,
)


class MotionDetector:
    def __init__(
        self,
        frame_gap=5,
        difference_threshold=20,
        min_motion_area=70,
        min_box_width=12,
        min_box_height=18,
        max_box_area_fraction=0.035,
        blur_size=5,
        open_size=3,
        close_size=7,
        group_size=9,
        hud_top_fraction=0.16,
        hud_bottom_fraction=0.18,
    ):
        self.frame_gap = max(1, frame_gap)
        self.difference_threshold = difference_threshold
        self.min_motion_area = min_motion_area
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height
        self.max_box_area_fraction = max_box_area_fraction

        self.blur_size = blur_size
        self.open_size = open_size
        self.close_size = close_size
        self.group_size = group_size

        self.hud_top_fraction = hud_top_fraction
        self.hud_bottom_fraction = hud_bottom_fraction

        self.history = deque(maxlen=self.frame_gap + 1)

        self.last_mask = None
        self.last_difference = None
        self.last_boxes = []

    def reset(self):
        self.history.clear()
        self.last_mask = None
        self.last_difference = None
        self.last_boxes = []

    @staticmethod
    def _odd(value):
        value = max(1, int(value))
        return value if value % 2 else value + 1

    def _hud_mask(self, shape):
        """
        Exclude fixed scoreboard and interface regions.

        These are conservative diagnostic masks.
        They may also hide players near screen boundaries.
        """
        height, width = shape[:2]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        # Upper-left scoreboard.
        cv2.rectangle(
            mask,
            (0, 0),
            (
                int(width * 0.25),
                int(height * self.hud_top_fraction),
            ),
            255,
            -1,
        )

        # Bottom interface.
        bottom_start = int(
            height * (1.0 - self.hud_bottom_fraction)
        )

        cv2.rectangle(
            mask,
            (0, bottom_start),
            (width, height),
            255,
            -1,
        )

        return mask

    def _align_previous(
        self,
        previous_frame,
        previous_homography,
        current_homography,
        output_shape,
    ):
        """Align a historical frame to the current camera view."""
        height, width = output_shape[:2]

        inverse_previous = invert_homography(
            previous_homography
        )

        if inverse_previous is None:
            return None, None

        transform = (
            current_homography @ inverse_previous
        )

        if not np.all(np.isfinite(transform)):
            return None, None

        aligned = cv2.warpPerspective(
            previous_frame,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        valid_source = np.full(
            previous_frame.shape[:2],
            255,
            dtype=np.uint8,
        )

        valid = cv2.warpPerspective(
            valid_source,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return aligned, valid

    def _difference(
        self,
        current_frame,
        aligned_previous,
        valid,
    ):
        """Calculate thresholded difference between aligned frames."""
        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        previous_gray = cv2.cvtColor(
            aligned_previous,
            cv2.COLOR_BGR2GRAY,
        )

        blur = self._odd(self.blur_size)

        current_gray = cv2.GaussianBlur(
            current_gray,
            (blur, blur),
            0,
        )

        previous_gray = cv2.GaussianBlur(
            previous_gray,
            (blur, blur),
            0,
        )

        difference = cv2.absdiff(
            current_gray,
            previous_gray,
        )

        _, mask = cv2.threshold(
            difference,
            self.difference_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        mask[valid == 0] = 0

        return difference, mask

    def _clean_mask(self, mask, field_info):
        """Remove noise and restrict detections to usable field."""
        height, width = mask.shape[:2]

        hud = self._hud_mask(mask.shape)
        mask[hud > 0] = 0

        if field_info is not None:
            mask[field_info.region == 0] = 0

        # Remove isolated pixels.
        open_size = self._odd(self.open_size)

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        # Connect nearby pixels belonging to the same object.
        close_size = self._odd(self.close_size)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        # Group nearby motion fragments.
        group_size = self._odd(self.group_size)

        group_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (group_size, group_size),
        )

        grouped = cv2.dilate(
            mask,
            group_kernel,
            iterations=1,
        )

        return grouped

    def _extract_boxes(self, motion_mask):
        """Extract sufficiently large motion components."""
        height, width = motion_mask.shape[:2]

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                motion_mask,
                connectivity=8,
            )
        )

        max_area = (
            width
            * height
            * self.max_box_area_fraction
        )

        boxes = []

        for index in range(1, count):
            x, y, w, h, area = stats[index]

            if area < self.min_motion_area:
                continue

            if area > max_area:
                continue

            if w < self.min_box_width:
                continue

            if h < self.min_box_height:
                continue

            boxes.append((
                int(x),
                int(y),
                int(x + w),
                int(y + h),
            ))

        return boxes

    def detect(self, frame, homography, field_info=None):
        """
        Detect moving regions using a frame-gap comparison.

        Returns bounding boxes in current screen coordinates.
        """
        height, width = frame.shape[:2]

        self.history.append(
            (frame.copy(), homography.copy())
        )

        # Need enough history to compare frames.
        if len(self.history) <= self.frame_gap:
            self.last_mask = np.zeros(
                (height, width),
                dtype=np.uint8,
            )
            self.last_difference = self.last_mask.copy()
            self.last_boxes = []
            return []

        previous_frame, previous_homography = (
            self.history[0]
        )

        aligned, valid = self._align_previous(
            previous_frame,
            previous_homography,
            homography,
            frame.shape,
        )

        if aligned is None:
            self.last_mask = np.zeros(
                (height, width),
                dtype=np.uint8,
            )
            self.last_difference = self.last_mask.copy()
            self.last_boxes = []
            return []

        difference, motion_mask = self._difference(
            frame,
            aligned,
            valid,
        )

        self.last_difference = difference

        motion_mask = self._clean_mask(
            motion_mask,
            field_info,
        )

        self.last_mask = motion_mask

        boxes = self._extract_boxes(
            motion_mask
        )

        self.last_boxes = boxes

        return boxes


def create_debug_frame(frame, motion_mask, boxes):
    """Show gameplay and motion mask side by side."""
    height, width = frame.shape[:2]

    annotated = frame.copy()

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

    cv2.putText(
        annotated,
        f"Motion candidates: {len(boxes)}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    if motion_mask is None:
        motion_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    mask_bgr = cv2.cvtColor(
        motion_mask,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.putText(
        mask_bgr,
        "Homography-compensated motion",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return np.hstack([
        annotated,
        mask_bgr,
    ])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-frame motion detection for NFL Blitz."
    )

    parser.add_argument(
        "video",
        nargs="?",
        default="gameplay3",
    )

    parser.add_argument(
        "--frame-gap",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--difference-threshold",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-motion-area",
        type=int,
        default=70,
    )

    parser.add_argument(
        "--min-box-width",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-box-height",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=9,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_path = resolve_video(args.video)
    initialization_path = calibration_path(
        input_path
    )

    if not initialization_path.exists():
        raise RuntimeError(
            f"No calibration found for {input_path.name}."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{input_path.stem}_motion_debug.mp4"
    )

    detector = MotionDetector(
        frame_gap=args.frame_gap,
        difference_threshold=args.difference_threshold,
        min_motion_area=args.min_motion_area,
        min_box_width=args.min_box_width,
        min_box_height=args.min_box_height,
        group_size=args.group_size,
    )

    background = FieldBackground()
    writer = None

    try:
        for state in track_field(
            input_path,
            initialization_path,
        ):
            if state.confidence == "LOW":
                detector.reset()
                boxes = []
                motion_mask = np.zeros(
                    state.frame.shape[:2],
                    dtype=np.uint8,
                )
            else:
                field_info = background.analyze(
                    state.frame,
                    state.homography,
                )

                boxes = detector.detect(
                    state.frame,
                    state.homography,
                    field_info,
                )

                motion_mask = detector.last_mask

            debug_frame = create_debug_frame(
                state.frame,
                motion_mask,
                boxes,
            )

            if writer is None:
                height, width = debug_frame.shape[:2]

                writer = create_writer(
                    output_path,
                    state.fps,
                    (width, height),
                )

            writer.write(debug_frame)

    finally:
        if writer is not None:
            writer.release()

    print(f"Saved motion debug video: {output_path}")


if __name__ == "__main__":
    main()
