# Runs both detectors on the calibration frame so you can judge them
# before trusting either one on Blitz footage.
import argparse

import cv2
import numpy as np

from src.field_mapping import attach_field_positions, invert_homography
from src.player_detection import create_detector
from src.video_paths import OUTPUT_DIR, calibration_path, resolve_video


def main():
    parser = argparse.ArgumentParser(
        description="Compare player detectors on the calibration frame.",
    )
    parser.add_argument("video")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--all-classes", action="store_true")
    arguments = parser.parse_args()

    input_path = resolve_video(arguments.video)
    calibration = np.load(calibration_path(input_path))

    homography = calibration["homography"].astype(np.float64)
    inverse = invert_homography(homography)

    capture = cv2.VideoCapture(str(input_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(calibration["frame_index"]))
    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError("Could not read the calibration frame.")

    panels = []

    for name in ("color", "yolo"):
        try:
            detector = create_detector(
                name,
                weights=arguments.weights,
                conf=arguments.conf,
                person_only=not arguments.all_classes,
            )

            detections = attach_field_positions(
                detector.detect(frame, homography),
                inverse,
            )
        except Exception as error:
            print(f"{name}: skipped ({error})")
            continue

        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(
            annotated,
            f"{name}: {len(detections)} players on field",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        print(f"{name}: {len(detections)} detections on the field")
        panels.append(annotated)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_path.stem}_detector_comparison.png"
    cv2.imwrite(str(output_path), np.hstack(panels))
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()