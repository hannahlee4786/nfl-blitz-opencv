"""Interactively classify manually selected jersey regions in one frame."""

import argparse
import json
from pathlib import Path

import cv2

from src.team_classifier import (
    build_color_prototype,
    classify_jersey_region,
    crop_region,
    load_calibration_report,
    prototype_distance,
)


DEFAULT_INPUT_PATH = Path("data/input/gameplay3.mp4")
DEFAULT_OUTPUT_PATH = Path("data/output/team_classification.png")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Select two team-color samples and classify manually selected "
            "jersey regions in one image or video frame."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input image or video (default: data/input/gameplay3.mp4)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based frame number when the input is a video (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Annotated output image path",
    )
    parser.add_argument("--team-a", default="TEAM A", help="First team label")
    parser.add_argument("--team-b", default="TEAM B", help="Second team label")
    parser.add_argument(
        "--min-saturation",
        type=int,
        default=55,
        help="Ignore pixels below this OpenCV HSV saturation (default: 55)",
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=35,
        help="Ignore pixels below this OpenCV HSV brightness (default: 35)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Save the result without opening a final preview window",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        help=(
            "Reuse team colors from a JSON report created by an earlier run "
            "instead of selecting new Team A and Team B samples"
        ),
    )
    return parser.parse_args()


def load_frame(input_path, frame_index):
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if frame_index < 0:
        raise ValueError("--frame must be zero or greater.")

    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {input_path}")
        return frame

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_index} from video: {input_path}"
        )

    return frame


def _selection_view(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, 1200 / width, 800 / height)

    if scale == 1.0:
        return frame.copy(), scale

    return (
        cv2.resize(frame, (round(width * scale), round(height * scale))),
        scale,
    )


def _to_original_box(box, scale, frame_shape):
    x, y, width, height = (round(float(value) / scale) for value in box)
    frame_height, frame_width = frame_shape[:2]
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = max(0, min(width, frame_width - x))
    height = max(0, min(height, frame_height - y))
    return x, y, width, height


def select_one_region(frame, title):
    displayed, scale = _selection_view(frame)
    box = cv2.selectROI(
        title,
        displayed,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(title)
    original_box = _to_original_box(box, scale, frame.shape)

    if original_box[2] == 0 or original_box[3] == 0:
        raise RuntimeError(f"Selection cancelled: {title}")

    return original_box


def select_jersey_regions(frame):
    displayed, scale = _selection_view(frame)
    title = "Select jersey regions; SPACE/ENTER adds one; ESC finishes"
    boxes = cv2.selectROIs(
        title,
        displayed,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(title)

    return [
        _to_original_box(box, scale, frame.shape)
        for box in boxes
        if box[2] > 0 and box[3] > 0
    ]


def _box_as_dict(box):
    x, y, width, height = box
    return {"x": x, "y": y, "width": width, "height": height}


def _prototype_as_dict(prototype, box=None):
    result = {
        "label": prototype.label,
        "hsv": [round(value, 2) for value in prototype.hsv],
        "bgr": list(prototype.bgr),
        "sample_pixel_count": prototype.sample_pixel_count,
    }

    if box is not None:
        result["sample_box"] = _box_as_dict(box)

    return result


def _draw_labeled_box(image, box, text, color):
    x, y, width, height = box
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    horizontal_padding = 10
    available_text_width = max(1, image.shape[1] - horizontal_padding)
    if text_width > available_text_width:
        font_scale *= available_text_width / text_width
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

    label_width = min(image.shape[1], text_width + horizontal_padding)
    label_left = min(max(0, x), image.shape[1] - label_width)
    text_top = max(0, y - text_height - baseline - 8)
    cv2.rectangle(
        image,
        (label_left, text_top),
        (label_left + label_width - 1, y),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (label_left + 5, max(text_height + 2, y - baseline - 4)),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main():
    args = parse_arguments()

    if args.calibration_report is None and args.team_a == args.team_b:
        raise ValueError("--team-a and --team-b must use different labels.")

    if not 0 <= args.min_saturation <= 255:
        raise ValueError("--min-saturation must be between 0 and 255.")

    if not 0 <= args.min_value <= 255:
        raise ValueError("--min-value must be between 0 and 255.")

    frame = load_frame(args.input, args.frame)

    team_a_box = None
    team_b_box = None

    if args.calibration_report is not None:
        team_a, team_b = load_calibration_report(args.calibration_report)
        print(f"Reusing team colors from: {args.calibration_report}")
    else:
        print(
            f"Select a tight, colorful jersey patch for {args.team_a}, "
            "then press ENTER."
        )
        team_a_box = select_one_region(
            frame, f"Select {args.team_a} color sample"
        )
        print(
            f"Select a tight, colorful jersey patch for {args.team_b}, "
            "then press ENTER."
        )
        team_b_box = select_one_region(
            frame, f"Select {args.team_b} color sample"
        )

        team_a = build_color_prototype(
            crop_region(frame, team_a_box),
            args.team_a,
            min_saturation=args.min_saturation,
            min_value=args.min_value,
        )
        team_b = build_color_prototype(
            crop_region(frame, team_b_box),
            args.team_b,
            min_saturation=args.min_saturation,
            min_value=args.min_value,
        )

    separation = prototype_distance(team_a, team_b)
    print(f"{team_a.label} HSV: {tuple(round(v, 1) for v in team_a.hsv)}")
    print(f"{team_b.label} HSV: {tuple(round(v, 1) for v in team_b.hsv)}")
    print(f"Team-color separation: {separation:.3f}")

    if separation < 0.15:
        print(
            "WARNING: The selected team colors are very similar. "
            "Try tighter sample patches if classifications are uncertain."
        )

    print()
    print("Draw a tight box around each jersey/upper torso.")
    print("Press SPACE or ENTER after each box, then ESC when all are selected.")
    jersey_boxes = select_jersey_regions(frame)

    if not jersey_boxes:
        raise RuntimeError("No jersey regions were selected.")

    annotated = frame.copy()

    if team_a_box is not None and team_b_box is not None:
        _draw_labeled_box(
            annotated, team_a_box, f"{team_a.label} SAMPLE", team_a.bgr
        )
        _draw_labeled_box(
            annotated, team_b_box, f"{team_b.label} SAMPLE", team_b.bgr
        )

    classifications = []

    for index, box in enumerate(jersey_boxes, start=1):
        result = classify_jersey_region(
            crop_region(frame, box),
            team_a,
            team_b,
            min_saturation=args.min_saturation,
            min_value=args.min_value,
        )

        color = {
            team_a.label: team_a.bgr,
            team_b.label: team_b.bgr,
        }.get(result.label, (160, 160, 160))
        confidence_percent = round(result.confidence * 100)
        coverage_percent = round(result.matched_fraction * 100)

        if result.label == "UNKNOWN":
            label_text = f"UNKNOWN | color coverage {coverage_percent}%"
        else:
            label_text = (
                f"{result.label} {confidence_percent}% | "
                f"color coverage {coverage_percent}%"
            )
        _draw_labeled_box(annotated, box, label_text, color)

        classifications.append(
            {
                "selection": index,
                "box": _box_as_dict(box),
                "label": result.label,
                "confidence": round(result.confidence, 4),
                "team_a_score": round(result.team_a_score, 4),
                "team_b_score": round(result.team_b_score, 4),
                "valid_pixel_count": result.valid_pixel_count,
                "matched_pixel_count": result.matched_pixel_count,
                "matched_fraction": round(result.matched_fraction, 4),
                "vote_agreement": round(result.vote_agreement, 4),
            }
        )
        print(f"Selection {index}: {label_text}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"Could not save output image: {args.output}")

    report_path = args.output.with_suffix(".json")
    is_image = args.input.suffix.lower() in IMAGE_EXTENSIONS
    report = {
        "input": str(args.input),
        "frame_index": None if is_image else args.frame,
        "calibration_report": (
            str(args.calibration_report)
            if args.calibration_report is not None
            else None
        ),
        "team_a": _prototype_as_dict(team_a, team_a_box),
        "team_b": _prototype_as_dict(team_b, team_b_box),
        "prototype_distance": round(separation, 4),
        "classifications": classifications,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Saved annotated frame to: {args.output}")
    print(f"Saved classification report to: {report_path}")

    if not args.no_preview:
        displayed, _ = _selection_view(annotated)
        cv2.imshow("Team classification result - press any key to close", displayed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
