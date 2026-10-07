# Field tracking + player detection + player tracking + top-down view
import argparse
import csv

import cv2
import numpy as np

from src.field_mapping import attach_field_positions, invert_homography
from src.field_model import create_field_grid
from src.player_detection import create_detector
from src.player_tracking import PlayerTracker
from src.player_visualization import compose_side_by_side, draw_player_tracks
from src.topdown_view import TopDownField
from src.track_video import create_writer, track_field
from src.video_paths import (
    OUTPUT_DIR,
    calibration_path,
    list_input_videos,
    resolve_video,
)
from src.visualization import draw_field_grid, draw_tracking_status


def overlay_mask(frame, mask):
    """Tint the detector's candidate pixels (for tuning thresholds)."""

    if mask is None:
        return frame

    result = frame.copy()
    tint = np.array([255, 255, 0], dtype=np.float32)

    result[mask > 0] = (
        0.5 * result[mask > 0].astype(np.float32) + 0.5 * tint
    ).astype(np.uint8)

    return result


def process_video(input_path, initialization_path, output_path, detector,
                  arguments):
    tracker = PlayerTracker(
        max_age=arguments.max_age,
        min_hits=arguments.min_hits,
        gate_yards=arguments.gate,
    )

    csv_path = output_path.with_suffix(".csv")
    topdown = None
    writer = None
    grid = create_field_grid()

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["frame", "id", "field_x", "field_y", "screen_x", "screen_y"]
        )

        try:
            for state in track_field(input_path, initialization_path):
                inverse = invert_homography(state.homography)

                detections = []

                # Do not feed detections through a field estimate we do
                # not trust; tracks coast until the field is good again.
                if inverse is not None and state.confidence != "LOW":
                    detections = attach_field_positions(
                        detector.detect(state.frame, state.homography),
                        inverse,
                    )

                tracks = tracker.update(detections)

                gameplay = state.frame

                if arguments.debug_mask:
                    gameplay = overlay_mask(gameplay, detector.last_mask)

                gameplay = draw_field_grid(gameplay, grid, state.homography)
                gameplay = draw_tracking_status(
                    gameplay,
                    state.frame_index,
                    state.point_count,
                    state.confidence,
                )
                gameplay = draw_player_tracks(gameplay, tracks)

                if topdown is None:
                    topdown = TopDownField(
                        int(state.size[0] * arguments.panel_width),
                    )

                combined = compose_side_by_side(
                    gameplay,
                    topdown.render(tracks),
                )

                if writer is None:
                    height, width = combined.shape[:2]
                    writer = create_writer(
                        output_path,
                        state.fps,
                        (width, height),
                    )

                writer.write(combined)

                for track in tracks:
                    if track.matched:
                        csv_writer.writerow(
                            [
                                state.frame_index,
                                track.id,
                                f"{track.position[0]:.3f}",
                                f"{track.position[1]:.3f}",
                                f"{track.last_foot[0]:.1f}",
                                f"{track.last_foot[1]:.1f}",
                            ]
                        )
        finally:
            if writer is not None:
                writer.release()

    print(f"Saved visualization to: {output_path}")
    print(f"Saved player positions to: {csv_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Track players and map them to a top-down field.",
    )

    parser.add_argument(
        "videos",
        nargs="*",
        help="Video paths, or the start of file names in data/input. "
        "Leave empty to process every calibrated video.",
    )

    parser.add_argument("--detector", choices=["color", "yolo"],
                        default="color")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="YOLO weights (only for --detector yolo).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold.")
    parser.add_argument("--all-classes", action="store_true",
                        help="Keep every YOLO class (fine-tuned models).")

    parser.add_argument("--min-hits", type=int, default=3,
                        help="Consecutive matches before a track gets an ID.")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Frames a missed track keeps coasting.")
    parser.add_argument("--gate", type=float, default=2.0,
                        help="Base matching distance, in yards.")

    parser.add_argument("--panel-width", type=float, default=1.5,
                        help="Top-down panel width as a multiple of the "
                        "gameplay width.")
    parser.add_argument("--debug-mask", action="store_true",
                        help="Tint the color detector's candidate pixels.")

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.videos:
        input_paths = [resolve_video(v) for v in arguments.videos]
    else:
        input_paths = [
            video for video in list_input_videos()
            if calibration_path(video).exists()
        ]

    if not input_paths:
        raise RuntimeError(
            "No calibrated videos. "
            "Run python -m src.initialize_homography first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = create_detector(
        arguments.detector,
        weights=arguments.weights,
        conf=arguments.conf,
        person_only=not arguments.all_classes,
    )

    for input_path in input_paths:
        initialization_path = calibration_path(input_path)

        if not initialization_path.exists():
            print(f"Skipping {input_path.name}: no calibration.")
            continue

        print(f"Tracking players in {input_path.name}...")

        process_video(
            input_path,
            initialization_path,
            OUTPUT_DIR / f"{input_path.stem}_players.mp4",
            detector,
            arguments,
        )


if __name__ == "__main__":
    main()