
"""NFL Blitz player detection, tracking, and top-down visualization."""

import argparse
import csv

import cv2
import numpy as np

from src.detection_filters import FilterConfig
from src.field_mapping import invert_homography
from src.field_model import create_field_grid
from src.jersey_color import JerseyConfig
from src.player_detection import create_detector
from src.player_tracking import PlayerTracker
from src.player_visualization import (
    compose_side_by_side,
    draw_player_tracks,
    draw_rejected,
)
from src.topdown_view import TopDownField, auto_orientation
from src.track_video import create_writer, track_field
from src.video_paths import (
    OUTPUT_DIR,
    calibration_path,
    list_input_videos,
    resolve_video,
)
from src.visualization import (
    draw_field_grid,
    draw_tracking_status,
)


def overlay_mask(frame, mask):
    if mask is None:
        return frame

    result = frame.copy()
    tint = np.array([255, 255, 0], dtype=np.float32)

    selected = mask > 0

    result[selected] = (
        0.5 * result[selected].astype(np.float32)
        + 0.5 * tint
    ).astype(np.uint8)

    return result


def process_video(
    input_path,
    initialization_path,
    output_path,
    detector,
    arguments,
):
    tracker = PlayerTracker(
        max_age=arguments.max_age,
        min_hits=arguments.min_hits,
        gate_yards=arguments.gate,
        min_quality=arguments.min_quality,
        max_players=arguments.max_players,
        team_decay=arguments.team_decay,
        team_switch_ratio=arguments.team_switch,
        tentative_max_age=arguments.tentative_max_age,
        iou_weight=arguments.iou_weight,
        max_match_cost=arguments.max_match_cost,
        min_detection_quality=arguments.min_detection_quality,
        entry_margin_fraction=arguments.entry_margin,
        initialization_frames=arguments.initialization_frames,
        max_initial_players=arguments.max_initial_players,
    )

    detector.reset()

    csv_path = output_path.with_suffix(".csv")
    topdown = None
    writer = None
    grid = create_field_grid()

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame",
            "id",
            "field_x",
            "field_y",
            "screen_x",
            "screen_y",
            "team",
        ])

        try:
            for state in track_field(
                input_path,
                initialization_path,
            ):
                inverse = invert_homography(
                    state.homography
                )

                detections = []
                detector.clear()

                if (
                    inverse is not None
                    and state.confidence != "LOW"
                ):
                    detections = detector.detect(
                        state.frame,
                        state.homography,
                    )

                tracks = tracker.update(
                    detections,
                    frame_shape=state.frame.shape,
                )

                gameplay = state.frame

                if arguments.debug_mask:
                    gameplay = overlay_mask(
                        gameplay,
                        detector.last_mask,
                    )

                gameplay = draw_field_grid(
                    gameplay,
                    grid,
                    state.homography,
                )

                gameplay = draw_tracking_status(
                    gameplay,
                    state.frame_index,
                    state.point_count,
                    state.confidence,
                )

                if arguments.debug_rejections:
                    gameplay = draw_rejected(
                        gameplay,
                        detector.rejected,
                    )

                gameplay = draw_player_tracks(
                    gameplay,
                    tracks,
                )

                if topdown is None:
                    orientation = arguments.orientation

                    if orientation == "auto":
                        orientation = auto_orientation(
                            state.homography,
                            state.size,
                        )

                    topdown = TopDownField(
                        int(state.size[1] * arguments.panel_height),
                        orientation=orientation,
                        trail=False,
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
                    if not track.matched:
                        continue

                    csv_writer.writerow([
                        state.frame_index,
                        track.id,
                        f"{track.position[0]:.3f}",
                        f"{track.position[1]:.3f}",
                        f"{track.last_foot[0]:.1f}",
                        f"{track.last_foot[1]:.1f}",
                        track.team or "",
                    ])

        finally:
            if writer is not None:
                writer.release()

    print(f"Saved visualization to: {output_path}")
    print(f"Saved player positions to: {csv_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Track NFL Blitz players."
    )

    parser.add_argument("videos", nargs="*")

    # Detector
    parser.add_argument(
        "--detector",
        choices=["color", "yolo"],
        default="color",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
    )

    # Detection filtering
    parser.add_argument(
        "--min-height",
        type=float,
        default=1.6,
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--max-width",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-white",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.25,
    )

    # Tracking
    parser.add_argument(
        "--min-hits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=45,
    )
    parser.add_argument(
        "--tentative-max-age",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--min-detection-quality",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--iou-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-match-cost",
        type=float,
        default=1.5,
    )

    # Controlled spawning
    parser.add_argument(
        "--entry-margin",
        type=float,
        default=0.08,
        help="Fraction of screen width eligible for new tracks.",
    )
    parser.add_argument(
        "--initialization-frames",
        type=int,
        default=5,
        help="Frames during which new tracks may start anywhere.",
    )
    parser.add_argument(
        "--max-initial-players",
        type=int,
        default=14,
        help="Maximum initial player candidates.",
    )

    # Jersey filtering
    parser.add_argument(
        "--no-jersey-filter",
        action="store_true",
    )
    parser.add_argument(
        "--jersey-strong",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--jersey-weak",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--jersey-min-area",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--jersey-max",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--blue-hue-min",
        type=int,
        default=104,
    )
    parser.add_argument(
        "--blue-hue-max",
        type=int,
        default=132,
    )
    parser.add_argument(
        "--red-sat",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--blue-sat",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--jersey-min-value",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--team-decay",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--team-switch",
        type=float,
        default=1.5,
    )

    # Visualization
    parser.add_argument(
        "--panel-height",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--debug-mask",
        action="store_true",
    )
    parser.add_argument(
        "--debug-rejections",
        action="store_true",
    )
    parser.add_argument(
        "--orientation",
        choices=["auto", "camera", "cw", "ccw", "flip"],
        default="auto",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.videos:
        input_paths = [
            resolve_video(video)
            for video in arguments.videos
        ]
    else:
        input_paths = [
            video for video in list_input_videos()
            if calibration_path(video).exists()
        ]

    if not input_paths:
        raise RuntimeError(
            "No calibrated videos found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filter_config = FilterConfig(
        min_height=arguments.min_height,
        max_height=arguments.max_height,
        min_width=arguments.min_width,
        max_width=arguments.max_width,
        field_margin=arguments.margin,
        max_white_fraction=arguments.max_white,
        min_ground_support=arguments.min_support,
        use_jersey_filter=not arguments.no_jersey_filter,
        jersey=JerseyConfig(
            strong_fraction=arguments.jersey_strong,
            weak_fraction=arguments.jersey_weak,
            min_area_yards2=arguments.jersey_min_area,
            max_fraction=arguments.jersey_max,
            blue_hue_min=arguments.blue_hue_min,
            blue_hue_max=arguments.blue_hue_max,
            red_min_saturation=arguments.red_sat,
            blue_min_saturation=arguments.blue_sat,
            min_value=arguments.jersey_min_value,
        ),
    )

    detector = create_detector(
        arguments.detector,
        weights=arguments.weights,
        conf=arguments.conf,
        person_only=not arguments.all_classes,
        filter_config=filter_config,
    )

    for input_path in input_paths:
        initialization_path = calibration_path(
            input_path
        )

        if not initialization_path.exists():
            print(
                f"Skipping {input_path.name}: "
                "no calibration found."
            )
            continue

        output_path = (
            OUTPUT_DIR
            / f"{input_path.stem}_players.mp4"
        )

        print(f"Tracking {input_path.name}...")

        process_video(
            input_path,
            initialization_path,
            output_path,
            detector,
            arguments,
        )


if __name__ == "__main__":
    main()
