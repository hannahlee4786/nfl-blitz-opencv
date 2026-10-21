
"""YOLO + ByteTrack player tracking with existing field homography.

Run:
    python -m src.track_players_yolo gameplay3
"""

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.field_mapping import (
    invert_homography,
    screen_to_field,
    upright_scale,
)
from src.field_model import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    create_field_grid,
)
from src.jersey_color import JerseyClassifier, JerseyConfig
from src.player_visualization import (
    compose_side_by_side,
    team_color,
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


DEFAULT_MODEL = (
    "runs/detect/runs/nfl_blitz/"
    "player_detector_v1/weights/best.pt"
)


@dataclass
class PlayerTrack:
    id: int
    bbox: tuple
    last_foot: tuple
    position: tuple | None
    team: str | None = None
    matched: bool = True
    confidence: float = 0.0
    history: list = field(default_factory=list)


class TeamMemory:
    """Smooth red/blue jersey classification by tracking ID."""

    def __init__(self, window=12):
        self.scores = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.teams = {}

    def update(self, track_id, jersey_result):
        if (
            jersey_result.level is not None
            and jersey_result.team in ("red", "blue")
        ):
            weight = (
                2.0 if jersey_result.level == "strong"
                else 1.0
            )

            self.scores[track_id].append(
                (jersey_result.team, weight)
            )

        red = sum(
            weight
            for team, weight in self.scores[track_id]
            if team == "red"
        )

        blue = sum(
            weight
            for team, weight in self.scores[track_id]
            if team == "blue"
        )

        previous = self.teams.get(track_id)

        if previous is None:
            if red > blue:
                previous = "red"
            elif blue > red:
                previous = "blue"

        elif previous == "red" and blue > red * 1.5:
            previous = "blue"

        elif previous == "blue" and red > blue * 1.5:
            previous = "red"

        self.teams[track_id] = previous
        return previous


def draw_tracks(frame, tracks):
    """Draw thin team-colored boxes with small tracking IDs."""

    result = frame.copy()

    for track in tracks:
        if not track.matched:
            continue

        color = team_color(track)

        x1, y1, x2, y2 = [
            int(round(value))
            for value in track.bbox
        ]

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            1,
        )

        label = f"ID {track.id}"

        cv2.putText(
            result,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return result


def get_yolo_tracks(model, frame, confidence, iou):
    """Return ByteTrack detections with persistent IDs."""

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=confidence,
        iou=iou,
        verbose=False,
    )

    detections = []

    for result in results:
        boxes = result.boxes

        if boxes is None or boxes.id is None:
            continue

        for box in boxes:
            track_id = int(box.id.item())
            confidence_score = float(box.conf.item())

            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy().tolist()
            )

            detections.append({
                "id": track_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence_score,
            })

    return detections


def process_video(
    input_path,
    initialization_path,
    output_path,
    arguments,
):
    # Create a fresh tracker for each video.
    model = YOLO(str(arguments.weights))

    jersey_classifier = JerseyClassifier(
        JerseyConfig()
    )

    team_memory = TeamMemory()

    histories = defaultdict(
        lambda: deque(maxlen=30)
    )

    last_positions = {}
    last_position_frames = {}
    last_seen = {}
    last_tracks = {}

    grid = create_field_grid()

    topdown = None
    writer = None

    csv_path = output_path.with_suffix(".csv")

    try:
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
                "confidence",
            ])

            for state in track_field(
                input_path,
                initialization_path,
            ):
                frame = state.frame
                height, width = frame.shape[:2]

                # Run YOLO + ByteTrack on every frame.
                detections = get_yolo_tracks(
                    model,
                    frame,
                    arguments.conf,
                    arguments.iou,
                )

                inverse = invert_homography(
                    state.homography
                )

                mapping_valid = (
                    inverse is not None
                    and state.confidence != "LOW"
                )

                hsv = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2HSV,
                )

                tracks = []
                fresh_mappings = 0

                for detection in detections:
                    track_id = detection["id"]

                    x1, y1, x2, y2 = detection["bbox"]

                    # Reject invalid bounding boxes.
                    if (
                        x2 <= x1
                        or y2 <= y1
                        or x1 >= width
                        or y1 >= height
                        or x2 <= 0
                        or y2 <= 0
                    ):
                        continue

                    bbox = (
                        max(0.0, x1),
                        max(0.0, y1),
                        min(float(width - 1), x2),
                        min(float(height - 1), y2),
                    )

                    # Approximate ground contact at box bottom.
                    foot = (
                        (bbox[0] + bbox[2]) / 2.0,
                        bbox[3],
                    )

                    position = None

                    # Fallback estimate for jersey scale.
                    scale = max(
                        1.0,
                        (bbox[3] - bbox[1]) / 2.5,
                    )

                    if mapping_valid:
                        mapped = screen_to_field(
                            [foot],
                            inverse,
                        )[0]

                        if (
                            np.all(np.isfinite(mapped))
                            and -1 <= mapped[0] <= FIELD_LENGTH + 1
                            and -1 <= mapped[1] <= FIELD_WIDTH + 1
                        ):
                            position = (
                                float(mapped[0]),
                                float(mapped[1]),
                            )

                            estimated_scale = upright_scale(
                                state.homography,
                                [position],
                            )[0]

                            if (
                                np.isfinite(estimated_scale)
                                and estimated_scale > 0
                            ):
                                scale = float(estimated_scale)

                    jersey_result = jersey_classifier.analyze(
                        hsv,
                        bbox,
                        scale,
                    )

                    team = team_memory.update(
                        track_id,
                        jersey_result,
                    )

                    # Update cached position only when mapping
                    # succeeds in the current frame.
                    if position is not None:
                        last_positions[track_id] = position
                        last_position_frames[track_id] = (
                            state.frame_index
                        )
                        histories[track_id].append(position)
                        fresh_mappings += 1

                    # Use a recent mapped position if this frame
                    # has unreliable homography.
                    render_position = None

                    if track_id in last_positions:
                        position_age = (
                            state.frame_index
                            - last_position_frames[track_id]
                        )

                        if position_age <= arguments.missing_frames:
                            render_position = last_positions[track_id]

                    # Always retain the YOLO detection for gameplay,
                    # even when no valid field position exists.
                    track = PlayerTrack(
                        id=track_id,
                        bbox=bbox,
                        last_foot=foot,
                        position=render_position,
                        team=team,
                        matched=True,
                        confidence=detection["confidence"],
                        history=list(histories[track_id]),
                    )

                    # Add each tracked player exactly once.
                    tracks.append(track)

                    last_tracks[track_id] = track
                    last_seen[track_id] = state.frame_index

                    # Export only fresh mapped coordinates.
                    if position is not None:
                        csv_writer.writerow([
                            state.frame_index,
                            track_id,
                            f"{position[0]:.3f}",
                            f"{position[1]:.3f}",
                            f"{foot[0]:.1f}",
                            f"{foot[1]:.1f}",
                            team or "",
                            f"{detection['confidence']:.3f}",
                        ])

                active_ids = {
                    track.id for track in tracks
                }

                # Display recently missing players as hollow
                # markers, but only when their field positions
                # are still recent enough.
                for track_id, previous in list(
                    last_tracks.items()
                ):
                    if track_id in active_ids:
                        continue

                    age = (
                        state.frame_index
                        - last_seen.get(track_id, -99999)
                    )

                    position_age = (
                        state.frame_index
                        - last_position_frames.get(
                            track_id, -99999
                        )
                    )

                    if (
                        0 < age <= arguments.missing_frames
                        and position_age <= arguments.missing_frames
                        and track_id in last_positions
                    ):
                        tracks.append(
                            PlayerTrack(
                                id=track_id,
                                bbox=previous.bbox,
                                last_foot=previous.last_foot,
                                position=last_positions[track_id],
                                team=previous.team,
                                matched=False,
                                history=list(histories[track_id]),
                            )
                        )

                # Separate gameplay and field visualization.
                gameplay_tracks = [
                    track for track in tracks
                    if track.matched
                ]

                field_tracks = [
                    track for track in tracks
                    if track.position is not None
                ]

                gameplay = draw_field_grid(
                    frame.copy(),
                    grid,
                    state.homography,
                )

                gameplay = draw_tracking_status(
                    gameplay,
                    state.frame_index,
                    state.point_count,
                    state.confidence,
                )

                status = (
                    f"Tracked: {len(gameplay_tracks)}"
                    f" | Mapped now: {fresh_mappings}"
                    f" | Field markers: {len(field_tracks)}"
                )

                cv2.putText(
                    gameplay,
                    status,
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                gameplay = draw_tracks(
                    gameplay,
                    gameplay_tracks,
                )

                if topdown is None:
                    orientation = arguments.orientation

                    if orientation == "auto":
                        orientation = auto_orientation(
                            state.homography,
                            state.size,
                        )

                    topdown = TopDownField(
                        int(
                            state.size[1]
                            * arguments.panel_height
                        ),
                        orientation=orientation,
                        trail=arguments.trails,
                    )

                panel = topdown.render(field_tracks)

                combined = compose_side_by_side(
                    gameplay,
                    panel,
                )

                if writer is None:
                    output_height, output_width = (
                        combined.shape[:2]
                    )

                    writer = create_writer(
                        output_path,
                        state.fps or 30,
                        (output_width, output_height),
                    )

                writer.write(combined)

    finally:
        if writer is not None:
            writer.release()

    print(f"Saved video: {output_path}")
    print(f"Saved CSV: {csv_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="YOLO + ByteTrack NFL Blitz field tracking."
    )

    parser.add_argument(
        "videos",
        nargs="*",
        help="Video names or paths. Defaults to calibrated videos.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(DEFAULT_MODEL),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--orientation",
        choices=["auto", "camera", "cw", "ccw", "flip"],
        default="auto",
    )

    parser.add_argument(
        "--panel-height",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--missing-frames",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--trails",
        action="store_true",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not arguments.weights.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {arguments.weights}"
        )

    if arguments.videos:
        input_paths = [
            resolve_video(name)
            for name in arguments.videos
        ]
    else:
        input_paths = [
            video
            for video in list_input_videos()
            if calibration_path(video).exists()
        ]

    if not input_paths:
        raise RuntimeError(
            "No calibrated videos found. "
            "Run src.initialize_homography first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for input_path in input_paths:
        initialization_path = calibration_path(
            input_path
        )

        if not initialization_path.exists():
            print(
                f"Skipping {input_path.name}: "
                "no homography calibration found."
            )
            continue

        output_path = (
            OUTPUT_DIR
            / f"{input_path.stem}_yolo_players.mp4"
        )

        print(f"Processing {input_path.name}...")

        process_video(
            input_path,
            initialization_path,
            output_path,
            arguments,
        )


if __name__ == "__main__":
    main()
