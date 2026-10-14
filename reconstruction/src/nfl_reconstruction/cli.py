"""Command-line entry points for the independent reconstruction workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import typer

from .annotations import anchor_homography, load_anchor
from .camera_solve import (
    describe_solution,
    export_pose_json,
    pose_to_state,
    solve_camera,
)
from .detection import UltralyticsDetector, export_yolo_dataset
from .evaluation import evaluate_run
from .evidence import build_exclusion_mask
from .field import NFLField
from .grid_tracking import measure_line_errors, track_grid
from .line_score import get_tracking_status
from .mosaic import build_mosaic
from .pipeline import run_pipeline
from .preprocessing import find_active_area
from .schemas import (
    CameraFrame,
    DetectionRecord,
    FrameRecord,
    RunManifest,
    TrackRecord,
    VideoInfo,
)
from .tracking import track_players
from .video import build_manifest, probe_video
from .visualization import render_minimap, render_overlay

app = typer.Typer(help="Reconstruct a static NFL field from a moving-camera clip.")

# How far later stages may trust a camera, by its line-error rating. LOST
# frames fall below every camera confidence threshold.
LINE_CONFIDENCE = {"GOOD": 1.0, "FAIR": 0.5, "LOST": 0.0}


def _parse_frame_range(value: str | None, default: str) -> range:
    raw = value or default
    try:
        start_text, end_text = raw.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, AttributeError) as error:
        raise typer.BadParameter("frame range must use START:END") from error
    if start < 0 or end <= start:
        raise typer.BadParameter("frame range must be non-empty and non-negative")
    return range(start, end)


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (config_path.parent / candidate).resolve()
    return candidate


def _read_jsonl(path: Path, model):
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, records) -> None:
    path.write_text("".join(record.model_dump_json() + "\n" for record in records))


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _describe_errors(errors: np.ndarray) -> dict[str, float | None]:
    measured = errors[~np.isnan(errors)]
    if not measured.size:
        return {"median": None, "p95": None}
    return {
        "median": float(np.median(measured)),
        "p95": float(np.percentile(measured, 95)),
    }


class ReconstructionRunner:
    """Concrete filesystem stages used by the CLI orchestrator."""

    def prepare(self, output_dir: Path, config: dict) -> None:
        selected = range(*config["frame_range"])
        manifest = build_manifest(config["input"], output_dir / "frames", selected)
        final_frames = Path(config["run_dir"]) / "prepare" / "frames"
        records = tuple(
            record.model_copy(update={"path": str(final_frames / Path(record.path).name)})
            for record in manifest.frames
        )
        manifest.model_copy(update={"frames": records}).model_dump_json(indent=2)
        (output_dir / "manifest.json").write_text(
            manifest.model_copy(update={"frames": records}).model_dump_json(indent=2)
        )

    def camera(self, output_dir: Path, config: dict) -> None:
        run_dir = Path(config["run_dir"])
        prepared = RunManifest.model_validate_json(
            (run_dir / "prepare" / "manifest.json").read_text()
        )
        frames = [np.asarray(Image.open(record.path).convert("RGB")) for record in prepared.frames]
        if not frames:
            raise RuntimeError("Camera stage requires prepared frames.")
        image_size = (frames[0].shape[1], frames[0].shape[0])
        anchor = load_anchor(config["anchor"], image_size)
        indices = [record.index for record in prepared.frames]
        if anchor.frame not in indices:
            raise ValueError("Selected frame range must contain the manual anchor frame.")

        active_area = find_active_area(frames)
        rectangles = [tuple(item) for item in config.get("hud_rectangles", [])]
        excluded = build_exclusion_mask(image_size, rectangles, active_area=active_area)
        base_homography = anchor_homography(anchor)
        anchor_field = np.asarray([(point.x, point.y) for point in anchor.field_points])
        anchor_image = np.asarray([(point.x, point.y) for point in anchor.image_points])
        anchor_error = float(
            np.linalg.norm(_project(anchor_field, base_homography) - anchor_image, axis=1).mean()
        )
        grid = track_grid(frames, excluded, indices.index(anchor.frame), base_homography)
        solution = solve_camera(
            grid.homographies, grid.statuses, grid.snapped, image_size, active_area
        )
        tracked_errors, raw_errors, line_errors = measure_line_errors(
            frames,
            excluded,
            [list(grid.homographies), solution.raw_homographies, list(solution.homographies)],
        )
        camera_records = []
        for position, index in enumerate(indices):
            line_error = float(line_errors[position])
            fit_error = float(solution.fit_errors[position])
            rating = get_tracking_status(line_error)
            camera_records.append(
                CameraFrame(
                    frame=index,
                    state=pose_to_state(
                        solution.parameters[position],
                        solution.focals[position],
                        solution.up_sign,
                    ),
                    confidence=LINE_CONFIDENCE[rating],
                    status="uncertain" if rating == "LOST" else "reliable",
                    reprojection_error_px=None if math.isnan(fit_error) else fit_error,
                    homography=tuple(
                        tuple(float(value) for value in row)
                        for row in solution.homographies[position]
                    ),
                    anchor_source=anchor.frame,
                    line_error_px=None if math.isnan(line_error) else line_error,
                    source=str(solution.sources[position]),
                )
            )
        _write_jsonl(output_dir / "camera.jsonl", camera_records)
        video = VideoInfo.model_validate(config["video"])
        export_pose_json(
            output_dir / "camera-pose.json",
            config["input"],
            image_size,
            video.fps_numerator / video.fps_denominator,
            indices,
            solution,
        )
        (output_dir / "camera-summary.json").write_text(
            json.dumps(
                {
                    "seed_anchor_reprojection_error_px": anchor_error,
                    "motion_model": "Optical-flow points on the grass, with ORB frame matching for fast moves, snapped onto the painted yard lines",
                    "camera_model": "Pinhole camera solved on every frame and smoothed over time",
                    "reliable": sum(record.status == "reliable" for record in camera_records),
                    "uncertain": sum(record.status == "uncertain" for record in camera_records),
                    "active_area": list(active_area),
                    "snapped_fraction": float(grid.snapped.mean()),
                    "sources": {
                        name: int((solution.sources == name).sum())
                        for name in ("snapped", "predicted", "filled")
                    },
                    "lens_changes": solution.lens_changes,
                    "fov_degrees": [float(solution.fovs.min()), float(solution.fovs.max())],
                    "up_sign": int(solution.up_sign),
                    "line_error_px": {
                        "tracked_grid": _describe_errors(tracked_errors),
                        "camera": _describe_errors(raw_errors),
                        "smoothed_camera": _describe_errors(line_errors),
                    },
                    "report": describe_solution(
                        solution, tracked_errors, raw_errors, line_errors
                    ),
                },
                indent=2,
            )
        )

    def mosaic(self, output_dir: Path, config: dict) -> None:
        run_dir = Path(config["run_dir"])
        prepared = RunManifest.model_validate_json(
            (run_dir / "prepare" / "manifest.json").read_text()
        )
        frames = [np.asarray(Image.open(record.path).convert("RGB")) for record in prepared.frames]
        cameras = _read_jsonl(run_dir / "camera" / "camera.jsonl", CameraFrame)
        camera_summary = json.loads((run_dir / "camera" / "camera-summary.json").read_text())
        active_area = camera_summary.get("active_area")
        rectangles = [tuple(item) for item in config.get("hud_rectangles", [])]
        mask = build_exclusion_mask(
            (frames[0].shape[1], frames[0].shape[0]),
            rectangles,
            active_area=tuple(active_area) if active_area else None,
        )
        masks = [mask] * len(frames)
        result = build_mosaic(
            frames,
            cameras,
            masks,
            NFLField.standard(),
            pixels_per_yard=int(config.get("pixels_per_yard", 4)),
            confidence_threshold=float(config.get("camera_confidence_threshold", 0.25)),
        )
        np.savez_compressed(
            output_dir / "mosaic.npz",
            image=result.image,
            coverage=result.coverage,
            sample_count=result.sample_count,
        )
        Image.fromarray(result.image).save(output_dir / "mosaic.png")
        overlay = render_overlay(frames[0], NFLField.standard(), cameras[0], [], [])
        Image.fromarray(overlay).save(output_dir / "overlay.png")
        sample_indices = sorted({0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1})
        tiles = []
        for sample_index in sample_indices:
            rendered = render_overlay(
                frames[sample_index],
                NFLField.standard(),
                cameras[sample_index],
                [],
                [],
            )
            tile = Image.fromarray(rendered).resize((432, 307))
            tiles.append(tile)
        contact_sheet = Image.new("RGB", (864, 614))
        for tile_index, tile in enumerate(tiles):
            contact_sheet.paste(tile, ((tile_index % 2) * 432, (tile_index // 2) * 307))
        contact_sheet.save(output_dir / "overlay-contact-sheet.png")
        minimap = render_minimap(NFLField.standard(), [])
        Image.fromarray(minimap).save(output_dir / "minimap.png")
        report = evaluate_run(cameras, [])
        summary = report.model_dump()
        summary["coverage_fraction"] = float(result.coverage.mean())
        summary.update(camera_summary)
        (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))

    def detect(self, output_dir: Path, config: dict) -> None:
        run_dir = Path(config["run_dir"])
        prepared = RunManifest.model_validate_json(
            (run_dir / "prepare" / "manifest.json").read_text()
        )
        detector = UltralyticsDetector(config["weights"])
        detections = []
        for record in prepared.frames:
            frame = np.asarray(Image.open(record.path).convert("RGB"))
            detections.extend(detector.detect(frame, record.index))
        _write_jsonl(output_dir / "detections.jsonl", detections)

    def track(self, output_dir: Path, config: dict) -> None:
        run_dir = Path(config["run_dir"])
        cameras = _read_jsonl(run_dir / "camera" / "camera.jsonl", CameraFrame)
        detections = _read_jsonl(run_dir / "detect" / "detections.jsonl", DetectionRecord)
        by_frame = {camera.frame: [] for camera in cameras}
        for detection in detections:
            by_frame.setdefault(detection.frame, []).append(detection)
        tracks = track_players(
            [by_frame.get(camera.frame, []) for camera in cameras], cameras, NFLField.standard()
        )
        _write_jsonl(output_dir / "tracks.jsonl", tracks)

    def evaluate(self, output_dir: Path, config: dict) -> None:
        run_dir = Path(config["run_dir"])
        cameras = _read_jsonl(run_dir / "camera" / "camera.jsonl", CameraFrame)
        tracks_path = run_dir / "track" / "tracks.jsonl"
        tracks = _read_jsonl(tracks_path, TrackRecord) if tracks_path.exists() else []
        report = evaluate_run(cameras, tracks)
        (output_dir / "evaluation.json").write_text(report.model_dump_json(indent=2))


def _execute(
    config_path: Path,
    run_dir: Path,
    stage_names: list[str],
    frame_range: str | None,
    input_path: Path | None,
    weights: Path | None = None,
):
    payload = json.loads(config_path.read_text())
    source = input_path.resolve() if input_path else _resolve(config_path, payload["input"])
    anchor = _resolve(config_path, payload["anchor"])
    if not source.is_file():
        raise typer.BadParameter(f"input video does not exist: {source}")
    if not anchor.is_file():
        raise typer.BadParameter(f"anchor does not exist: {anchor}")
    selected = _parse_frame_range(frame_range, payload.get("default_frame_range", "26:86"))
    video = probe_video(source)
    resolved = {
        **payload,
        "run_id": payload.get("run_id", f"{source.stem}-{selected.start}-{selected.stop}"),
        "input": str(source),
        "anchor": str(anchor),
        "video": video.model_dump(),
        "run_dir": str(run_dir.resolve()),
        "frame_range": [selected.start, selected.stop],
        "stages": stage_names,
    }
    configured_weights = weights or (
        _resolve(config_path, payload["weights"]) if payload.get("weights") else None
    )
    if configured_weights is not None:
        resolved["weights"] = str(configured_weights.resolve())
    if "detect" in stage_names and configured_weights is None:
        raise typer.BadParameter(
            "detect requires --weights or a weights path in the run config"
        )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as temporary:
        json.dump(resolved, temporary, sort_keys=True)
        resolved_path = Path(temporary.name)
    try:
        return run_pipeline(resolved_path, run_dir, runner=ReconstructionRunner())
    finally:
        resolved_path.unlink(missing_ok=True)


@app.command("run")
def run_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="Run config JSON."),
    run_dir: Path = typer.Option(..., help="Artifact directory."),
    stages: str = typer.Option(
        "prepare,camera,mosaic",
        help="Comma-separated stages: prepare,camera,mosaic,detect,track,evaluate.",
    ),
    frame_range: str | None = typer.Option(None, help="Half-open frame range START:END."),
    input_path: Path | None = typer.Option(None, "--input", help="Override input video path."),
    weights: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Detector weights for the detect stage."
    ),
    dry_run: bool = typer.Option(False, help="Resolve and validate without writing."),
) -> None:
    """Run prepare, camera, mosaic, detect, track, and/or evaluate stages."""

    payload = json.loads(config.read_text())
    source = input_path.resolve() if input_path else _resolve(config, payload["input"])
    anchor = _resolve(config, payload["anchor"])
    stage_names = [stage.strip() for stage in stages.split(",") if stage.strip()]
    allowed = {"prepare", "camera", "mosaic", "detect", "track", "evaluate"}
    unknown = set(stage_names) - allowed
    if unknown:
        raise typer.BadParameter(f"unknown stages: {', '.join(sorted(unknown))}")
    if dry_run:
        if not source.is_file():
            raise typer.BadParameter(f"input video does not exist: {source}")
        if not anchor.is_file():
            raise typer.BadParameter(f"anchor does not exist: {anchor}")
        selected = _parse_frame_range(frame_range, payload.get("default_frame_range", "26:86"))
        typer.echo(f"input={source}")
        typer.echo(f"anchor={anchor}")
        typer.echo(f"frames={selected.start}:{selected.stop}")
        typer.echo(f"stages={','.join(stage_names)}")
        return
    manifest = _execute(config, run_dir, stage_names, frame_range, input_path, weights)
    for name in stage_names:
        record = manifest.stages[name]
        typer.echo(f"{name}: {record.status} -> {record.output_path}")
    camera_summary = run_dir / "camera" / "camera-summary.json"
    if "camera" in stage_names and camera_summary.exists():
        summary = json.loads(camera_summary.read_text())
        camera_records = _read_jsonl(run_dir / "camera" / "camera.jsonl", CameraFrame)
        typer.echo(
            "camera frames: "
            f"reliable={summary['reliable']} "
            f"uncertain={summary['uncertain']} "
            f"missing={sum(record.status == 'missing' for record in camera_records)}"
        )
        for line in summary.get("report", []):
            typer.echo(line)
    mosaic_evaluation = run_dir / "mosaic" / "evaluation.json"
    if "mosaic" in stage_names and mosaic_evaluation.exists():
        temporary = run_dir / ".evaluation.json.tmp"
        temporary.write_text(mosaic_evaluation.read_text())
        temporary.replace(run_dir / "evaluation.json")


def _single_stage(name: str, config: Path, run_dir: Path, frame_range: str | None):
    manifest = _execute(config, run_dir, [name], frame_range, None)
    typer.echo(f"{name}: {manifest.stages[name].status}")


@app.command("prepare")
def prepare_command(config: Path = typer.Option(...), run_dir: Path = typer.Option(...), frame_range: str | None = typer.Option(None)):
    """Decode deterministic source frames."""
    _single_stage("prepare", config, run_dir, frame_range)


@app.command("fit-camera")
def fit_camera_command(config: Path = typer.Option(...), run_dir: Path = typer.Option(...), frame_range: str | None = typer.Option(None)):
    """Estimate camera motion from a manual field anchor."""
    _single_stage("camera", config, run_dir, frame_range)


@app.command("build-mosaic")
def build_mosaic_command(config: Path = typer.Option(...), run_dir: Path = typer.Option(...), frame_range: str | None = typer.Option(None)):
    """Warp reliable observations into the static field plane."""
    _single_stage("mosaic", config, run_dir, frame_range)


@app.command("detect")
def detect_command(run_dir: Path = typer.Option(...), weights: Path = typer.Option(..., exists=True, dir_okay=False)):
    """Run an optional trained Ultralytics player detector."""
    prepared = RunManifest.model_validate_json(
        (run_dir / "prepare" / "manifest.json").read_text()
    )
    detector = UltralyticsDetector(weights)
    output_dir = run_dir / "detect"
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = []
    for record in prepared.frames:
        frame = np.asarray(Image.open(record.path).convert("RGB"))
        detections.extend(detector.detect(frame, record.index))
    _write_jsonl(output_dir / "detections.jsonl", detections)
    typer.echo(f"wrote {len(detections)} detections")


@app.command("track")
def track_command(run_dir: Path = typer.Option(...)):
    """Associate detections and project them into field coordinates."""
    cameras = _read_jsonl(run_dir / "camera" / "camera.jsonl", CameraFrame)
    detections = _read_jsonl(
        run_dir / "detect" / "detections.jsonl", DetectionRecord
    )
    by_frame = {camera.frame: [] for camera in cameras}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)
    tracks = track_players(
        [by_frame.get(camera.frame, []) for camera in cameras],
        cameras,
        NFLField.standard(),
    )
    output_dir = run_dir / "track"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "tracks.jsonl", tracks)
    typer.echo(f"wrote {len(tracks)} track records")


@app.command("evaluate")
def evaluate_command(run_dir: Path = typer.Option(...)):
    """Compute camera, detector, and tracking diagnostics."""
    cameras = _read_jsonl(run_dir / "camera" / "camera.jsonl", CameraFrame)
    tracks_path = run_dir / "track" / "tracks.jsonl"
    tracks = _read_jsonl(tracks_path, TrackRecord) if tracks_path.exists() else []
    report = evaluate_run(cameras, tracks)
    output_dir = run_dir / "evaluate"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(report.model_dump_json(indent=2))
    typer.echo(report.model_dump_json(indent=2))


@app.command("export-labels")
def export_labels_command(run_dir: Path = typer.Option(...), output: Path = typer.Option(...)):
    """Export recorded detections as a YOLO training dataset."""
    labels_path = run_dir / "detect" / "detections.jsonl"
    labels = _read_jsonl(labels_path, DetectionRecord)
    export_yolo_dataset(labels, run_dir / "prepare" / "frames", output)
    typer.echo(f"exported labels to {output}")


if __name__ == "__main__":
    app()
