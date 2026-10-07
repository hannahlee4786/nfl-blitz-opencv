"""Command-line entry points for the independent reconstruction workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage
import typer

from .annotations import anchor_homography, load_anchor
from .detection import UltralyticsDetector, export_yolo_dataset
from .evaluation import evaluate_run
from .evidence import build_exclusion_mask, field_evidence
from .field import NFLField
from .mosaic import build_mosaic
from .pipeline import run_pipeline
from .schemas import (
    CameraFrame,
    CameraState,
    DetectionRecord,
    FrameRecord,
    RunManifest,
    TrackRecord,
)
from .tracking import track_players
from .video import build_manifest, probe_video
from .visualization import render_overlay

app = typer.Typer(help="Reconstruct a static NFL field from a moving-camera clip.")


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


def _phase_motion(previous: np.ndarray, current: np.ndarray, excluded: np.ndarray):
    """Return the image translation from previous to current using phase correlation."""

    def signal(frame):
        gray = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        edges = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
        edges[excluded] = 0
        window = np.outer(np.hanning(edges.shape[0]), np.hanning(edges.shape[1]))
        return (edges - edges.mean()) * window

    first, second = signal(previous), signal(current)
    cross = np.fft.fft2(first) * np.conj(np.fft.fft2(second))
    cross /= np.maximum(np.abs(cross), 1e-9)
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = np.asarray(peak, dtype=float)
    dimensions = np.asarray(correlation.shape)
    shift[shift > dimensions / 2] -= dimensions[shift > dimensions / 2]
    response = float(correlation[peak] / max(correlation.mean(), 1e-9))
    # Phase correlation returns the shift applied to current to align it to previous.
    return float(-shift[1]), float(-shift[0]), response


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _field_samples(field: NFLField) -> np.ndarray:
    return np.concatenate(
        [np.linspace(start, end, 80) for start, end in field.marking_segments()]
    )


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

        rectangles = [tuple(item) for item in config.get("hud_rectangles", [])]
        excluded = build_exclusion_mask(image_size, rectangles)
        cumulative = [(0.0, 0.0)]
        responses = [math.inf]
        for previous, current in zip(frames, frames[1:]):
            dx, dy, response = _phase_motion(previous, current, excluded)
            old_x, old_y = cumulative[-1]
            cumulative.append((old_x + dx, old_y + dy))
            responses.append(response)
        anchor_position = cumulative[indices.index(anchor.frame)]
        base_homography = anchor_homography(anchor)
        anchor_field = np.asarray([(point.x, point.y) for point in anchor.field_points])
        anchor_image = np.asarray([(point.x, point.y) for point in anchor.image_points])
        anchor_error = float(
            np.linalg.norm(_project(anchor_field, base_homography) - anchor_image, axis=1).mean()
        )
        field = NFLField.standard()
        samples = _field_samples(field)
        state = CameraState(
            target_x=55.0,
            target_y=field.width_yards / 2,
            distance=55.0,
            yaw_degrees=0.0,
            pitch_degrees=48.0,
            focal_px=720.0,
        )
        camera_records = []
        for position, response, index, frame in zip(
            cumulative, responses, indices, frames, strict=True
        ):
            dx = position[0] - anchor_position[0]
            dy = position[1] - anchor_position[1]
            transform = np.array(((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0)))
            homography = transform @ base_homography
            evidence = field_evidence(frame, excluded)
            pixels = _project(samples, homography)
            rounded = np.rint(pixels).astype(int)
            valid = (
                np.isfinite(pixels).all(axis=1)
                & (rounded[:, 0] >= 0)
                & (rounded[:, 0] < image_size[0])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < image_size[1])
            )
            error = float(np.median(evidence.distance[rounded[valid, 1], rounded[valid, 0]])) if valid.any() else float(max(image_size))
            confidence = 1.0 if index == anchor.frame else float(
                np.clip(math.exp(-error / 8.0) * min(1.0, response / 15.0), 0.0, 1.0)
            )
            camera_records.append(
                CameraFrame(
                    frame=index,
                    state=state,
                    confidence=confidence,
                    status="reliable" if confidence >= 0.25 else "uncertain",
                    reprojection_error_px=anchor_error if index == anchor.frame else error,
                    homography=tuple(tuple(float(value) for value in row) for row in homography),
                    anchor_source=anchor.frame,
                )
            )
        _write_jsonl(output_dir / "camera.jsonl", camera_records)
        (output_dir / "camera-summary.json").write_text(
            json.dumps(
                {
                    "seed_anchor_reprojection_error_px": anchor_error,
                    "motion_model": "phase-correlation translation composed with manual homography",
                    "reliable": sum(record.status == "reliable" for record in camera_records),
                    "uncertain": sum(record.status == "uncertain" for record in camera_records),
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
        rectangles = [tuple(item) for item in config.get("hud_rectangles", [])]
        masks = [
            build_exclusion_mask((frame.shape[1], frame.shape[0]), rectangles)
            for frame in frames
        ]
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
        report = evaluate_run(cameras, [])
        summary = report.model_dump()
        summary["coverage_fraction"] = float(result.coverage.mean())
        summary.update(
            json.loads((run_dir / "camera" / "camera-summary.json").read_text())
        )
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
    manifest = _execute(config, run_dir, stage_names, frame_range, input_path)
    for name in stage_names:
        record = manifest.stages[name]
        typer.echo(f"{name}: {record.status} -> {record.output_path}")


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
