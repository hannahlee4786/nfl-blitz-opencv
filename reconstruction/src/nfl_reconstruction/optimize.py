"""Camera fitting against manual anchors and field evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .camera import camera_visibility, field_homography, project_field
from .evidence import EvidenceMap
from .field import NFLField
from .schemas import AnchorSet, CameraFrame, CameraState


@dataclass(frozen=True)
class CameraFit:
    state: CameraState
    mean_reprojection_error_px: float
    confidence: float
    losses: dict[str, float]


@dataclass(frozen=True)
class SequenceFitConfig:
    window_size: int = 45
    overlap: int = 15
    confidence_threshold: float = 0.25
    steps_per_frame: int = 60


def camera_confidence(losses: dict[str, float], visibility: float) -> float:
    reprojection = losses.get("reprojection", math.inf)
    if visibility < 0.05 or not math.isfinite(reprojection):
        return 0.0
    return float(max(0.0, min(1.0, math.exp(-reprojection / 10.0))))


def _parameter_state(initial: CameraState):
    names = (
        "target_x", "target_y", "distance", "yaw_degrees",
        "pitch_degrees", "focal_px", "roll_degrees",
    )
    return {
        name: torch.tensor(
            float(getattr(initial, name)), dtype=torch.float64, requires_grad=True
        )
        for name in names
    }


def _materialize(parameters) -> CameraState:
    return CameraState(
        **{name: float(value.detach()) for name, value in parameters.items()}
    )


def fit_anchor_camera(
    anchor: AnchorSet,
    initial: CameraState,
    image_size: tuple[int, int],
    steps: int = 500,
) -> CameraFit:
    field_points = torch.tensor(
        [(point.x, point.y) for point in anchor.field_points],
        dtype=torch.float64,
    )
    image_points = torch.tensor(
        [(point.x, point.y) for point in anchor.image_points],
        dtype=torch.float64,
    )
    parameters = _parameter_state(initial)
    optimizer = torch.optim.Adam(
        [
            {"params": [parameters["target_x"], parameters["target_y"]], "lr": 0.08},
            {"params": [parameters["distance"]], "lr": 0.08},
            {"params": [parameters["yaw_degrees"], parameters["pitch_degrees"]], "lr": 0.03},
            {"params": [parameters["focal_px"]], "lr": 1.0},
            {"params": [parameters["roll_degrees"]], "lr": 0.01},
        ]
    )
    best = None
    iterations = max(1, steps)
    for iteration in range(iterations):
        optimizer.zero_grad()
        projected = project_field(field_points, parameters, image_size)
        residuals = torch.linalg.vector_norm(projected - image_points, dim=1)
        reprojection_loss = torch.nn.functional.smooth_l1_loss(
            projected, image_points, beta=3.0
        )
        bounds_loss = (
            torch.relu(1.0 - parameters["distance"]) ** 2
            + torch.relu(1.0 - parameters["focal_px"]) ** 2
            + torch.relu(1.0 - parameters["pitch_degrees"]) ** 2
            + torch.relu(parameters["pitch_degrees"] - 89.0) ** 2
        )
        roll_loss = parameters["roll_degrees"] ** 2
        loss = reprojection_loss + 0.001 * bounds_loss + 0.0001 * roll_loss
        mean_error = float(residuals.mean().detach())
        if math.isfinite(mean_error) and (best is None or mean_error < best[0]):
            best = (mean_error, _materialize(parameters), float(loss.detach()))
        if iteration < steps:
            loss.backward()
            optimizer.step()
    if best is None:
        return CameraFit(initial, math.inf, 0.0, {"reprojection": math.inf})
    mean_error, state, total_loss = best
    visibility = camera_visibility(state, NFLField.standard(), image_size)
    losses = {"reprojection": mean_error, "total": total_loss}
    return CameraFit(
        state=state,
        mean_reprojection_error_px=mean_error,
        confidence=camera_confidence(losses, visibility),
        losses=losses,
    )


def _interpolate_state(
    frame: int,
    anchors: dict[int, CameraState],
) -> tuple[CameraState, int]:
    indices = sorted(anchors)
    if not indices:
        raise ValueError("Camera sequence fitting requires at least one anchor.")
    before = max((index for index in indices if index <= frame), default=indices[0])
    after = min((index for index in indices if index >= frame), default=indices[-1])
    if before == after:
        return anchors[before], before
    ratio = (frame - before) / (after - before)
    values = {}
    for name in (
        "target_x", "target_y", "distance", "yaw_degrees",
        "pitch_degrees", "focal_px", "roll_degrees",
    ):
        start = getattr(anchors[before], name)
        end = getattr(anchors[after], name)
        values[name] = start + ratio * (end - start)
    nearest = before if frame - before <= after - frame else after
    return CameraState(**values), nearest


def _marking_samples(field: NFLField) -> torch.Tensor:
    points = []
    for start, end in field.marking_segments():
        points.extend(np.linspace(start, end, 80))
    return torch.tensor(np.asarray(points), dtype=torch.float64)


def _evidence_error(
    evidence: EvidenceMap,
    state: CameraState,
) -> tuple[float, float]:
    height, width = evidence.distance.shape
    if evidence.line_likelihood.max() <= 0:
        return float(max(width, height)), 0.0
    pixels = project_field(
        _marking_samples(NFLField.standard()), state, (width, height)
    ).detach().numpy()
    rounded = np.rint(pixels).astype(int)
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    if valid.sum() < 4:
        return float(max(width, height)), 0.0
    error = float(
        np.median(evidence.distance[rounded[valid, 1], rounded[valid, 0]])
    )
    confidence = math.exp(-error / 5.0)
    return error, confidence


def merge_window_results(
    windows: list[list[CameraFrame]],
    overlap: int,
) -> list[CameraFrame]:
    del overlap
    by_frame: dict[int, list[CameraFrame]] = {}
    for window in windows:
        for record in window:
            by_frame.setdefault(record.frame, []).append(record)
    merged = []
    for frame in sorted(by_frame):
        candidates = by_frame[frame]
        best = max(candidates, key=lambda item: item.confidence)
        merged.append(best)
    return merged


def fit_camera_sequence(
    evidence: list[EvidenceMap],
    anchors: dict[int, CameraState],
    config: SequenceFitConfig = SequenceFitConfig(),
) -> list[CameraFrame]:
    if config.window_size <= config.overlap:
        raise ValueError("Camera window size must be greater than overlap.")
    records = []
    for frame, frame_evidence in enumerate(evidence):
        state, anchor_source = _interpolate_state(frame, anchors)
        error, confidence = _evidence_error(frame_evidence, state)
        status = (
            "reliable" if confidence >= config.confidence_threshold else "uncertain"
        )
        homography = field_homography(
            state,
            (frame_evidence.distance.shape[1], frame_evidence.distance.shape[0]),
        ).detach().numpy()
        records.append(
            CameraFrame(
                frame=frame,
                state=state,
                confidence=confidence,
                status=status,
                reprojection_error_px=error,
                homography=tuple(tuple(float(value) for value in row) for row in homography),
                anchor_source=anchor_source,
            )
        )
    stride = config.window_size - config.overlap
    windows = [
        records[start : start + config.window_size]
        for start in range(0, len(records), stride)
    ]
    return merge_window_results(windows, config.overlap)
