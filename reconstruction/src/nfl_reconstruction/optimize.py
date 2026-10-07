"""Camera fitting against manual anchors and field evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .camera import camera_visibility, project_field
from .field import NFLField
from .schemas import AnchorSet, CameraState


@dataclass(frozen=True)
class CameraFit:
    state: CameraState
    mean_reprojection_error_px: float
    confidence: float
    losses: dict[str, float]


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
