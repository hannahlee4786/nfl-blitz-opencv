"""Differentiable pinhole geometry for the constrained virtual camera."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .field import NFLField
from .schemas import CameraState

TensorState = CameraState | Mapping[str, torch.Tensor | float]


def _value(state: TensorState, name: str) -> torch.Tensor:
    value = state[name] if isinstance(state, Mapping) else getattr(state, name)
    return torch.as_tensor(value, dtype=torch.float64)


def _up_sign(state: TensorState) -> float:
    if isinstance(state, Mapping):
        return float(state.get("up_sign", 1))
    return float(state.up_sign)


def camera_matrix(
    state: TensorState,
    image_size: tuple[int, int],
) -> torch.Tensor:
    width, height = image_size
    focal = _value(state, "focal_px")
    zero = torch.zeros_like(focal)
    one = torch.ones_like(focal)
    return torch.stack(
        [
            torch.stack([focal, zero, zero + (width - 1) / 2]),
            torch.stack([zero, focal, zero + (height - 1) / 2]),
            torch.stack([zero, zero, one]),
        ]
    )


def _camera_center(state: TensorState) -> torch.Tensor:
    distance = _value(state, "distance")
    yaw = torch.deg2rad(_value(state, "yaw_degrees"))
    pitch = torch.deg2rad(_value(state, "pitch_degrees"))
    horizontal = distance * torch.cos(pitch)
    return torch.stack(
        [
            _value(state, "target_x") + horizontal * torch.sin(yaw),
            _value(state, "target_y") + horizontal * torch.cos(yaw),
            _up_sign(state) * distance * torch.sin(pitch),
        ]
    )


def world_to_camera(state: TensorState) -> torch.Tensor:
    center = _camera_center(state)
    target = torch.stack(
        [
            _value(state, "target_x"),
            _value(state, "target_y"),
            torch.zeros((), dtype=torch.float64),
        ]
    )
    forward = torch.nn.functional.normalize(target - center, dim=0)
    world_up = torch.tensor([0.0, 0.0, _up_sign(state)], dtype=torch.float64)
    right = torch.nn.functional.normalize(
        torch.linalg.cross(forward, world_up), dim=0
    )
    down = -torch.linalg.cross(right, forward)
    roll = torch.deg2rad(_value(state, "roll_degrees"))
    rolled_right = torch.cos(roll) * right + torch.sin(roll) * down
    rolled_down = -torch.sin(roll) * right + torch.cos(roll) * down
    rotation = torch.stack([rolled_right, rolled_down, forward])
    translation = -(rotation @ center)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    return torch.cat([torch.cat([rotation, translation[:, None]], dim=1), bottom])


def field_homography(
    state: TensorState,
    image_size: tuple[int, int],
) -> torch.Tensor:
    projection = camera_matrix(state, image_size) @ world_to_camera(state)[:3]
    homography = projection[:, [0, 1, 3]]
    determinant = torch.linalg.det(homography)
    if not torch.isfinite(homography).all() or abs(determinant.item()) < 1e-12:
        raise ValueError("Camera state produces a singular homography.")
    return homography / homography[2, 2]


def _transform_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    points = points.to(dtype=torch.float64)
    homogeneous = torch.cat(
        [points, torch.ones((*points.shape[:-1], 1), dtype=points.dtype)],
        dim=-1,
    )
    projected = homogeneous @ matrix.T
    return projected[..., :2] / projected[..., 2:3]


def project_field(
    points: torch.Tensor,
    state: TensorState,
    image_size: tuple[int, int],
) -> torch.Tensor:
    return _transform_points(points, field_homography(state, image_size))


def unproject_image(
    points: torch.Tensor,
    homography: torch.Tensor,
) -> torch.Tensor:
    return _transform_points(points, torch.linalg.inv(homography))


def camera_visibility(
    state: TensorState,
    field: NFLField,
    image_size: tuple[int, int],
) -> float:
    if _up_sign(state) * _camera_center(state)[2].item() <= 0:
        return 0.0
    width, height = image_size
    xs = torch.linspace(0.0, field.length_yards, 13, dtype=torch.float64)
    ys = torch.linspace(0.0, field.width_yards, 7, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    pixels = project_field(points, state, image_size)
    visible = (
        torch.isfinite(pixels).all(dim=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return float(visible.to(torch.float64).mean())
