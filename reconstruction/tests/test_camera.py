import pytest
import torch

from nfl_reconstruction.camera import (
    camera_matrix,
    camera_visibility,
    field_homography,
    project_field,
    unproject_image,
)
from nfl_reconstruction.field import NFLField
from tests.fixtures import synthetic_state


def test_project_then_unproject_recovers_field_points():
    state = synthetic_state()
    points = torch.tensor(
        [[50.0, 20.0], [55.0, 30.0], [60.0, 25.0]],
        dtype=torch.float64,
    )
    homography = field_homography(state, (864, 614))
    pixels = project_field(points, state, (864, 614))
    recovered = unproject_image(pixels, homography)

    torch.testing.assert_close(recovered, points, atol=1e-3, rtol=1e-4)


def test_camera_below_field_has_zero_visibility():
    state = synthetic_state(pitch_degrees=-20.0)

    assert camera_visibility(state, NFLField.standard(), (864, 614)) == 0.0


def test_camera_intrinsics_use_image_center():
    matrix = camera_matrix(synthetic_state(), (864, 614))

    assert matrix[0, 0].item() == pytest.approx(720.0)
    assert matrix[1, 1].item() == pytest.approx(720.0)
    assert matrix[0, 2].item() == pytest.approx(431.5)
    assert matrix[1, 2].item() == pytest.approx(306.5)


def test_projection_is_differentiable():
    points = torch.tensor([[55.0, 26.0]], dtype=torch.float64, requires_grad=True)

    project_field(points, synthetic_state(), (864, 614)).sum().backward()

    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
