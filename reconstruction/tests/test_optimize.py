import numpy as np
import torch

from nfl_reconstruction.camera import project_field
from nfl_reconstruction.optimize import fit_anchor_camera
from nfl_reconstruction.schemas import AnchorSet, FieldPoint, ImagePoint
from tests.fixtures import synthetic_state


def synthetic_anchor():
    state = synthetic_state()
    field = torch.tensor(
        [[45.0, 20.0], [45.0, 32.0], [55.0, 20.0], [55.0, 32.0], [65.0, 20.0], [65.0, 32.0]],
        dtype=torch.float64,
    )
    image = project_field(field, state, (864, 614)).detach().numpy()
    return AnchorSet(
        frame=0,
        image_width=864,
        image_height=614,
        field_points=tuple(FieldPoint(x=x, y=y) for x, y in field.numpy()),
        image_points=tuple(ImagePoint(x=x, y=y) for x, y in image),
    )


def test_anchor_fit_recovers_perturbed_camera():
    expected = synthetic_state()
    start = expected.model_copy(
        update={"target_x": expected.target_x + 4.0, "focal_px": 700.0}
    )

    result = fit_anchor_camera(synthetic_anchor(), start, (864, 614), steps=300)

    assert result.mean_reprojection_error_px < 1.0
    assert result.confidence > 0.9
    assert np.isfinite(list(result.losses.values())).all()


def test_anchor_fit_marks_below_field_camera_unreliable():
    result = fit_anchor_camera(
        synthetic_anchor(),
        synthetic_state(pitch_degrees=-20.0),
        (864, 614),
        steps=0,
    )

    assert result.confidence == 0.0
