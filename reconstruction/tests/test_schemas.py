import math

import pytest
from pydantic import ValidationError

from nfl_reconstruction.schemas import CameraFrame, CameraState, FieldPoint


def test_camera_frame_requires_explicit_confidence():
    with pytest.raises(ValidationError, match="confidence"):
        CameraFrame.model_validate(
            {
                "schema_version": 1,
                "frame": 0,
                "state": {
                    "target_x": 50,
                    "target_y": 26,
                    "distance": 40,
                    "yaw_degrees": 0,
                    "pitch_degrees": 45,
                    "focal_px": 700,
                },
            }
        )


def test_camera_frame_rejects_out_of_range_confidence():
    state = CameraState(
        target_x=50,
        target_y=26,
        distance=40,
        yaw_degrees=0,
        pitch_degrees=45,
        focal_px=700,
    )

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        CameraFrame(frame=0, state=state, confidence=1.01)


def test_coordinate_models_reject_non_finite_values():
    with pytest.raises(ValidationError, match="finite"):
        FieldPoint(x=math.inf, y=20)


def test_schema_models_are_immutable():
    point = FieldPoint(x=10, y=20)

    with pytest.raises(ValidationError, match="frozen"):
        point.x = 11


def test_camera_state_up_sign_defaults_to_up_and_rejects_other_values():
    values = {
        "target_x": 50,
        "target_y": 26,
        "distance": 40,
        "yaw_degrees": 0,
        "pitch_degrees": 45,
        "focal_px": 700,
    }

    assert CameraState(**values).up_sign == 1
    assert CameraState(**values, up_sign=-1).up_sign == -1
    with pytest.raises(ValidationError, match="up_sign"):
        CameraState(**values, up_sign=0)


def test_camera_frame_rejects_unknown_grid_source():
    state = CameraState(
        target_x=50,
        target_y=26,
        distance=40,
        yaw_degrees=0,
        pitch_degrees=45,
        focal_px=700,
    )

    with pytest.raises(ValidationError, match="source"):
        CameraFrame(frame=0, state=state, confidence=1, source="guessed")
