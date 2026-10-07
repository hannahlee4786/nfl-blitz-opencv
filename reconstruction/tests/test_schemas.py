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
