import numpy as np

from nfl_reconstruction.camera_overview import (
    CAMERA_COLOR,
    FIELD_COLOR,
    PATH_COLOR,
    VIEW_COLOR,
    build_camera_overview,
    project_overview,
    render_camera_overview,
)
from nfl_reconstruction.camera_solve import pose_to_state
from nfl_reconstruction.field import NFLField
from nfl_reconstruction.schemas import CameraFrame
from tests.fixtures import FOCAL, FRAME_SIZE, blitz_homography


def moving_cameras():
    parameters = [
        np.asarray((83.0, 26.67, 15.7, 180.0, 27.0, 0.0)),
        np.asarray((78.0, 27.5, 15.7, 175.0, 27.0, 0.0)),
    ]
    return [
        CameraFrame(
            frame=index,
            state=pose_to_state(values, FOCAL, 1.0),
            confidence=1.0,
            status="reliable",
            homography=tuple(
                tuple(float(value) for value in row)
                for row in blitz_homography(values)
            ),
            line_error_px=0.5,
            source="snapped",
        )
        for index, values in enumerate(parameters)
    ]


def test_overview_orients_camera_below_its_field_target():
    cameras = moving_cameras()

    overview = build_camera_overview(
        cameras, FRAME_SIZE, NFLField.standard(), size=(960, 720)
    )
    camera_pixel, camera_visible = project_overview(
        overview, overview.shadows[:1]
    )
    target = cameras[0].state
    target_pixel, target_visible = project_overview(
        overview, np.asarray([[target.target_x, target.target_y, 0.0]])
    )

    assert camera_visible[0] and target_visible[0]
    assert camera_pixel[0, 1] > target_pixel[0, 1] + 10


def test_overview_draws_complete_field_footprint_camera_and_path():
    overview = build_camera_overview(
        moving_cameras(), FRAME_SIZE, NFLField.standard(), size=(960, 720)
    )

    first = render_camera_overview(overview, 0)
    second = render_camera_overview(overview, 1)

    assert first.shape == (720, 960, 3)
    assert np.count_nonzero(np.all(overview.field_image == FIELD_COLOR, axis=2)) > 5_000
    assert np.any(np.all(first == VIEW_COLOR, axis=2))
    assert np.any(np.all(first == CAMERA_COLOR, axis=2))
    assert np.any(np.all(second == PATH_COLOR, axis=2))
    assert not np.array_equal(first, second)
