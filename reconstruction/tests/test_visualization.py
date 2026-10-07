import numpy as np

from nfl_reconstruction.field import NFLField
from nfl_reconstruction.schemas import CameraFrame, DetectionRecord, TrackRecord
from nfl_reconstruction.visualization import render_minimap, render_overlay
from tests.fixtures import synthetic_state


def test_overlay_uses_uncertain_camera_color():
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    camera = CameraFrame(
        frame=0, state=synthetic_state(), confidence=0.1, status="uncertain",
        homography=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )

    rendered = render_overlay(frame, NFLField.standard(), camera, [], [])

    assert rendered.shape == frame.shape
    assert ((rendered[..., 0] > 200) & (rendered[..., 1] > 150)).any()


def test_minimap_draws_player_at_field_position():
    track = TrackRecord(
        frame=0, track_id=1, detection_id="d", team="offense", confidence=1,
        screen_x=0, screen_y=0, field_x=60, field_y=26.666,
        camera_confidence=1, projection_status="reliable",
    )

    image = render_minimap(NFLField.standard(), [track], size=(600, 267))

    assert image.shape == (267, 600, 3)
    assert image[133, 300, 0] > 150

