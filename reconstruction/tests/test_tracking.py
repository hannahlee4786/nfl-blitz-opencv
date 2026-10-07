from nfl_reconstruction.field import NFLField
from nfl_reconstruction.schemas import CameraFrame, DetectionRecord
from nfl_reconstruction.tracking import (
    TrackingConfig,
    project_detection,
    track_players,
)
from tests.fixtures import synthetic_state


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def camera(frame, confidence=1.0):
    return CameraFrame(
        frame=frame,
        state=synthetic_state(),
        confidence=confidence,
        status="reliable" if confidence >= 0.25 else "uncertain",
        homography=IDENTITY,
    )


def detection(frame, x, team="offense", identifier=None):
    return DetectionRecord(
        frame=frame,
        detection_id=identifier or f"{frame}-{team}",
        team=team,
        confidence=0.9,
        x1=x - 5, y1=15, x2=x + 5, y2=35,
        ground_x=x, ground_y=35,
    )


def test_track_survives_two_frame_occlusion():
    detections = [
        [detection(0, 20)],
        [detection(1, 22)],
        [],
        [],
        [detection(4, 28)],
    ]
    cameras = [camera(index) for index in range(5)]

    tracks = track_players(
        detections,
        cameras,
        NFLField.standard(),
        TrackingConfig(max_staleness=3),
    )

    visible_ids = {record.track_id for record in tracks if record.detection_id}
    assert len(visible_ids) == 1
    assert len([record for record in tracks if record.projection_status == "prediction"]) == 2


def test_low_camera_confidence_disables_field_projection():
    projected = project_detection(
        detection(0, 20), camera(0, confidence=0.1), NFLField.standard()
    )

    assert projected.field_x is None
    assert projected.field_y is None
    assert projected.projection_status == "camera_uncertain"


def test_team_mismatch_prevents_association_when_players_cross():
    detections = [
        [detection(0, 20, "offense"), detection(0, 40, "defense")],
        [detection(1, 38, "offense"), detection(1, 22, "defense")],
    ]
    tracks = track_players(
        detections,
        [camera(0), camera(1)],
        NFLField.standard(),
        TrackingConfig(max_center_distance=30),
    )
    offense = {record.track_id for record in tracks if record.team == "offense"}
    defense = {record.track_id for record in tracks if record.team == "defense"}

    assert offense.isdisjoint(defense)
    assert len(offense) == len(defense) == 1


def test_out_of_field_projection_is_flagged_not_clamped():
    projected = project_detection(
        detection(0, 130), camera(0), NFLField.standard()
    )

    assert projected.field_x == 130
    assert projected.projection_status == "out_of_bounds"
