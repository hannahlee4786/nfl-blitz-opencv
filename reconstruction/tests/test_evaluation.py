import pytest

from nfl_reconstruction.evaluation import evaluate_run
from nfl_reconstruction.schemas import CameraFrame, TrackRecord
from tests.fixtures import synthetic_state


def test_evaluation_reports_reprojection_detection_and_fragmentation():
    cameras = [
        CameraFrame(
            frame=index,
            state=synthetic_state(target_x=50 + index),
            confidence=1,
            reprojection_error_px=error,
        )
        for index, error in enumerate((1.0, 2.0, 3.0, 4.0))
    ]
    tracks = [
        TrackRecord(
            frame=index, track_id=index % 2, detection_id=f"d-{index}",
            team="offense", confidence=1, screen_x=10, screen_y=20,
            field_x=20, field_y=20, camera_confidence=1,
            projection_status="reliable",
        )
        for index in range(4)
    ]

    report = evaluate_run(
        cameras,
        tracks,
        annotated_player_count=10,
        matched_detection_count=9,
        expected_track_count=1,
    )

    assert report.median_reprojection_error_px == pytest.approx(2.5)
    assert report.detection_recall == pytest.approx(0.9)
    assert report.track_fragments == 1


def test_evaluation_counts_out_of_bounds_tracks():
    tracks = [
        TrackRecord(
            frame=0, track_id=0, detection_id="d", team="defense",
            confidence=1, screen_x=10, screen_y=20, field_x=130, field_y=20,
            camera_confidence=1, projection_status="out_of_bounds",
        )
    ]

    report = evaluate_run([], tracks)

    assert report.out_of_bounds_rate == 1.0



def test_evaluation_reports_line_error_and_lost_frames():
    cameras = [
        CameraFrame(
            frame=index,
            state=synthetic_state(),
            confidence=1,
            line_error_px=error,
            source="snapped",
        )
        for index, error in enumerate((0.0, 1.0, 2.0, 5.0, None))
    ]

    report = evaluate_run(cameras, [])

    assert report.median_line_error_px == pytest.approx(1.5)
    assert report.lost_fraction == pytest.approx(0.4)


def test_evaluation_without_line_scores_leaves_them_empty():
    cameras = [CameraFrame(frame=0, state=synthetic_state(), confidence=1)]

    report = evaluate_run(cameras, [])

    assert report.median_line_error_px is None
    assert report.lost_fraction is None
